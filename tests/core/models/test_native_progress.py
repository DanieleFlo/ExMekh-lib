"""Progress negotiation, real socket liveness and backend/model-neutral metadata."""

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest
from openai import APIStatusError, OpenAI
from PIL import Image

from exmekh_lib.core.models.native_suite import NativeSuiteClient
from exmekh_lib.interfaces.operation_progress import (
    OperationProgress,
    operation_progress_context,
    operation_progress_observer,
)


def batch():
    raw = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(raw, "PNG")
    return {
        "created": 1,
        "ds4": {},
        "data": [
            {
                "b64_json": base64.b64encode(raw.getvalue()).decode(),
                "ds4": {
                    "file_id": "file-one",
                    "seed": 17,
                    "width": 32,
                    "height": 32,
                    "recipe": {},
                },
            }
        ],
    }


def event(kind="progress", seq=1, **extra):
    return {
        "version": 1,
        "type": "operation." + kind,
        "operation_id": "op-one",
        "request_id": "req-one",
        "seq": seq,
        "elapsed_seconds": 0.1,
        "state": "running",
        "phase": "generation",
        "unit": "token",
        "completed_units": 7,
        "total_units": None,
        "overall_fraction": None,
        "overall_kind": "indeterminate",
        **extra,
    }


def frame(payload):
    return (
        "event: " + payload["type"] + "\ndata: " + json.dumps(payload) + "\n\n"
    ).encode()


class Fragments(httpx2.SyncByteStream):
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def __iter__(self):
        for start in range(0, len(self.payload), 7):
            yield self.payload[start : start + 7]

    def close(self):
        self.closed = True


def test_stream_negotiation_non_qwen_tokens_unknown_total_and_original_png():
    stream = Fragments(
        frame(event("started"))
        + frame(event(seq=2))
        + frame(event("completed", 3, state="completed", result=batch()))
    )
    model = "future/autoregressive"
    calls = []

    def respond(request):
        calls.append(request)
        if request.method == "GET":
            return httpx2.Response(
                200,
                json={
                    "data": [
                        {
                            "id": model,
                            "ds4": {
                                "version": 1,
                                "revision": "mock",
                                "profile": "bf16-native-v1",
                                "operations": ["image_generate"],
                                "input_modalities": ["text"],
                                "configured": True,
                                "resident": False,
                                "ready": True,
                                "image_progress_version": 1,
                            },
                        }
                    ]
                },
            )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )

    seen = []
    with OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as transport:
        client = NativeSuiteClient(transport, image_model=model)
        [entry] = client.models()
        assert entry.id == model and entry.ds4.image_progress_version == 1
        result = client.images("red", width=32, height=32, on_progress=seen.append)
    assert json.loads(calls[1].content)["model"] == model
    assert calls[1].headers["Accept"] == "text/event-stream"
    assert calls[1].headers["Idempotency-Key"]
    assert [item.seq for item in seen] == [1, 2, 3]
    assert seen[1].unit == "token" and seen[1].total_units is None
    assert result.data[0].png_bytes() == base64.b64decode(
        batch()["data"][0]["b64_json"]
    )
    assert "result" not in seen[-1].model_dump()
    assert stream.closed and len(calls) == 2


@pytest.mark.parametrize("code", [409, 503])
def test_failure_inside_stream_preserves_cause_and_never_retries(code):
    cause = {
        "error": {
            "message": "operation cancelled"
            if code == 409
            else "insufficient resources",
            "code": "request_cancelled" if code == 409 else "insufficient_resources",
        }
    }
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=frame(
                event("failed", state="failed", status_code=code, failure=cause)
            ),
        )

    with OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as transport:
        with pytest.raises(APIStatusError) as error:
            NativeSuiteClient(transport).images(
                "red", width=32, height=32, on_progress=lambda _: None
            )
    assert error.value.status_code == code and error.value.body == cause
    assert error.value.response.json() == cause
    assert error.value.response.headers["content-type"] == "application/json"
    assert error.value.request_id == "req-one"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        frame(event()),
        frame(event(seq=2)) + frame(event(seq=1)),
        frame(event()) + frame(event(seq=2, operation_id="other")),
        frame(event()) + frame(event(seq=2, request_id="other")),
        frame(event("completed", state="running", result=batch())),
        frame(event("completed", state="completed")),
        frame(event("completed", state="completed", result={**batch(), "data": []})),
        frame(event("failed", state="failed", failure={})),
        frame(event("failed", state="failed", status_code=200, failure={})),
        frame(event("failed", state="failed", status_code=503, failure="bad payload")),
    ],
)
def test_incomplete_or_invalid_stream_does_not_report_success(payload):
    seen = []

    def respond(request):
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=payload
        )

    with OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as transport:
        with pytest.raises(ValueError):
            NativeSuiteClient(transport).images(
                "red", width=32, height=32, on_progress=seen.append
            )
    assert all(snapshot.state == "running" for snapshot in seen)


def test_older_json_server_is_compatible_with_negotiation():
    def respond(request):
        return httpx2.Response(200, json=batch())

    seen = []
    with OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as transport:
        assert (
            NativeSuiteClient(transport)
            .images("red", width=32, height=32, on_progress=seen.append)
            .data
        )
    assert not seen


def test_batch_progress_is_shared_only_by_active_items():
    snapshot = OperationProgress.model_validate(
        event(
            item_index=1,
            item_count=4,
            active_items=2,
            overall_fraction=0.5,
            overall_kind="estimated",
        )
    )
    assert [snapshot.item_fraction(i) for i in range(4)] == [1, 0.5, 0.5, None]
    completed = OperationProgress.model_validate(
        event(
            "completed",
            state="completed",
            item_count=4,
        )
    )
    assert [completed.item_fraction(i) for i in (-1, 0, 3, 4)] == [None, 1, 1, None]
    with pytest.raises(ValueError):
        OperationProgress.model_validate(
            event(item_index=1, item_count=2, active_items=2)
        )


def test_callback_context_is_nested_and_restored_after_failure():
    assert operation_progress_observer() is None
    with operation_progress_context(lambda _: None):
        outer = operation_progress_observer()
        with pytest.raises(RuntimeError), operation_progress_context(None):
            assert operation_progress_observer() is None
            raise RuntimeError("failure")
        assert operation_progress_observer() is outer
    assert operation_progress_observer() is None


def test_heartbeats_keep_real_socket_read_alive_beyond_read_timeout():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for seq in range(1, 7):
                self.wfile.write(frame(event("heartbeat", seq)))
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(
                frame(event("completed", 7, state="completed", result=batch()))
            )
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        with OpenAI(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="test",
            timeout=httpx2.Timeout(0.2, connect=2),
        ) as transport:
            seen = []
            start = time.monotonic()
            result = NativeSuiteClient(transport).images(
                "red", width=32, height=32, on_progress=seen.append
            )
            assert time.monotonic() - start > 0.2
        assert len(seen) == 7 and result.data
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
