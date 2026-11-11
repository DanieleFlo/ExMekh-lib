"""Native media transport, explicit tools and attachment behavior."""

import base64
import inspect
import io
import json
from contextlib import nullcontext
from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import MagicMock

import httpx2
import pytest
from openai import OpenAI
from PIL import Image
from pydantic import TypeAdapter

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.native_suite import NATIVE_SUITE, ImageDimension
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.tools.deep_tools import DeepTools
from exmekh_lib.core.cache.user_message_codec import user_message_image_urls
from exmekh_lib.core.models import native_suite
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.native_media_tools import (
    generate_images,
    workspace_media,
)
from exmekh_lib.core.models.native_suite import (
    EmbeddingItem,
    MediaContent,
    NativeSuiteClient,
    TextContent,
    native_suite_client,
    native_suite_context,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.interfaces.provider import Provider


def transport(handler):
    return NativeSuiteClient(
        OpenAI(
            base_url="http://native/v1",
            api_key="test",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
    )


def image_response(width=32, height=32, *, n=1, raw=b""):
    return {
        "created": 1,
        "ds4": {},
        "data": [
            {
                "b64_json": base64.b64encode(raw).decode(),
                "ds4": {
                    "file_id": f"file-{i}",
                    "seed": i,
                    "width": width,
                    "height": height,
                    "recipe": {},
                },
            }
            for i in range(n)
        ],
    }


def png_payload(width=32, height=32, *, image_format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(stream, image_format)
    return stream.getvalue()


def test_order_task_and_no_residency_commands():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["input"][0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "video", "file_id": "file-video", "audio": "include"},
            {"type": "text", "text": "last"},
        ]
        assert body["ds4"] == {
            "version": 1,
            "task": "document",
            "overflow": "error",
            "title": "Title",
        }
        return httpx2.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 128}]}
        )

    suite = transport(handler)
    result = suite.embeddings(
        [
            EmbeddingItem(
                content=[
                    TextContent(text="first"),
                    MediaContent(type="video", file_id="file-video", audio="include"),
                    TextContent(text="last"),
                ]
            )
        ],
        dimensions=128,
        task="document",
        title="Title",
    )
    assert len(result["data"]) == 1
    assert [r.url.path for r in requests] == ["/v1/embeddings"]
    with native_suite_context(lambda: suite):
        with native_suite_client() as borrowed:
            assert borrowed is suite


def test_registered_tools_have_their_own_entrypoints(tmp_path):
    tools = DeepTools(
        [
            CAPABILITY_FORMAT.bundled_root / "tools" / name
            for name in ("generate-image",)
        ],
        work_root=tmp_path,
    )
    loaded = tools.load()
    assert not (CAPABILITY_FORMAT.bundled_root / "tools/embed-content/main.py").exists()
    assert len(loaded) == 1
    tools.unload()
    for method in (
        NativeSuiteClient.images,
        NativeSuiteClient.embeddings,
        generate_images,
    ):
        assert "model" not in inspect.signature(method).parameters
    assert "size" not in inspect.signature(NativeSuiteClient.images).parameters
    assert {"width", "height"} <= inspect.signature(
        NativeSuiteClient.images
    ).parameters.keys()


def test_multiple_images_preview_configuration_and_workspace_paths(tmp_path):
    stream = io.BytesIO()
    Image.new("RGBA", (32, 32), "red").save(stream, "PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Idempotency-Key"]
        body = json.loads(request.content)
        assert body["n"] == 2
        return httpx2.Response(
            200,
            json={
                "created": 1,
                "ds4": {"revision": "test", "profile": "bf16"},
                "data": [
                    {
                        "b64_json": encoded,
                        "ds4": {
                            "file_id": f"file-{i}",
                            "seed": 17 + i,
                            "width": 32,
                            "height": 32,
                            "recipe": {"steps": 8},
                        },
                    }
                    for i in range(2)
                ],
            },
        )

    suite = transport(handler)
    with native_suite_context(lambda: suite):
        result = generate_images(
            tmp_path, "red", n=2, width=32, height=32, preview_scale=0.5
        )
        assert isinstance(result, AttachedToolResult)
        parts = list(result.attachments.iter_content_parts())
        assert isinstance(parts[0], str)
        previews = [part for part in parts if isinstance(part, Image.Image)]
        assert [item.size for item in previews] == [(16, 16), (16, 16)]
        for row in result["images"]:
            assert Path(row["path"]).read_bytes() == stream.getvalue()
        no_pixels = generate_images(
            tmp_path, "red", n=2, width=32, height=32, include_image_in_result=False
        )
        assert not isinstance(no_pixels, AttachedToolResult)
    with pytest.raises(ValueError):
        workspace_media(tmp_path, "../escape.png")
    assert all(r.url.path == "/v1/images/generations" for r in requests)


def test_upload_uses_native_file_purpose_and_returns_reference(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF-fixture")

    def handler(request):
        assert request.url.path == "/v1/files"
        assert b"ds4_media" in request.content
        assert b"RIFF-fixture" in request.content
        return httpx2.Response(
            200,
            json={
                "id": "file-audio",
                "object": "file",
                "bytes": 12,
                "created_at": 1,
                "filename": "clip.wav",
                "purpose": "ds4_media",
            },
        )

    media = transport(handler).upload(path, kind="audio")
    assert media.type == "audio" and media.file_id == "file-audio"


@pytest.mark.parametrize("suffix", ["gif", "tiff", "bmp"])
def test_upload_converts_unsupported_raster_formats_to_png(tmp_path, suffix):
    path = tmp_path / f"reference.{suffix}"
    Image.new("RGB", (32, 24), "red").save(path)

    def handler(request):
        assert request.url.path == "/v1/files"
        assert b'filename="reference.png"' in request.content
        assert b"Content-Type: image/png" in request.content
        assert b"\x89PNG" in request.content
        return httpx2.Response(200, json={"id": "file-image"})

    media = transport(handler).upload(path, kind="image")
    assert media.type == "image" and media.file_id == "file-image"


def test_invalid_embedding_shape_is_rejected():
    suite = transport(lambda r: httpx2.Response(200, json={"data": []}))
    with pytest.raises(ValueError):
        suite.embeddings(["text"])


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_native_transport_lifetime_matches_its_owner(monkeypatch, borrowed, fails):
    owner = OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(200, json={"data": []})
            )
        ),
    )
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://native/v1")
    monkeypatch.setattr(native_suite, "OpenAI", lambda **kwargs: owner)
    context = (
        native_suite_context(lambda: NativeSuiteClient(owner))
        if borrowed
        else nullcontext()
    )
    try:
        with context:
            with pytest.raises(RuntimeError, match="stop") if fails else nullcontext():
                with native_suite_client() as suite:
                    assert suite.models() == []
                    if fails:
                        raise RuntimeError("stop")
        assert owner.is_closed() is (not borrowed)
    finally:
        owner.close()


def test_all_reference_paths_are_checked_before_upload(tmp_path):
    path = tmp_path / "reference.png"
    path.write_bytes(b"image")
    requests = []
    suite = transport(lambda request: requests.append(request))
    with native_suite_context(lambda: suite), pytest.raises(ValueError):
        generate_images(tmp_path, "red", images=["reference.png", "../escape.png"])
    assert requests == []


def test_registered_image_dimensions_are_exhaustive_enums(tmp_path):
    tools = DeepTools(
        [CAPABILITY_FORMAT.bundled_root / "tools/generate-image"], work_root=tmp_path
    )
    try:
        tool = tools.load()[0]
        hints = tool.get_type_hints()
        for axis in ("width", "height"):
            schema = TypeAdapter(hints[axis]).json_schema()
            assert schema["type"] == "integer"
            assert schema["enum"] == list(range(32, 2049, 32))
            assert tool.get_work_signature().parameters[axis].default == 2048
            for method in (NativeSuiteClient.images, generate_images):
                assert get_type_hints(method)[axis] == hints[axis]
    finally:
        tools.unload()


def test_all_advertised_image_dimensions_reach_transport():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["size"])
        width, height = map(int, json.loads(request.content)["size"].split("x"))
        return httpx2.Response(200, json=image_response(width, height))

    suite = transport(handler)
    for width in get_args(ImageDimension):
        for height in get_args(ImageDimension):
            suite.images("test", width=width, height=height)
    assert len(sent) == 4096
    assert sent[0] == "32x32" and sent[-1] == "2048x2048"


@pytest.mark.parametrize(
    "arguments",
    [
        {"width": 2560, "height": 1440},
        {"width": 1920, "height": 1080},
        {"width": 0},
        {"height": 31},
        {"width": 1024.0},
        {"height": True},
        {"n": 0},
        {"n": 9},
        {"n": True},
        {"n": 2, "width": 2048, "height": 2048},
        {"output_format": "jpeg"},
        {"background": "invalid"},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"seed": 1.5},
        {"seed": 2**64 - 1, "n": 2, "width": 32, "height": 32},
    ],
)
def test_invalid_image_geometry_rejected_before_transport_or_upload(
    tmp_path, arguments
):
    suite = transport(lambda request: pytest.fail("Invalid request reached transport"))
    with pytest.raises(ValueError):
        suite.images("test", **arguments)
    with native_suite_context(lambda: suite), pytest.raises(ValueError):
        generate_images(tmp_path, "test", images=["missing-reference.png"], **arguments)
    assert not (tmp_path / ".exmekh_lib").exists()


def test_image_default_is_2k_on_client_helper_and_loaded_tool(tmp_path):
    sizes = []

    def handler(request):
        sizes.append(json.loads(request.content)["size"])
        width, height = map(int, sizes[-1].split("x"))
        return httpx2.Response(
            200, json=image_response(width, height, raw=png_payload(width, height))
        )

    suite = transport(handler)
    tools = DeepTools(
        [CAPABILITY_FORMAT.bundled_root / "tools/generate-image"], work_root=tmp_path
    )
    try:
        tool = tools.load()[0]
        suite.images("fine fur")
        with native_suite_context(lambda: suite):
            generate_images(tmp_path, "fine fur", include_image_in_result=False)
            tool.execute(prompt="fine fur", include_image_in_result=False)
            tool.execute(
                prompt="requested smaller image",
                width=1024,
                height=768,
                include_image_in_result=False,
            )
        assert sizes == ["2048x2048", "2048x2048", "2048x2048", "1024x768"]
    finally:
        tools.unload()


@pytest.mark.parametrize("reference", ["missing.png", "directory"])
def test_missing_or_directory_reference_is_rejected_before_any_upload(
    tmp_path, reference
):
    (tmp_path / "reference.png").write_bytes(png_payload())
    (tmp_path / "directory").mkdir()
    suite = transport(
        lambda request: pytest.fail("Reference validation reached transport")
    )
    with (
        native_suite_context(lambda: suite),
        pytest.raises(ValueError, match=NATIVE_SUITE.invalid_reference),
    ):
        generate_images(tmp_path, "red", images=["reference.png", reference])
    assert not (tmp_path / ".exmekh_lib").exists()


@pytest.mark.parametrize(
    "width,height,n", [(32, 32, 0), (32, 32, 2), (64, 32, 1), (32, 64, 1)]
)
def test_image_response_must_match_the_request(width, height, n):
    suite = transport(
        lambda request: httpx2.Response(200, json=image_response(width, height, n=n))
    )
    with pytest.raises(ValueError, match=NATIVE_SUITE.invalid_response):
        suite.images("red", width=32, height=32)


@pytest.mark.parametrize("include_preview", [False, True])
@pytest.mark.parametrize(
    "invalid_payload", ["base64", "not-image", "jpeg", "geometry", "crc"]
)
def test_invalid_png_batch_is_rejected_before_saving_any_image(
    tmp_path, invalid_payload, include_preview
):
    raw = png_payload()
    response = image_response(n=2, raw=raw)
    if invalid_payload == "base64":
        response["data"][1]["b64_json"] = "%%%"
    else:
        bad = {
            "not-image": b"not an image",
            "jpeg": png_payload(image_format="JPEG"),
            "geometry": png_payload(64, 32),
            "crc": raw[:33] + b"bad!" + raw[37:],
        }[invalid_payload]
        response["data"][1]["b64_json"] = base64.b64encode(bad).decode()
    suite = transport(lambda request: httpx2.Response(200, json=response))
    with (
        native_suite_context(lambda: suite),
        pytest.raises(ValueError, match=NATIVE_SUITE.invalid_response),
    ):
        generate_images(
            tmp_path,
            "red",
            n=2,
            width=32,
            height=32,
            include_image_in_result=include_preview,
        )
    assert not (tmp_path / ".exmekh_lib").exists()


@pytest.mark.parametrize("n", [1, 2, 8])
def test_maximum_seed_leaves_room_for_the_entire_batch(n):
    seed = 2**64 - n

    def handler(request):
        body = json.loads(request.content)
        assert body["ds4"]["seed"] == seed
        return httpx2.Response(200, json=image_response(n=n))

    suite = transport(handler)
    assert len(suite.images("red", width=32, height=32, n=n, seed=seed).data) == n


def test_generated_image_previews_can_be_reused_as_edit_references(tmp_path):
    path = tmp_path / "reference.png"
    path.write_bytes(png_payload())
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/files":
            return httpx2.Response(
                200,
                json={
                    "id": "reference",
                    "object": "file",
                    "bytes": 1,
                    "created_at": 1,
                    "filename": "image.png",
                    "purpose": "ds4_media",
                },
            )
        if request.method == "GET":
            return httpx2.Response(200, content=png_payload())
        assert json.loads(request.content)["images"] == ["reference"]
        return httpx2.Response(200, json=image_response(raw=png_payload()))

    suite = transport(handler)
    assert suite.upload_image(Image.new("RGB", (32, 32), "red")).file_id == "reference"
    assert suite.content("reference") == png_payload()
    with native_suite_context(lambda: suite):
        result = generate_images(
            tmp_path, "edit", images=["reference.png"], width=32, height=32
        )
        assert isinstance(result, AttachedToolResult)
        assert [
            part.size
            for part in result.attachments.iter_content_parts()
            if isinstance(part, Image.Image)
        ] == [(32, 32)]
    assert requests[-1].url.path == "/v1/images/edits"
    with pytest.raises(ValueError, match=NATIVE_SUITE.invalid_scale):
        generate_images(tmp_path, "invalid preview", preview_scale=0)


def test_native_client_capability_and_saved_preview_contracts():
    provider = MagicMock(spec=Provider)
    suite = MagicMock(spec=NativeSuiteClient)
    provider.native_suite.return_value = suite
    assert Client(provider, SystemMessage("rules")).native_suite() is suite
    with pytest.raises(NotImplementedError, match=NATIVE_SUITE.unsupported_provider):
        Provider.native_suite(provider)
    assert user_message_image_urls({"parts": None}) == []
    assert user_message_image_urls(
        {
            "parts": [
                None,
                {"type": "text", "text": "caption"},
                {"type": "image", "data": "data:image/png;base64,pixels"},
                {"type": "image", "data": None},
            ]
        }
    ) == ["data:image/png;base64,pixels"]


def test_registered_image_tool_waits_for_native_service_restoration(tmp_path):
    tools = DeepTools(
        [CAPABILITY_FORMAT.bundled_root / "tools/generate-image"], work_root=tmp_path
    )
    try:
        tool = tools.load()[0]
        assert not tool.background_enabled
        assert "execution_mode" not in tool.get_type_hints()
        assert "wait_seconds" not in tool.get_work_signature().parameters
    finally:
        tools.unload()
