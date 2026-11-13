"""Explicit ranges and append-only prerequisite recovery with a simulated provider."""

import codecs
from collections import Counter
from dataclasses import replace
from threading import Event
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.file_inspection import reading
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.evidence import InspectionReport
from exmekh_lib.core.file_inspection.reading import (
    file_read_context,
    read_file,
)
from exmekh_lib.core.file_inspection.runtime import FileInspectionHistory
from exmekh_lib.core.models.annotation_schema import (
    AnnotationSchemaBuilder,
)
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments
from exmekh_lib.utils.xml import stringify_tool_result
from tests.capabilities_support import MOCK_CAPABILITIES_PATH
from tests.core.agents.node.test_node import SequenceProvider
from tests.file_inspection_support import inspect_file
from tests.tool_result_support import parse_tool_result


def call(name, call_id, **arguments):
    return ToolCall(call_id, FunctionCall(name, arguments))


def batch(*calls):
    return AssistantMessage(None, tool_calls=list(calls))


def read_call(call_id="read", path="notes.txt", start_index=1, end_index=1):
    return call(
        "read-file", call_id, path=path, start_index=start_index, end_index=end_index
    )


def open_files(call_id="open"):
    return batch(call("file-processing", call_id))


def make_node(
    tmp_path, responses, *, provider_type=SequenceProvider, hooks=None, overrides=None
):
    provider = provider_type(responses)
    node = Node(
        [MOCK_CAPABILITIES_PATH],
        name_hds="main",
        default_capabilities=DefaultCapabilities(tool_hooks=hooks, overrides=overrides),
    )
    node.initialize(provider, work_root=tmp_path)
    return node, provider


def replies(request):
    return [m for m in request.get_messages() if isinstance(m, ToolMessage)]


def result_for(request, call_id):
    return parse_tool_result(
        next(
            m for m in replies(request) if m.get_tool_call_id() == call_id
        ).get_message()
    )


def assert_paired(messages):
    calls = Counter(
        c.id
        for m in messages
        if isinstance(m, AssistantMessage)
        for c in m.get_tool_calls() or ()
    )
    results = Counter(
        m.get_tool_call_id() for m in messages if isinstance(m, ToolMessage)
    )
    assert calls == results
    assert all(count == 1 for count in calls.values())


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"one\r\ntwo\rthree\n")
    return path


@pytest.mark.parametrize(
    "raw,filename,expected",
    [
        (b"one\r\ntwo\rthree\n", "a.txt", ["one", "two", "three"]),
        ("a\u2028b\u0085c\n\n".encode(), "a.txt", ["a\u2028b\u0085c", ""]),
        (
            codecs.BOM_UTF16_LE + "uno\r\ndue".encode("utf-16-le"),
            "a.txt",
            ["uno", "due"],
        ),
        (codecs.BOM_UTF32_BE + "uno\ndue".encode("utf-32-be"), "a.txt", ["uno", "due"]),
        (
            b"# coding: latin-1\nname = '\xe8'\n",
            "a.py",
            ["# coding: latin-1", "name = 'è'"],
        ),
    ],
)
def test_physical_lines_match_inspection(tmp_path, raw, filename, expected):
    (tmp_path / filename).write_bytes(raw)
    report = inspect_file(tmp_path, filename)
    assert isinstance(report, InspectionReport)
    with file_read_context((report.evidence,)):
        result = read_file(tmp_path, filename, start_index=1, end_index=len(expected))
    assert (result["start_index"], result["end_index"]) == (1, len(expected))
    assert result["total_lines"] == report["total_lines"] == len(expected)
    assert result["lines"] == [
        {"line": number, "text": text} for number, text in enumerate(expected, 1)
    ]
    assert "digest" not in stringify_tool_result(report)


@pytest.mark.parametrize(
    "start,end,parameter",
    [
        (0, 1, "start_index"),
        (-1, 1, "start_index"),
        (True, 1, "start_index"),
        (1.0, 1, "start_index"),
        ("1", 1, "start_index"),
        (None, 1, "start_index"),
        (1, 0, "end_index"),
        (1, False, "end_index"),
        (1, "2", "end_index"),
        (3, 2, "start_index"),
    ],
)
def test_invalid_indices_fail_without_file_access(
    tmp_path, monkeypatch, start, end, parameter
):
    def unexpected(*args):
        pytest.fail("invalid arguments must not access files")

    monkeypatch.setattr(reading, "read_snapshot", unexpected)
    with pytest.raises(FileInspectionError, match=parameter):
        read_file(tmp_path, "notes.txt", start, end)


@pytest.mark.parametrize("path", [None, 1, "", " "])
def test_invalid_path_explains_parameter(tmp_path, path):
    with pytest.raises(FileInspectionError, match="path.*workroot"):
        read_file(tmp_path, path, 1, 1)


@pytest.mark.parametrize("start,end", [(1, 4), (4, 4)])
def test_range_is_never_clamped(tmp_path, text_file, start, end):
    report = inspect_file(tmp_path, text_file.name)
    with file_read_context((report.evidence,)):
        with pytest.raises(FileInspectionError, match=r"outside.*1\.\.3"):
            read_file(tmp_path, text_file.name, start, end)


@pytest.mark.parametrize("raw,error", [(b"", "empty"), (b"\x00binary", "no readable")])
def test_empty_and_nontext_files_are_explicit(tmp_path, raw, error):
    (tmp_path / "a.txt").write_bytes(raw)
    report = inspect_file(tmp_path, "a.txt")
    with file_read_context((report.evidence,)):
        with pytest.raises(FileInspectionError, match=error):
            read_file(tmp_path, "a.txt", 1, 1)


def test_missing_prerequisite_does_not_read_or_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reading, "read_snapshot", lambda *args: pytest.fail("early read")
    )
    assert (
        read_file(tmp_path, "a.txt", 1, 1)["error_kind"] == "file_inspection_required"
    )


def test_read_checks_only_the_requested_snapshot(tmp_path, text_file, monkeypatch):
    (tmp_path / "other.txt").write_text("one\ntwo\nthree\n")
    other = inspect_file(tmp_path, "other.txt")
    target = inspect_file(tmp_path, text_file.name)
    read_snapshot = reading.read_snapshot
    opened = []

    def record_snapshot(work_root, path, settings):
        opened.append(path)
        return read_snapshot(work_root, path, settings)

    monkeypatch.setattr(reading, "read_snapshot", record_snapshot)
    with file_read_context((other.evidence, target.evidence)):
        assert read_file(tmp_path, text_file.name, 1, 1).selected_text() == "one"
    assert opened == [text_file.name]


def test_read_schema_requires_all_indices_and_describes_prerequisite(tmp_path):
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    tool = next(t for t in loaded.tools if t.name == "read-file")
    mapper = OpenAIFunctionToolDefinitionMapper(AnnotationSchemaBuilder())
    for strict in (False, True):
        schema = mapper.map(tool, strict=strict).get_parameters()
        assert set(schema["properties"]) == {"path", "start_index", "end_index"}
        assert set(schema["required"]) == {"path", "start_index", "end_index"}
        assert schema["properties"]["start_index"]["type"] == "integer"
        assert schema["properties"]["end_index"]["type"] == "integer"
    assert "requires visible inspect-file of unchanged content" in tool.description
    assert "Images need no inspection" in tool.description
    assert "indices ignored" in tool.description
    assert "No implicit full read or range clamping" in tool.description


def test_successful_visible_analysis_is_reused_without_synthetic_calls(
    tmp_path, text_file
):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name)),
            batch(read_call("first", start_index=2, end_index=3), read_call("second")),
            AssistantMessage("done"),
        ],
    )
    assert node.run("read").get_message() == "done"
    assert result_for(provider.requests[-1], "first")["lines"] == [
        {"line": 2, "text": "two"},
        {"line": 3, "text": "three"},
    ]
    assert result_for(provider.requests[-1], "second")["ok"]
    assert len(provider.requests) == 4
    assert (
        len(
            [
                m
                for m in replies(provider.requests[-1])
                if m.get_name() == "inspect-file"
            ]
        )
        == 1
    )
    assert_paired(provider.requests[-1].get_messages())


def test_missing_analysis_appends_error_and_real_inspection_before_new_indices(
    tmp_path, text_file
):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call()),
            batch(read_call("retry", start_index=3, end_index=3)),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    request = provider.requests[2]
    assert result_for(request, "read")["error_kind"] == "file_inspection_required"
    tail = request.get_messages()[-4:]
    assert [m.get_role_name() for m in tail] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    auto = tail[2].get_tool_calls()[0]
    assert auto.function == FunctionCall("inspect-file", {"path": text_file.name})
    assert auto.id != "read"
    assert parse_tool_result(tail[3].get_message())["total_lines"] == 3
    assert result_for(provider.requests[-1], "retry")["lines"] == [
        {"line": 3, "text": "three"}
    ]
    prefix = provider.requests[1].get_messages()
    assert [m.message_dump() for m in request.get_messages()[: len(prefix)]] == [
        m.message_dump() for m in prefix
    ]
    assert_paired(provider.requests[-1].get_messages())
    assert all(set(r.get_extra()) == {"agentic"} for r in provider.requests)


def test_same_size_content_change_invalidates_hash(tmp_path, text_file):
    class EditingProvider(SequenceProvider):
        def call(self, request):
            if len(self.requests) == 2:
                text_file.write_bytes(b"new\r\ntwo\rthree\n")
            return super().call(request)

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name)),
            batch(read_call()),
            batch(read_call("retry")),
            AssistantMessage("done"),
        ],
        provider_type=EditingProvider,
    )
    node.run("read")
    assert result_for(provider.requests[3], "read")["ok"] is False
    assert result_for(provider.requests[-1], "retry")["lines"] == [
        {"line": 1, "text": "new"}
    ]
    assert_paired(provider.requests[-1].get_messages())


def test_mixed_group_completes_original_results_then_deduplicates_inspections(
    tmp_path, text_file
):
    (tmp_path / "b.txt").write_text("b\n")
    (tmp_path / "c.txt").write_text("c\n")
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name)),
            batch(
                read_call("valid"),
                read_call("b1", "b.txt"),
                read_call("c", "c.txt"),
                read_call("b2", "./b.txt"),
            ),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    messages = provider.requests[-1].get_messages()
    tail = messages[-8:]
    assert [m.get_role_name() for m in tail] == [
        "assistant",
        "tool",
        "tool",
        "tool",
        "tool",
        "assistant",
        "tool",
        "tool",
    ]
    assert [m.get_tool_call_id() for m in tail[1:5]] == ["valid", "b1", "c", "b2"]
    assert result_for(provider.requests[-1], "valid")["ok"]
    for call_id in ("b1", "c", "b2"):
        assert (
            result_for(provider.requests[-1], call_id)["error_kind"]
            == "file_inspection_required"
        )
    assert [c.get_arguments()["path"] for c in tail[5].get_tool_calls()] == [
        "b.txt",
        "c.txt",
    ]
    assert len(provider.requests) == 4
    assert_paired(messages)


@pytest.mark.parametrize("path", ["missing.txt", "../outside.txt"])
def test_automatic_inspection_failure_keeps_both_results_and_stops(tmp_path, path):
    if path.startswith(".."):
        (tmp_path.parent / "outside.txt").write_text("outside")
    node, provider = make_node(
        tmp_path, [open_files(), batch(read_call(path=path)), AssistantMessage("done")]
    )
    node.run("read")
    outputs = replies(provider.requests[-1])[-2:]
    assert [m.get_name() for m in outputs] == ["read-file", "inspect-file"]
    assert [parse_tool_result(m.get_message())["ok"] for m in outputs] == [False, False]
    assert (
        "workroot" in parse_tool_result(outputs[-1].get_message())["error"]
        if path.startswith("..")
        else "exist" in parse_tool_result(outputs[-1].get_message())["error"]
    )
    assert len(provider.requests) == 3
    assert_paired(provider.requests[-1].get_messages())


def test_unexpected_automatic_inspection_failure_is_a_real_xml_reply(tmp_path):
    def broken(path: str):
        raise OSError("unavailable")

    node, provider = make_node(
        tmp_path,
        [open_files(), batch(read_call()), AssistantMessage("done")],
        overrides={"inspect-file": Tool(broken, name="inspect-file")},
    )
    node.run("read")
    result = parse_tool_result(replies(provider.requests[-1])[-1].get_message())
    assert result["error_kind"] == "tool_execution_failed"
    assert "OSError" in result["error"]
    assert_paired(provider.requests[-1].get_messages())


@pytest.mark.parametrize("missing", ["path", "start_index", "end_index"])
def test_missing_argument_returns_descriptive_xml_without_auto_inspection(
    tmp_path, missing
):
    args = {"path": "notes.txt", "start_index": 1, "end_index": 1}
    args.pop(missing)
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("read-file", "read", **args)),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    result = result_for(provider.requests[-1], "read")
    assert result["error_kind"] == "invalid_tool_arguments"
    assert missing in result["error"]
    assert "inspect-file" in result["error"]
    assert replies(provider.requests[-1])[-1].get_name() == "read-file"


def test_closed_skill_analysis_and_cache_read_cannot_authorize_reopened_skill(
    tmp_path, text_file
):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name)),
            AssistantMessage("analyzed"),
        ],
    )
    node.run("inspect")
    inspection = next(
        m for m in replies(provider.requests[-1]) if m.get_name() == "inspect-file"
    )
    header = ET.fromstring(
        inspection.get_message().split("</header>", 1)[0] + "</header>"
    )
    output_id = header.findtext("id")
    provider._responses.extend(
        [
            batch(call("exit_agent", "exit", text="analysis saved", items=[output_id])),
            batch(call("read_tool_output", "saved", output_id=output_id)),
            open_files("reopen"),
            batch(read_call()),
            AssistantMessage("done"),
        ]
    )
    node.run("read in a new skill")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert not any(
        m.get_tool_call_id() == "inspect" for m in replies(provider.requests[-1])
    )
    assert_paired(provider.requests[-1].get_messages())


@pytest.mark.parametrize("compact", ["remove_pair", "reference", "summary"])
def test_current_history_compaction_invalidates_inspection(
    tmp_path, text_file, compact
):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name)),
            AssistantMessage("analyzed"),
        ],
    )
    node.run("inspect")
    messages = provider.requests[-1].get_messages()[1:]
    if compact in {"remove_pair", "reference"}:
        messages = [
            m
            for m in messages
            if not (
                (
                    isinstance(m, AssistantMessage)
                    and any(c.id == "inspect" for c in m.get_tool_calls() or ())
                )
                or (isinstance(m, ToolMessage) and m.get_tool_call_id() == "inspect")
            )
        ]
        if compact == "reference":
            messages.append(
                AssistantMessage("inspect-file was called: inspect; three lines.")
            )
    else:
        messages = [
            ToolMessage("Summary: three lines", "inspect", "inspect-file")
            if isinstance(m, ToolMessage) and m.get_tool_call_id() == "inspect"
            else m
            for m in messages
        ]
    node.restore_history(messages)
    provider._responses.extend([batch(read_call()), AssistantMessage("done")])
    node.run("read")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert replies(provider.requests[-1])[-1].get_name() == "inspect-file"


def test_history_proof_is_node_local_and_requires_unchanged_report(tmp_path, text_file):
    report = inspect_file(tmp_path, text_file.name)
    analysis_call = call("inspect-file", "inspect", path=text_file.name)
    response = ToolMessage(stringify_tool_result(report), "inspect", "inspect-file")
    history = [batch(analysis_call), response]
    registry = FileInspectionHistory()
    registry.record(analysis_call, report, response)
    assert registry.visible(history) == (report.evidence,)
    assert FileInspectionHistory().visible(history) == ()
    assert registry.visible([response, batch(analysis_call)]) == ()
    filtered = FileInspectionHistory()
    report["total_lines"] = 100
    filtered.record(analysis_call, report, response)
    filtered.record(analysis_call, {"ok": False}, response)
    assert filtered.visible(history) == ()


def test_hooks_apply_to_automatic_inspection_and_read(tmp_path, text_file):
    events = []

    class Before(BeforeTool):
        def handle(self, context):
            events.append(("before", context.arguments["path"], context.work_root))
            return ToolArguments({**context.arguments, "detailed": True})

    class After(AfterTool):
        def handle(self, context, result):
            events.append(("after", context.arguments["path"], context.work_root))
            return result

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call()),
            batch(read_call("retry")),
            AssistantMessage("done"),
        ],
        hooks={"inspect-file": ToolHooks(before=Before(), after=After())},
    )
    node.run("read")
    assert events == [
        ("before", text_file.name, tmp_path),
        ("after", text_file.name, tmp_path),
    ]
    assert result_for(provider.requests[-1], "retry")["ok"]
    assert "metadata" in parse_tool_result(
        next(
            m for m in replies(provider.requests[-1]) if m.get_name() == "inspect-file"
        ).get_message()
    )


def test_hash_validation_is_bounded_and_missing_files_are_reinspected(
    tmp_path, text_file, monkeypatch
):
    report = inspect_file(tmp_path, text_file.name)
    monkeypatch.setattr(
        reading, "FILE_INSPECTION", replace(FILE_INSPECTION, max_file_bytes=1)
    )
    with file_read_context((report.evidence,)):
        assert (
            read_file(tmp_path, text_file.name, 1, 1)["error_kind"]
            == "file_inspection_required"
        )
        text_file.unlink()
        assert (
            read_file(tmp_path, text_file.name, 1, 1)["error_kind"]
            == "file_inspection_required"
        )


def test_removed_inspection_file_cannot_authorize_another_file(tmp_path, text_file):
    report = inspect_file(tmp_path, text_file.name)
    (tmp_path / "other.txt").write_bytes(text_file.read_bytes())
    text_file.unlink()
    with file_read_context((report.evidence,)):
        assert (
            read_file(tmp_path, "other.txt", 1, 1)["error_kind"]
            == "file_inspection_required"
        )


def test_nontext_decode_is_explicit_even_with_text_evidence(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"\x00binary")
    report = inspect_file(tmp_path, "a.txt")
    with file_read_context((replace(report.evidence, readable=True),)):
        with pytest.raises(FileInspectionError, match="no readable"):
            read_file(tmp_path, "a.txt", 1, 1)


def test_summarized_inspection_response_is_never_registered(tmp_path, text_file):
    report = inspect_file(tmp_path, text_file.name)
    inspection = call("inspect-file", "inspect", path=text_file.name)
    response = ToolMessage("Summary: three lines", "inspect", "inspect-file")
    registry = FileInspectionHistory()
    registry.record(inspection, report, response)
    assert registry.visible([batch(inspection), response]) == ()


def test_read_after_hook_cannot_drop_runtime_recovery_request(tmp_path, text_file):
    class CopyResult(AfterTool):
        def handle(self, context, result):
            return dict(result)

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call()),
            batch(read_call("retry")),
            AssistantMessage("done"),
        ],
        hooks={"read-file": ToolHooks(after=CopyResult())},
    )
    node.run("read")
    assert result_for(provider.requests[2], "read")["ok"] is False
    assert replies(provider.requests[2])[-1].get_name() == "inspect-file"
    assert result_for(provider.requests[-1], "retry")["ok"]


def test_path_resolution_failure_is_reported_by_real_automatic_inspection(tmp_path):
    (tmp_path / "loop").symlink_to("loop")
    node, provider = make_node(
        tmp_path,
        [open_files(), batch(read_call(path="loop")), AssistantMessage("done")],
    )
    node.run("read")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert (
        parse_tool_result(replies(provider.requests[-1])[-1].get_message())["ok"]
        is False
    )
    assert_paired(provider.requests[-1].get_messages())


def test_unseen_sibling_inspection_does_not_authorize_indices(tmp_path, text_file):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=text_file.name), read_call()),
            batch(read_call("retry", start_index=2, end_index=2)),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    assert (
        result_for(provider.requests[2], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert result_for(provider.requests[-1], "retry")["lines"] == [
        {"line": 2, "text": "two"}
    ]
    assert_paired(provider.requests[-1].get_messages())


def test_background_inspection_only_qualifies_after_full_visible_delivery(
    tmp_path, text_file
):
    started = Event()
    release = Event()

    class WaitForRelease(AfterTool):
        def handle(self, context, result):
            started.set()
            assert release.wait(5)
            return result

    class ReleasingProvider(SequenceProvider):
        def call(self, request):
            if len(self.requests) == 2:
                assert started.wait(5)
                assert result_for(request, "inspect")["status"] == "running"
                release.set()
            return super().call(request)

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(
                call(
                    "inspect-file",
                    "inspect",
                    path=text_file.name,
                    execution_mode="background",
                )
            ),
            batch(
                call(
                    "control_tool_execution",
                    "wait",
                    tool_call_id="inspect",
                    action="wait",
                    wait_seconds=5,
                )
            ),
            batch(read_call()),
            AssistantMessage("done"),
        ],
        provider_type=ReleasingProvider,
        hooks={"inspect-file": ToolHooks(after=WaitForRelease())},
    )
    try:
        node.run("read")
    finally:
        release.set()
    assert result_for(provider.requests[-1], "read")["ok"]
    assert (
        len(
            [
                m
                for m in replies(provider.requests[-1])
                if m.get_name() == "inspect-file"
            ]
        )
        == 1
    )
    assert any(
        m.get_name() == "read_tool_output" for m in replies(provider.requests[-1])
    )
    assert_paired(provider.requests[-1].get_messages())


def test_running_inspection_never_authorizes_read(tmp_path, text_file):
    started = Event()
    release = Event()
    invocations = []

    class WaitOnce(AfterTool):
        def handle(self, context, result):
            invocations.append(context.arguments)
            if len(invocations) == 1:
                started.set()
                assert release.wait(5)
            else:
                release.set()
            return result

    class CheckingProvider(SequenceProvider):
        def call(self, request):
            if len(self.requests) == 2:
                assert started.wait(5)
            return super().call(request)

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(
                call(
                    "inspect-file",
                    "inspect",
                    path=text_file.name,
                    execution_mode="background",
                )
            ),
            batch(read_call()),
            AssistantMessage("done"),
        ],
        provider_type=CheckingProvider,
        hooks={"inspect-file": ToolHooks(after=WaitOnce())},
    )
    try:
        node.run("read")
    finally:
        release.set()
    assert (
        result_for(provider.requests[-1], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert len(invocations) == 2
    assert_paired(provider.requests[-1].get_messages())


def test_failed_analysis_keeps_private_hash_but_does_not_authorize_read(tmp_path):
    (tmp_path / "broken.json").write_text('{"x":')
    report = inspect_file(tmp_path, "broken.json")
    assert isinstance(report, InspectionReport)
    assert report.evidence.digest
    assert not report.evidence.successful
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path="broken.json")),
            batch(read_call(path="broken.json")),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert (
        parse_tool_result(replies(provider.requests[-1])[-1].get_message())[
            "analysis_status"
        ]
        == "invalid_format"
    )
    assert len(provider.requests) == 4
    assert_paired(provider.requests[-1].get_messages())
