from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from unittest.mock import MagicMock

import pytest

from exmekh_lib.capabilities.tools.read_tool_output.summary import ToolOutputSummarizer
from exmekh_lib.capabilities.tools.read_tool_output.tool_output_reader import (
    ToolOutputReader,
)
from exmekh_lib.capabilities.tools.search_tool_outputs.tool_call_search import (
    ToolCallSearch,
)
from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.file_inspection.output_access import output_file_access
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.interfaces.provider import Provider
from tests.file_inspection_support import inspect_file
from tests.tokenizer_support import StubTokenizer


@pytest.fixture
def output_system(tmp_path):
    with SessionCache(tmp_path / "cache") as session:
        cache = session.node("owner")
        preparer = ToolOutputPreparer(
            StubTokenizer(),
            OutputFileWorkspace(lambda: cache, tmp_path, ToolOutputSettings(10)),
        )
        manager = ToolExecutionManager()
        reader = ToolOutputReader(lambda: cache, manager, output_preparer=preparer)
        yield cache, preparer, manager, reader


def test_legacy_large_saved_record_is_migrated_and_cursors_cannot_reveal_it(
    output_system,
    xml_clock,
):
    cache, _, _, reader = output_system
    original = "private" * 20
    cache.store(ToolCallRecord("work", "id", {}, original))
    read = reader.read("id", cursor=99999)
    assert read.external_output is not None
    assert original not in read.content
    assert cache.get("id").result is None
    assert cache.get("id").external_output == read.external_output
    assert reader.read("id").content == read.content
    assert "private" not in str(ToolCallSearch(cache).search(query="private"))


def test_partial_export_keeps_cursor_and_final_read_switches_to_final_reference(
    output_system, tmp_path
):
    _, _, manager, reader = output_system
    release = Event()
    progress = "partial " * 10
    final = "final " * 10
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait(2) and final,
            read_partial=lambda cursor: (progress[cursor:], max(cursor, len(progress))),
        ),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )
    try:
        partial = reader.read("id", cursor=0)
        assert partial.external_output is not None
        assert "<next-cursor>80</next-cursor>" in partial.content
        assert "<response-type>partial</response-type>" in partial.content
        assert progress not in partial.content
        release.set()
        assert manager.wait("call", 1).is_finished
        complete = reader.read("id", cursor=80)
        assert complete.external_output is not None
        assert "<response-type>final</response-type>" in complete.content
        assert (
            "".join(
                (tmp_path / item.path).read_text()
                for item in complete.external_output.files
            )
            == final
        )
        assert complete.external_output != partial.external_output
        assert manager.by_output_id("id").result is None
        assert manager.next_delivery(("owner",)) is None
    finally:
        release.set()


def test_search_of_runtime_final_materializes_reference_without_acknowledging_delivery(
    output_system,
):
    cache, _, manager, reader = output_system
    release = Event()
    original = "private" * 20
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait(2) and original),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )
    release.set()
    manager.wait("call", 1)
    found = ToolCallSearch(cache, manager, output_system[1]).search(tool_name="work")
    assert original not in str(found)
    assert found["records"][0]["external_output"]["files"]
    assert manager.next_delivery(("owner",)) is not None
    assert reader.read("id").external_output is not None
    assert manager.next_delivery(("owner",)) is None
    assert cache.get_all() == []


def test_export_failure_does_not_acknowledge_final_read_and_can_be_retried(
    output_system, monkeypatch
):
    _, _, manager, reader = output_system
    release = Event()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait(2) and "long output" * 10),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )
    release.set()
    manager.wait("call", 1)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", MagicMock(side_effect=OSError("disk full")))
        failed = reader.read("id")
        assert failed.delivery_failed
        assert manager.next_delivery(("owner",)) is not None
    assert reader.read("id").external_output is not None
    assert manager.next_delivery(("owner",)) is None


def test_summaries_never_read_exported_text_or_call_provider(output_system):
    cache, preparer, _, _ = output_system
    reference = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("private" * 20)
        )
    ).external_output
    cache.store(ToolCallRecord("work", "id", {}, None, external_output=reference))
    provider = MagicMock(spec=Provider)
    provider.call.side_effect = AssertionError("must not summarize files")
    reader = ToolOutputReader(
        lambda: cache,
        summarizer=ToolOutputSummarizer(provider),
        output_preparer=preparer,
    )
    assert reader.read("id", summary_mode="brief").external_output == reference
    assert reader.read("id", summary_mode="detailed").external_output == reference
    provider.call.assert_not_called()


def test_file_tools_can_read_exact_budget_chunk_without_recursive_export(
    output_system, tmp_path
):
    _, preparer, _, _ = output_system
    reference = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("abcdefghijk")
        )
    ).external_output
    path = reference.files[0].path
    with output_file_access(preparer.files):
        inspection = inspect_file(
            tmp_path, path, max_items=1, output_settings=preparer.settings
        )
        plan = inspection["read_plan"]
        assert plan["ranges"] == [{"start_index": 1, "end_index": 1, "tokens": 10}]
        span = plan["ranges"][0]
        with file_read_context((inspection.evidence,)):
            read = read_file(tmp_path, path, span["start_index"], span["end_index"])
    assert read.selected_text() == "abcdefghij"
    result = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("read-file", "read"),
            PreformattedToolResult("verbose XML wrapping\n" + read.selected_text()),
            budget_text=read.selected_text(),
        )
    )
    assert result.external_output is None


def test_owned_multiline_files_larger_than_normal_byte_limit_can_be_inspected(
    output_system, tmp_path
):
    _, preparer, _, _ = output_system
    reference = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("abc\n" * 20)
        )
    ).external_output
    with output_file_access(preparer.files):
        inspection = inspect_file(
            tmp_path,
            reference.files[0].path,
            settings=replace(FILE_INSPECTION, max_file_bytes=2),
        )
    assert inspection["analysis_status"] == "complete"
    assert inspection["total_lines"] == 20


def test_finishing_during_partial_read_does_not_acknowledge_unread_final(output_system):
    _, _, manager, reader = output_system
    release = Event()

    def partial(cursor):
        release.set()
        assert manager.wait("call", 1).is_finished
        return "part", cursor + 4

    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait(2) and "done", read_partial=partial
        ),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )
    try:
        response = reader.read("id", cursor=0)
        assert "<response-type>partial</response-type>" in response.content
        assert manager.next_delivery(("owner",)) is not None
        assert reader.read("id").content.endswith("done")
        assert manager.next_delivery(("owner",)) is None
    finally:
        release.set()


def test_cancel_confirmation_during_export_cannot_publish_completed_text(
    output_system, monkeypatch
):
    _, preparer, manager, reader = output_system
    release = Event()
    cancelling = Event()
    confirm = Event()
    cancelled = Event()

    def cancel():
        cancelling.set()
        assert confirm.wait(2)
        return True

    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait(2) and "private" * 20, cancel=cancel
        ),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )

    def request_cancel():
        manager.cancel("call")
        cancelled.set()

    thread = Thread(target=request_cancel)
    thread.start()
    try:
        assert cancelling.wait(1)
        release.set()
        assert manager.wait("call", 1).status == "completed"
        prepare = preparer.prepare

        def exporting(*args, **kwargs):
            result = prepare(*args, **kwargs)
            confirm.set()
            assert cancelled.wait(1)
            return result

        monkeypatch.setattr(preparer, "prepare", exporting)
        result = reader.read("id")
        assert "<response-type>cancelled</response-type>" in result.content
        assert "private" * 20 not in result.content
    finally:
        release.set()
        confirm.set()
        thread.join(2)


def test_exported_record_rejects_retaining_the_original_result(output_system):
    _, preparer, _, _ = output_system
    reference = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("private" * 20)
        )
    ).external_output
    with pytest.raises(ValueError, match="original result"):
        ToolCallRecord("work", "id", {}, "private", external_output=reference)


def test_saved_running_export_preserves_unavailable_warning_without_reintroducing_content(
    output_system,
):
    _, preparer, _, _ = output_system
    reference = preparer.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("private" * 20)
        )
    ).external_output
    record = ToolCallRecord(
        "work",
        "id",
        {},
        None,
        execution={"status": "running"},
        external_output=reference,
    )
    saved = ToolOutputView(record).saved_record()
    assert saved.result is None
    assert saved.external_output == reference
    rendered = ToolOutputView(saved).format_saved(None)
    assert "<response-type>failed</response-type>" in rendered
    assert "Execution is unavailable" in rendered


def test_saved_export_failure_remains_failed_even_when_execution_completed(
    output_system,
):
    cache, _, _, reader = output_system
    record = ToolCallRecord(
        "work", "id", {}, "done", execution={"status": "completed"}
    ).with_delivery_failure({"ok": False, "error": "disk full"})
    cache.store(record)
    response = reader.read("id")
    assert "<status>completed</status>" in response.content
    assert "<response-type>failed</response-type>" in response.content


@pytest.mark.parametrize("include_runtime", [True, False])
def test_search_materializes_legacy_saved_records_with_or_without_runtime(
    output_system, tmp_path, include_runtime
):
    cache, preparer, manager, _ = output_system
    original = "private" * 20
    cache.store(ToolCallRecord("work", "id", {}, original))
    search = ToolCallSearch(cache, manager if include_runtime else None, preparer)
    result = search.search(tool_name="work")
    assert original not in str(result)
    files = result["records"][0]["external_output"]["files"]
    assert "".join((tmp_path / item["path"]).read_text() for item in files) == original
    assert search.search(query="private")["records"] == []
    assert cache.get("id").result == original


def test_managed_preformatted_delivery_failure_remains_failed_on_every_read(
    output_system,
):
    _, _, manager, reader = output_system
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: PreformattedToolResult("delivery problem", delivery_failed=True)
        ),
        ("owner",),
        ExecutionOptions(),
        "id",
    )
    for _ in range(2):
        response = reader.read("id")
        assert response.delivery_failed
        assert "<response-type>failed</response-type>" in response.content
        assert manager.next_delivery(("owner",)) is not None
