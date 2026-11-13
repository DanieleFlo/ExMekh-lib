"""Token budgets on inspection labels, PDF properties and real runtime calls."""

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from pypdf import PdfWriter

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.file_inspection.tokenization import (
    current_inspection_tokenizer,
    inspection_tokenizer_context,
)
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.tokenizers.text_budget import TextTokenizationError
from exmekh_lib.utils.xml import stringify_tool_result
from tests.core.agents.node.test_node import SequenceProvider
from tests.file_inspection_support import inspect_file
from tests.tokenizer_support import PairTokenizer, StubTokenizer
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize("length", [399, 400, 401])
def test_region_labels_are_limited_to_200_tokens(length, tmp_path):
    label = ("界😀àx" * 101)[:length]
    (tmp_path / "notes.md").write_text(f"# {label}\n", encoding="utf-8")
    tokenizer = PairTokenizer()

    report = inspect_file(tmp_path, "notes.md", tokenizer=tokenizer, detailed=True)

    assert report["regions"][0]["name"] == label[:400]
    assert tokenizer.count_tokens(report["regions"][0]["name"]) <= 200
    assert report["selection"]["labels_truncated"] is (length > 400)
    assert report["limits"]["max_label_tokens"] == 200
    assert "max_label_chars" not in report["limits"]


def test_pdf_properties_share_the_label_token_budget(tmp_path):
    tokenizer = PairTokenizer()
    long_title = "界😀àx" * 110
    author = "界😀àx" * 90
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Title": long_title, "/Author": author})
    buffer = io.BytesIO()
    writer.write(buffer)
    (tmp_path / "report.pdf").write_bytes(buffer.getvalue())

    report = inspect_file(tmp_path, "report.pdf", tokenizer=tokenizer)

    properties = report["analysis"]["properties"]
    assert properties["title"] == long_title[:400]
    assert properties["author"] == author
    assert tokenizer.count_tokens(properties["title"]) == 200


def test_custom_label_budget_is_expressed_in_tokens(tmp_path):
    (tmp_path / "notes.md").write_text("# à界😀abc\n", encoding="utf-8")
    report = inspect_file(
        tmp_path,
        "notes.md",
        tokenizer=PairTokenizer(),
        settings=replace(FILE_INSPECTION, max_label_tokens=2),
    )
    assert report["regions"][0]["name"] == "à界😀a"
    assert report["selection"]["labels_truncated"]


def test_read_plan_covers_physical_lines_with_the_read_budget(tmp_path):
    (tmp_path / "notes.txt").write_bytes("ab界\r\nxyz\r01234567890\n\nñ".encode())
    tokenizer = PairTokenizer()
    options = {
        "tokenizer": tokenizer,
        "output_settings": ToolOutputSettings(max_tokens=4),
        "settings": replace(FILE_INSPECTION, read_ranges_per_page=2),
    }
    inspection = inspect_file(tmp_path, "notes.txt", **options)
    plan = inspection["read_plan"]
    assert plan == {
        "max_tokens": 4,
        "ranges": [
            {"start_index": 1, "end_index": 2, "tokens": 4},
        ],
        "next_line": 3,
    }
    oversized = inspect_file(tmp_path, "notes.txt", read_from_line=3, **options)[
        "read_plan"
    ]
    assert oversized["ranges"] == [
        {"start_index": 3, "end_index": 3, "tokens": 6, "over_limit": True},
    ]
    assert oversized["next_line"] == 4
    tail = inspect_file(tmp_path, "notes.txt", read_from_line=4, **options)["read_plan"]
    assert tail == {
        "max_tokens": 4,
        "ranges": [{"start_index": 4, "end_index": 5, "tokens": 1}],
    }
    with file_read_context((inspection.evidence,)):
        for entry in plan["ranges"] + oversized["ranges"] + tail["ranges"]:
            report = read_file(
                tmp_path, "notes.txt", entry["start_index"], entry["end_index"]
            )
            assert tokenizer.count_tokens(report.selected_text()) == entry["tokens"]


def test_read_plan_prefers_a_nearby_section_boundary(tmp_path):
    (tmp_path / "notes.md").write_text("# A\n1234\n# B\n1234\n# C\n1234\n")
    report = inspect_file(
        tmp_path,
        "notes.md",
        max_items=1,
        output_settings=ToolOutputSettings(max_tokens=13),
    )
    assert report["read_plan"]["ranges"] == [
        {"start_index": 1, "end_index": 2, "tokens": 8},
    ]
    assert report["read_plan"]["next_line"] == 3


def test_read_plan_handles_empty_and_unreadable_text(tmp_path):
    (tmp_path / "empty.txt").touch()
    (tmp_path / "broken.json").write_text('{"x":')
    assert inspect_file(tmp_path, "empty.txt")["read_plan"]["ranges"] == [
        {"start_index": 0, "end_index": 0, "tokens": 0}
    ]
    assert "read_plan" not in inspect_file(tmp_path, "broken.json")


def test_inspection_paginates_to_fit_the_rendered_xml_budget(tmp_path):
    labels = [f"Section {i} " + "<&>" * 110 for i in range(8)]
    (tmp_path / "notes.md").write_text("\n".join(f"# {name}\nbody" for name in labels))
    tokenizer = StubTokenizer()
    settings = replace(FILE_INSPECTION, max_report_tokens=1800)
    offset = 0
    starts = []
    while offset is not None:
        report = inspect_file(
            tmp_path, "notes.md", tokenizer=tokenizer, settings=settings, offset=offset
        )
        assert tokenizer.count_tokens(stringify_tool_result(report)) <= 1800
        assert report["regions"]
        starts.extend(region["start_line"] for region in report["regions"])
        offset = report["selection"].get("next_offset")
    assert starts == list(range(1, 16, 2))


def test_tight_report_budget_keeps_an_actionable_read(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n")
    tokenizer = StubTokenizer()
    report = inspect_file(
        tmp_path,
        "notes.txt",
        tokenizer=tokenizer,
        output_settings=ToolOutputSettings(max_tokens=750),
    )

    assert tokenizer.count_tokens(stringify_tool_result(report)) <= 750
    assert report["total_lines"] == 3
    assert report["read_plan"]["ranges"] == [
        {"start_index": 1, "end_index": 3, "tokens": 13}
    ]
    with file_read_context((report.evidence,)):
        assert (
            read_file(tmp_path, "notes.txt", 1, 3).selected_text() == "one\ntwo\nthree"
        )


def test_exact_symbol_read_plan_stops_at_the_symbol_end(tmp_path):
    (tmp_path / "service.ts").write_text(
        "function fetchHTTPDataArchive() {}\nfunction fetchHTTPData() {\n"
        "  return value;\n}\nconst unrelated = true;\n"
    )
    report = inspect_file(tmp_path, "service.ts", query="fetch http data", max_items=1)
    assert report["regions"][0]["name"] == "fetchHTTPData"
    assert report["selection"]["matching_items"] == 2
    plan = report["read_plan"]
    assert [(r["start_index"], r["end_index"]) for r in plan["ranges"]] == [(2, 4)]
    assert "next_line" not in plan


@pytest.mark.parametrize("query, expected_line", [("alpha", 1), ("beta", 3)])
def test_search_ranks_complete_names_even_when_display_labels_are_identical(
    tmp_path, query, expected_line
):
    (tmp_path / "notes.md").write_text("# same alpha\n\n# same beta\n")
    report = inspect_file(
        tmp_path,
        "notes.md",
        query=query,
        max_items=1,
        settings=replace(FILE_INSPECTION, max_label_tokens=4),
    )

    assert report["regions"][0]["name"] == "same"
    assert report["regions"][0]["start_line"] == expected_line
    assert report["selection"]["labels_truncated"]
    assert report["selection"]["ranking"] == "lexical"
    assert report["selection"]["has_more"] is False


@pytest.mark.parametrize(
    "cause", [ValueError("invalid vocabulary"), OSError("cache unavailable")]
)
def test_tokenizer_failures_are_not_misreported_as_invalid_file_formats(
    tmp_path, cause
):
    class UnavailableTokenizer(StubTokenizer):
        def count_tokens(self, text):
            raise cause

        def truncate_text(self, text, start_tokens, end_tokens=0):
            raise cause

    (tmp_path / "notes.md").write_text("# Valid document\n")
    with pytest.raises(RuntimeError, match="Tokenization failed") as failure:
        inspect_file(tmp_path, "notes.md", tokenizer=UnavailableTokenizer())
    assert failure.value.__cause__ is cause


def test_inspection_tokenizer_context_restores_state_even_after_failure():
    first, second = StubTokenizer(), PairTokenizer()
    with pytest.raises(TextTokenizationError, match="requires the model tokenizer"):
        current_inspection_tokenizer()
    with inspection_tokenizer_context(first):
        with pytest.raises(RuntimeError, match="interrupted"):
            with inspection_tokenizer_context(second):
                assert current_inspection_tokenizer() is second
                raise RuntimeError("interrupted")
        assert current_inspection_tokenizer() is first
    with pytest.raises(TextTokenizationError):
        current_inspection_tokenizer()


def test_inspection_tokenizer_context_is_isolated_between_threads():
    first, second = StubTokenizer(), PairTokenizer()
    barrier = Barrier(2)

    def count(tokenizer):
        with inspection_tokenizer_context(tokenizer):
            barrier.wait(timeout=5)
            return current_inspection_tokenizer().count_tokens("à界😀x")

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(count, (first, second))) == [4, 2]


@pytest.mark.parametrize("execution_mode", ["wait", "background"])
def test_node_supplies_its_model_tokenizer_to_file_inspection(
    tmp_path, execution_mode, monkeypatch
):
    label = "界😀àx" * 101
    (tmp_path / "notes.md").write_text(f"# {label}\n", encoding="utf-8")
    tokenizer = PairTokenizer()
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall("files", FunctionCall("file-processing", {})),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "inspect",
                        FunctionCall(
                            "inspect-file",
                            {
                                "path": "notes.md",
                                "execution_mode": execution_mode,
                            },
                        ),
                    ),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "exit",
                        FunctionCall("exit_agent", {"text": "done", "items": []}),
                    ),
                ],
            ),
            AssistantMessage("done"),
        ],
        tokenizer=tokenizer,
    )
    node = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(include={"main"}),
    )
    monkeypatch.setenv("EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS", "1500")
    node.initialize(provider, work_root=tmp_path)
    monkeypatch.setenv("EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS", "2000")

    assert node.run("Inspect notes").get_message() == "done"
    state = node.control_tool_execution("inspect", wait_seconds=1)
    report = parse_tool_result(node.read_tool_output(state["output_id"]).content)

    assert state["status"] == "completed"
    assert report["regions"][0]["name"] == label[:400]
    assert report["selection"]["labels_truncated"]
    assert report["read_plan"] == {
        "max_tokens": 1500,
        "ranges": [{"start_index": 1, "end_index": 1, "tokens": 203}],
    }
