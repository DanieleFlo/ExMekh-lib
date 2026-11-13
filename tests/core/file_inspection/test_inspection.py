"""Real file fixtures for bounded inspection and harness integration."""

import io
import os
import tarfile
import zipfile
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import NameObject, NumberObject, TextStringObject

from exmekh_lib.capabilities.tools.read_tool_output.tool_output_reader import (
    ToolOutputReader,
)
from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.file_inspection import workspace
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.tokenization import inspection_tokenizer_context
from exmekh_lib.core.models.annotation_schema import (
    AnnotationSchemaBuilder,
)
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool import ExecutionOptions
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from tests.capabilities_support import MOCK_CAPABILITIES_PATH
from tests.core.agents.node.test_node import SequenceProvider
from tests.file_inspection_support import inspect_file
from tests.tokenizer_support import StubTokenizer
from tests.tool_result_support import parse_tool_result


def inspect_text(tmp_path, text, filename="sample.txt", **kwargs):
    (tmp_path / filename).write_bytes(text.encode("utf-8"))
    return inspect_file(tmp_path, filename, **kwargs)


@pytest.mark.parametrize(
    ("text", "lines", "style"),
    [
        ("", 0, "none"),
        ("one", 1, "none"),
        ("one\n", 1, "lf"),
        ("\n\n", 2, "lf"),
        ("one\r\ntwo\r\n", 2, "crlf"),
        ("one\rtwo", 2, "cr"),
        ("one\r\ntwo\nthree\r", 3, "mixed"),
        ("one\u2028two\n", 1, "lf"),
    ],
)
def test_physical_lines_and_newline_style(tmp_path, text, lines, style):
    result = inspect_text(tmp_path, text, detailed=True)
    assert result["total_lines"] == lines
    assert result["analysis"]["newline_style"] == style
    assert result["regions"] == (
        [
            {
                "kind": "content",
                "name": "file content",
                "start_line": 1,
                "end_line": lines,
            }
        ]
        if lines
        else []
    )


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_unicode_boms_do_not_pollute_the_first_region(tmp_path, encoding):
    (tmp_path / "sample.md").write_bytes("# Title\nbody\n".encode(encoding))
    result = inspect_file(tmp_path, "sample.md")
    assert result["analysis"]["encoding"] == encoding
    assert result["regions"][0]["name"] == "Title"


def test_python_symbols_include_decorators_and_qualified_nested_names(tmp_path):
    result = inspect_text(
        tmp_path,
        (
            "class Service:\n"
            "    @property\n"
            "    def token(self):\n"
            "        def inner():\n"
            "            return 1\n"
            "        return inner()\n"
            "\n"
            "async def run():\n"
            "    pass\n"
        ),
        "code.py",
    )
    assert [(r["name"], r["start_line"], r["end_line"]) for r in result["regions"]] == [
        ("Service", 1, 6),
        ("Service.token", 2, 6),
        ("Service.token.inner", 4, 5),
        ("run", 8, 9),
    ]
    assert result["analysis"]["syntax_valid"] is True


def test_python_coding_cookie_is_respected_without_execution(tmp_path):
    (tmp_path / "latin.py").write_bytes(
        b"# coding: latin-1\n# caf\xe9\ndef run():\n    raise RuntimeError('never execute')\n"
    )
    result = inspect_file(tmp_path, "latin.py")
    assert result["analysis"]["encoding"] == "iso-8859-1"
    assert result["regions"][0]["name"] == "run"


def test_invalid_python_preserves_text_metrics(tmp_path):
    result = inspect_text(tmp_path, "def broken(:\n", "code.py")
    assert result["analysis_status"] == "invalid_format"
    assert result["analysis"]["syntax_valid"] is False
    assert result["analysis"]["error_line"] == 1
    assert result["total_lines"] == 1


def test_markdown_ignores_headings_in_fences_and_handles_setext_and_nested_sections(
    tmp_path,
):
    result = inspect_text(
        tmp_path,
        (
            "# Main\n\n````python\n# fake\n```\n````\n"
            "## Child\nbody\nNext\n====\n~~~\n# hidden\n"
        ),
        "notes.md",
    )
    assert [
        (r["kind"], r["name"], r["start_line"], r["end_line"])
        for r in result["regions"]
    ] == [
        ("section", "Main", 1, 8),
        ("code_fence", "python", 3, 6),
        ("section", "Child", 7, 8),
        ("section", "Next", 9, 12),
        ("code_fence", "untyped", 11, 12),
    ]


@pytest.mark.parametrize(
    ("filename", "text", "names"),
    [
        (
            "data.json",
            '{\n  "a": {\n    "b": 1\n  },\n  "items": [{"name": "secret-value"}]\n}\n',
            ["/a", "/a/b", "/items", "/items[0]/name"],
        ),
        ("data.yaml", "server:\n  host: secret-value\n", ["/server", "/server/host"]),
        ("data.toml", '[server]\nhost="secret-value"\n', ["server", "server.host"]),
        (
            "data.xml",
            "<root>\n  <item>secret-value</item>\n</root>\n",
            ["root", "item"],
        ),
    ],
)
def test_structured_navigation_omits_values(tmp_path, filename, text, names):
    result = inspect_text(tmp_path, text, filename)
    assert result["analysis_status"] == "complete"
    assert [r["name"] for r in result["regions"]] == names
    assert "secret-value" not in repr(result)
    assert all(
        1 <= r["start_line"] <= r["end_line"] <= result["total_lines"]
        for r in result["regions"]
    )


def test_json_key_ranges_are_inclusive(tmp_path):
    result = inspect_text(tmp_path, '{\n  "a": {\n    "b": 1\n  }\n}\n', "data.json")
    assert [(r["start_line"], r["end_line"]) for r in result["regions"]] == [
        (2, 4),
        (3, 3),
    ]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("data.json", "{no json}"),
        ("data.yaml", "a: ["),
        ("data.toml", "[broken"),
        ("data.xml", "<root>"),
        ("data.csv", '"unterminated'),
    ],
)
def test_malformed_text_formats_report_errors_with_line_counts(
    tmp_path, filename, content
):
    result = inspect_text(tmp_path, content, filename)
    assert result["analysis_status"] == "invalid_format"
    assert result["total_lines"] == 1
    assert result["warnings"]


def test_xml_entities_are_never_expanded(tmp_path):
    result = inspect_text(
        tmp_path,
        '<!DOCTYPE root [<!ENTITY x SYSTEM "file:///etc/passwd">]><root>&x;</root>',
        "data.xml",
    )
    assert result["analysis_status"] == "invalid_format"
    assert "root:" not in repr(result)


def test_yaml_cycles_and_alias_expansion_have_explicit_limits(tmp_path):
    cycle = inspect_text(tmp_path, "a: &a\n  again: *a\n  repeated: *a\n", "cycle.yaml")
    assert "Recursive YAML alias" in repr(cycle["warnings"])
    bomb = "a: &a [1, 2]\nb: &b [*a, *a]\nc: [*b, *b]\n"
    result = inspect_text(
        tmp_path,
        bomb,
        "aliases.yaml",
        settings=replace(FILE_INSPECTION, max_structure_nodes=5),
    )
    assert result["selection"]["index_truncated"]


def test_structural_depth_limit_is_reported(tmp_path):
    result = inspect_text(
        tmp_path,
        "a:\n  b:\n    c: 1\n",
        "data.yaml",
        settings=replace(FILE_INSPECTION, max_structure_depth=1),
    )
    assert result["selection"]["index_truncated"]


def test_csv_multiline_records_and_inconsistent_columns(tmp_path):
    result = inspect_text(
        tmp_path, 'name,value\n"two\nlines",1\nlast,2,extra\n', "table.csv"
    )
    assert result["total_lines"] == 4
    assert result["analysis"]["records"] == 3
    assert result["analysis"]["columns"] == 3
    assert result["analysis"]["inconsistent_columns"]
    assert max(region["end_line"] for region in result["regions"]) == 4


def test_tsv_and_complete_record_chunks(tmp_path):
    result = inspect_text(
        tmp_path,
        "a\tb\n1\t2\n3\t4\n5\t6\n",
        "data.tsv",
        settings=replace(
            FILE_INSPECTION, table_rows_per_region=2, max_report_tokens=5000
        ),
    )
    assert result["analysis"]["delimiter"] == "\t"
    assert [
        (r["start_line"], r["end_line"])
        for r in result["regions"]
        if r["kind"] == "rows"
    ] == [(1, 2), (3, 4)]


def test_pagination_and_query_search_the_index_before_slicing(tmp_path):
    text = "# First\n# Second\n# DecodeUserMessage\n"
    first = inspect_text(tmp_path, text, "notes.md", max_items=1)
    assert first["selection"]["next_offset"] == 1
    second = inspect_file(tmp_path, "notes.md", max_items=1, offset=1)
    assert second["regions"][0]["name"] == "Second"
    ranked = inspect_file(tmp_path, "notes.md", max_items=1, query="user message")
    assert ranked["regions"][0]["name"] == "DecodeUserMessage"
    assert ranked["selection"]["ranking"] == "lexical"
    assert ranked["selection"]["matching_items"] == 1
    assert ranked["read_plan"]["ranges"][0]["start_index"] == 3
    unmatched = inspect_file(tmp_path, "notes.md", query="missing")
    assert unmatched["selection"]["matching_items"] == 0
    assert unmatched["regions"] == []
    assert unmatched["read_plan"]["ranges"][0]["start_index"] == 1
    resumed = inspect_file(tmp_path, "notes.md", query="user", read_from_line=2)
    assert resumed["read_plan"]["ranges"][0]["start_index"] == 2
    with pytest.raises(FileInspectionError, match="3 physical lines"):
        inspect_file(tmp_path, "notes.md", read_from_line=4)
    empty = inspect_file(tmp_path, "notes.md", offset=100)
    assert empty["regions"] == []
    assert empty["selection"]["has_more"] is False


def test_default_page_size_comes_from_supplied_settings(tmp_path):
    result = inspect_text(
        tmp_path,
        "# First\n# Second\n",
        "notes.md",
        settings=replace(
            FILE_INSPECTION, default_max_items=1, max_items=1, max_report_tokens=5000
        ),
    )
    assert [region["name"] for region in result["regions"]] == ["First"]
    assert result["selection"]["next_offset"] == 1


def test_index_limit_differs_from_pagination_and_labels_are_bounded(tmp_path):
    result = inspect_text(
        tmp_path,
        "# AAAAAAAA\n# B\n# C\n",
        "notes.md",
        max_items=1,
        settings=replace(FILE_INSPECTION, max_regions=2, max_label_tokens=3),
    )
    assert result["regions"][0]["name"] == "AAA"
    assert result["selection"]["index_truncated"]
    assert result["selection"]["indexed_items"] == 2
    assert result["selection"]["has_more"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items": 0},
        {"max_items": 501},
        {"max_items": True},
        {"max_items": 1.2},
        {"offset": -1},
        {"offset": True},
        {"offset": "1"},
        {"read_from_line": 0},
        {"read_from_line": True},
        {"read_from_line": "2"},
        {"detailed": 1},
        {"query": ""},
        {"query": "   "},
        {"query": "!!!"},
        {"query": 1},
        {"query": "a" * 1001},
    ],
)
def test_invalid_selections_fail_before_file_access(tmp_path, kwargs):
    with pytest.raises(FileInspectionError):
        inspect_file(tmp_path, "missing", **kwargs)


@pytest.mark.parametrize("path", ["", " ", "missing", ".", "../outside", "\x00"])
def test_invalid_paths_are_rejected(tmp_path, path):
    with pytest.raises(FileInspectionError):
        inspect_file(tmp_path, path)


def test_internal_symlinks_absolute_paths_and_spaces(tmp_path):
    file = tmp_path / " name.txt "
    file.write_text("hello")
    (tmp_path / "link").symlink_to(file)
    result = inspect_file(tmp_path, "link", detailed=True)
    assert result["path"] == " name.txt "
    assert result["metadata"]["is_symlink"]
    assert inspect_file(tmp_path, str(file))["path"] == " name.txt "
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_text("never read")
    try:
        (tmp_path / "escape").symlink_to(outside)
        with pytest.raises(FileInspectionError, match="workroot"):
            inspect_file(tmp_path, "escape")
    finally:
        outside.unlink()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_special_files_are_rejected_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(FileInspectionError, match="regular"):
        inspect_file(tmp_path, "fifo")


def test_metadata_and_size_limit_do_not_read_oversized_content(tmp_path):
    result = inspect_text(
        tmp_path,
        "too large",
        settings=replace(FILE_INSPECTION, max_file_bytes=1),
        detailed=True,
    )
    assert result["analysis_status"] == "skipped_limit"
    assert "total_lines" not in result
    assert result["metadata"]["modified_at"].endswith("+00:00")
    assert result["metadata"]["created_at"] is None or isinstance(
        result["metadata"]["created_at"], str
    )
    assert result["limits"]["max_file_bytes"] == 1


@pytest.mark.parametrize("raw", [b"\x00\x01", b"\xff\x80"])
def test_binary_data_is_not_guessed_as_text(tmp_path, raw):
    (tmp_path / "unknown.bin").write_bytes(raw)
    result = inspect_file(tmp_path, "unknown.bin")
    assert result["analysis_status"] == "unsupported"
    assert "total_lines" not in result


def test_zip_indexes_are_not_extracted_and_report_unsafe_paths(tmp_path):
    file = tmp_path / "bundle.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("safe/", "")
        archive.writestr("../escape", "secret")
        archive.writestr("C:\\escape", "secret")
    result = inspect_file(
        tmp_path,
        file.name,
        settings=replace(FILE_INSPECTION, max_archive_entries=2, max_expanded_bytes=1),
    )
    assert result["analysis"]["entries"] == 3
    assert result["analysis"]["unsafe_paths"] == 2
    assert result["selection"]["index_truncated"]
    assert result["warnings"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle.zip"]


def test_tar_index_stops_before_expanding_an_oversized_member(tmp_path):
    file = tmp_path / "bundle.tar.gz"
    with tarfile.open(file, "w:gz") as archive:
        member = tarfile.TarInfo("data.txt")
        member.size = 100
        archive.addfile(member, io.BytesIO(b"x" * 100))
    result = inspect_file(
        tmp_path, file.name, settings=replace(FILE_INSPECTION, max_expanded_bytes=10)
    )
    assert result["analysis_status"] == "skipped_limit"
    assert result["selection"]["index_truncated"]


def test_tar_scan_has_complete_totals_when_within_limits(tmp_path):
    file = tmp_path / "bundle.tar"
    with tarfile.open(file, "w") as archive:
        archive.addfile(tarfile.TarInfo("folder"))
    result = inspect_file(tmp_path, file.name)
    assert result["analysis_status"] == "complete"
    assert result["analysis"]["entries_scanned"] == 1
    assert result["regions"][0]["name"] == "folder"


def test_image_metadata_does_not_return_pixels(tmp_path):
    file = tmp_path / "image.png"
    Image.new("RGB", (10, 20)).save(file)
    result = inspect_file(tmp_path, file.name)
    assert result["analysis"] == {
        "width": 10,
        "height": 20,
        "mode": "RGB",
        "animated": False,
    }
    assert result["regions"] == []
    limited = inspect_file(
        tmp_path, file.name, settings=replace(FILE_INSPECTION, max_image_pixels=1)
    )
    assert limited["analysis_status"] == "skipped_limit"


@pytest.mark.parametrize("extension", [".zip", ".tar", ".png", ".pdf", ".py"])
def test_invalid_binary_formats_are_explicit(tmp_path, extension):
    file = tmp_path / ("bad" + extension)
    file.write_bytes(b"not the indicated format\xff")
    result = inspect_file(tmp_path, file.name)
    assert result["analysis_status"] == "invalid_format"
    assert result["warnings"]


def write_pdf(path, *, encrypted=False):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.add_metadata({"/Title": "A <title>", "/Author": "Someone"})
    if encrypted:
        writer.encrypt("password")
    writer.write(path)


def test_pdf_returns_only_metadata_without_text_or_rendering(tmp_path):
    file = tmp_path / "report.pdf"
    write_pdf(file)
    result = inspect_file(tmp_path, file.name)
    assert result["format"] == "pdf"
    assert result["analysis"]["page_count"] == 1
    assert result["analysis"]["encrypted"] is False
    assert result["analysis"]["properties"]["title"] == "A <title>"
    assert result["analysis"]["first_page_size_points"] == {
        "width": 100.0,
        "height": 200.0,
    }
    assert "total_lines" not in result
    assert result["regions"] == []


def test_encrypted_and_large_pdf_have_explicit_metadata_limits(tmp_path):
    encrypted = tmp_path / "locked.pdf"
    write_pdf(encrypted, encrypted=True)
    result = inspect_file(tmp_path, encrypted.name)
    assert result["analysis"] == {"encrypted": True}
    assert result["warnings"]
    file = tmp_path / "report.pdf"
    write_pdf(file)
    result = inspect_file(
        tmp_path, file.name, settings=replace(FILE_INSPECTION, max_pdf_pages=0)
    )
    assert result["analysis_status"] == "skipped_limit"
    assert result["analysis"]["page_count"] == 1


@pytest.fixture
def bundled_tool(tmp_path):
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    with inspection_tokenizer_context(StubTokenizer()):
        yield next(tool for tool in loaded.tools if tool.name == "inspect-file")


def test_bundled_tool_schema_and_structured_errors(bundled_tool):
    mapper = OpenAIFunctionToolDefinitionMapper(AnnotationSchemaBuilder())
    schema = mapper.map(bundled_tool, strict=True).get_parameters()
    assert set(schema["properties"]) == {
        "path",
        "detailed",
        "max_items",
        "offset",
        "query",
        "read_from_line",
        "execution_mode",
        "wait_seconds",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert (
        bundled_tool.execute(path="missing")["error_kind"] == "invalid_file_inspection"
    )


def test_background_result_is_retrievable_as_xml(bundled_tool, tmp_path):
    (tmp_path / "notes.md").write_text("# A <title>\n")
    manager = ToolExecutionManager()
    operation = bundled_tool.create_execution_with_hooks(
        {"path": "notes.md"}, call_id="inspect-call", work_root=tmp_path, history=()
    )
    operation.add_run_context(inspection_tokenizer_context(StubTokenizer()))
    snapshot = manager.start(
        "inspect-call",
        "inspect-file",
        {"path": "notes.md"},
        operation,
        ("node",),
        ExecutionOptions(mode="background"),
        output_id="inspect-output",
    )
    assert snapshot.status in {"running", "completed"}
    assert manager.wait("inspect-call", 5).status == "completed"
    reader = ToolOutputReader(lambda: None, manager)
    content = reader.read("inspect-output").content
    assert "<response-type>final</response-type>" in content
    assert parse_tool_result(content)["regions"][0]["name"] == "A <title>"


def test_file_processing_skill_inspects_exits_and_rereads_saved_xml(tmp_path):
    (tmp_path / "notes.md").write_text("# A <title>\n")
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall("open-files", FunctionCall("file-processing", {}))
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "inspect-call",
                        FunctionCall("inspect-file", {"path": "notes.md"}),
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "exit-files",
                        FunctionCall(
                            "exit_agent", {"text": "Found the title.", "items": []}
                        ),
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node(
        [MOCK_CAPABILITIES_PATH],
        name_hds="main",
        default_capabilities=DefaultCapabilities(),
    )
    loaded = node.initialize(provider, work_root=tmp_path)
    skill = next(
        skill for skill in loaded.deep_skills if skill.name == "file-processing"
    )
    assert skill.get_sub_tool_names() == (
        "inspect-file",
        "read-file",
        "edit-file",
        "new-file",
        "delete-file",
    )
    assert not skill.inherits_parent_tools()
    assert node.run("inspect").get_message() == "done"
    root_tools = set(provider.requests[0].get_extra()["agentic"]["allowed_tools"])
    assert "mock-tool" in root_tools
    assert "inspect-file" not in root_tools
    skill_tools = set(provider.requests[1].get_extra()["agentic"]["allowed_tools"])
    assert skill_tools == (root_tools - {"mock-tool"}) | {
        "inspect-file",
        "read-file",
        "edit-file",
        "new-file",
        "delete-file",
        "exit_agent",
    }
    result = next(
        m
        for m in provider.requests[2].get_messages()
        if isinstance(m, ToolMessage) and m.get_tool_call_id() == "inspect-call"
    )
    content = result.get_message()
    assert parse_tool_result(content)["path"] == "notes.md"
    header = ET.fromstring(content.split("</header>", 1)[0] + "</header>")
    output_id = header.findtext("id")
    assert (
        set(provider.requests[3].get_extra()["agentic"]["allowed_tools"]) == root_tools
    )
    assert not any(
        isinstance(m, ToolMessage) and m.get_tool_call_id() == "inspect-call"
        for m in provider.requests[3].get_messages()
    )
    provider._responses.extend(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "read-call",
                        FunctionCall("read_tool_output", {"output_id": output_id}),
                    )
                ],
            ),
            AssistantMessage("reread"),
        ]
    )
    assert node.run("reread").get_message() == "reread"
    reread = next(
        m
        for m in reversed(provider.requests[-1].get_messages())
        if isinstance(m, ToolMessage)
    )
    assert parse_tool_result(reread.get_message())["regions"][0]["name"] == "A <title>"


def test_file_processing_skill_can_be_selected_without_other_harness_skills(tmp_path):
    loaded = Capabilities(
        None,
        default_capabilities=DefaultCapabilities(
            harness=False,
            include={
                "file-processing",
                "inspect-file",
                "read-file",
                "edit-file",
                "new-file",
                "delete-file",
            },
        ),
    ).load(work_root=tmp_path)
    assert {skill.name for skill in loaded.deep_skills} == {"file-processing"}
    assert {tool.name for tool in loaded.tools} == {
        "inspect-file",
        "read-file",
        "edit-file",
        "new-file",
        "delete-file",
    }


@pytest.mark.parametrize("filename", ["empty.yaml", "empty.csv", "empty.toml"])
def test_empty_structured_files_have_no_navigation(tmp_path, filename):
    result = inspect_text(tmp_path, "", filename)
    assert result["analysis_status"] == "complete"
    assert result["regions"] == []


def test_toml_arrays_and_markup_depth_limits(tmp_path):
    result = inspect_text(tmp_path, '[[servers]]\nhost="value"\n', "array.toml")
    assert any("servers[0].host" == region["name"] for region in result["regions"])
    limited = inspect_file(
        tmp_path, "array.toml", settings=replace(FILE_INSPECTION, max_structure_depth=0)
    )
    assert limited["selection"]["index_truncated"]
    xml = inspect_text(
        tmp_path,
        "<root><child/></root>",
        "deep.xml",
        settings=replace(FILE_INSPECTION, max_structure_depth=1),
    )
    assert xml["selection"]["index_truncated"]
    assert [region["name"] for region in xml["regions"]] == ["root"]


def test_mapping_node_budget_stops_wide_maps(tmp_path):
    result = inspect_text(
        tmp_path,
        "a: 1\nb: 2\nc: 3\n",
        "wide.yaml",
        settings=replace(FILE_INSPECTION, max_structure_nodes=1),
    )
    assert result["selection"]["index_truncated"]
    assert len(result["regions"]) == 1


def test_json_parser_recursion_limit_is_a_recoverable_analysis_status(tmp_path):
    result = inspect_text(tmp_path, "[" * 2000 + "0" + "]" * 2000, "deep.json")
    assert result["analysis_status"] == "skipped_limit"
    assert result["total_lines"] == 1


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("w:gz", ".tar.gz"), ("w:bz2", ".tar.bz2"), ("w:xz", ".tar.xz")],
)
def test_compressed_tar_formats_are_scanned_with_bounded_decompression(
    tmp_path, mode, suffix
):
    file = tmp_path / ("bundle" + suffix)
    with tarfile.open(file, mode) as archive:
        member = tarfile.TarInfo("folder/")
        member.type = tarfile.DIRTYPE
        archive.addfile(member)
    result = inspect_file(tmp_path, file.name)
    assert result["analysis_status"] == "complete"
    assert result["regions"][0]["kind"] == "directory"


def test_uncompressed_tar_member_and_entry_limits_are_explicit(tmp_path):
    file = tmp_path / "bundle.tar"
    with tarfile.open(file, "w") as archive:
        member = tarfile.TarInfo("data.txt")
        member.size = 100
        archive.addfile(member, io.BytesIO(b"x" * 100))
        archive.addfile(tarfile.TarInfo("second.txt"))
    for settings in [
        replace(FILE_INSPECTION, max_archive_entries=1),
        replace(FILE_INSPECTION, max_expanded_bytes=10),
    ]:
        result = inspect_file(tmp_path, file.name, settings=settings)
        assert result["analysis_status"] == "skipped_limit"
        assert result["analysis"]["entries_scanned"] == 1


def test_small_zip_does_not_report_a_limit(tmp_path):
    file = tmp_path / "bundle.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("safe.txt", "value")
    result = inspect_file(tmp_path, file.name)
    assert result["analysis_status"] == "complete"
    assert result["warnings"] == []
    assert result["selection"]["index_truncated"] is False


def test_pdf_without_pages_or_metadata_is_supported(tmp_path):
    file = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.metadata = None
    writer.write(file)
    result = inspect_file(tmp_path, file.name)
    assert result["analysis"] == {"encrypted": False, "page_count": 0}


@pytest.mark.parametrize("invalid_tree", [False, True])
def test_invalid_pdf_page_tree_is_reported(tmp_path, invalid_tree):
    file = tmp_path / "bad.pdf"
    writer = PdfWriter()
    if invalid_tree:
        writer.root_object[NameObject("/Pages")] = TextStringObject("invalid")
    else:
        writer.root_object["/Pages"][NameObject("/Count")] = NumberObject(-1)
    writer.write(file)
    result = inspect_file(tmp_path, file.name)
    assert result["analysis_status"] == "invalid_format"


def test_permission_errors_have_actionable_tool_errors(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("hello")

    def denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(workspace.os, "open", denied)
    with pytest.raises(FileInspectionError, match="permissions"):
        inspect_file(tmp_path, "sample.txt")


@pytest.mark.parametrize(
    "scenario", ["special_descriptor", "changed_descriptor", "grew", "changed_path"]
)
def test_file_replacement_and_growth_are_detected(tmp_path, monkeypatch, scenario):
    file = tmp_path / "sample.txt"
    file.write_bytes(b"hello")
    real_fstat = workspace.os.fstat
    calls = 0

    def fstat(descriptor):
        nonlocal calls
        calls += 1
        actual = real_fstat(descriptor)
        values = {
            name: getattr(actual, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_mode",
            )
        }
        if scenario == "special_descriptor":
            values["st_mode"] = 0
        elif scenario == "changed_descriptor" and calls == 2:
            values["st_size"] += 1
        elif scenario == "grew":
            values["st_size"] = 0
        elif scenario == "changed_path":
            values["st_ino"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(workspace.os, "fstat", fstat)
    with pytest.raises(FileInspectionError, match="regular|changed"):
        inspect_file(
            tmp_path, file.name, settings=replace(FILE_INSPECTION, max_file_bytes=1)
        )
