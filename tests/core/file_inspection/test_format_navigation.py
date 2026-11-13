"""Targeted navigation for source, markup and line-oriented records."""

from dataclasses import replace

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from tests.file_inspection_support import inspect_file


@pytest.mark.parametrize(
    "path, text, query, kind, name, start, end",
    [
        (
            "app.mjs",
            '// function fake() {}\r\nconst s = "}";\rexport const loadData = () => {\n'
            "  return s;\n};\n",
            "load data",
            "function",
            "loadData",
            3,
            5,
        ),
        (
            "model.ts",
            'import { State } from "./state";\nexport interface Item {\n state: State;\n}\n',
            "item",
            "interface",
            "Item",
            2,
            4,
        ),
        (
            "view.tsx",
            'export const View = () => (\n  <main>{"}"}</main>\n);\n',
            "view",
            "function",
            "View",
            1,
            3,
        ),
        (
            "default.tsx",
            "export default () => <section>Welcome</section>;\n",
            "default",
            "function",
            "default",
            1,
            1,
        ),
        (
            "service.ts",
            "export class Service {\n  readData = () => {\n    return true;\n  };\n}\n",
            "read data",
            "method",
            "Service.readData",
            2,
            4,
        ),
        (
            "nested.html",
            "<main>\n<h2>Account <em>Security</em> &amp; access</h2>\n</main>\n",
            "security",
            "heading",
            "h2 Account Security & access",
            2,
            2,
        ),
        (
            "declarations.c",
            "#include <stdio.h>\r\nint first(void), (second)(void), (*third)(void);\r\n",
            "second",
            "function",
            "second",
            2,
            2,
        ),
        (
            "index.html",
            '<!-- <main id="decoy"> -->\n<main id="catalog">\n'
            '<script>const x = "<section>";</script>\n<h2>Products</h2>\n</main>\n',
            "catalog",
            "element",
            "main#catalog",
            2,
            5,
        ),
        (
            "style.css",
            "/* #decoy {} */\n@media (min-width: 10px) {\n .search-box {\n"
            '  content: "}";\n }\n}\n',
            "search box",
            "selector",
            ".search-box",
            3,
            5,
        ),
        (
            "api.h",
            '#include "other.h"\n#define LIMIT(x) x\n'
            "static int (*makeHandler(int n))(int) {\n  return 0;\n}\n",
            "make handler",
            "function",
            "makeHandler",
            3,
            5,
        ),
    ],
)
def test_format_regions_feed_the_read_plan(
    tmp_path, path, text, query, kind, name, start, end
):
    (tmp_path / path).write_bytes(text.encode())

    report = inspect_file(tmp_path, path, query=query, max_items=1)

    assert report["analysis"]["syntax_recovered"] is False
    assert report["selection"]["matching_items"] == 1
    assert report["regions"] == [
        {"kind": kind, "name": name, "start_line": start, "end_line": end}
    ]
    assert report["read_plan"]["ranges"][0]["start_index"] == start


def test_html_indexes_headings_and_ignores_script_text(tmp_path):
    (tmp_path / "index.htm").write_text(
        '<script>const tag = "<nav>";</script>\n<h2>Product catalog</h2>\n'
        '<input id="search">\n'
    )

    report = inspect_file(tmp_path, "index.htm", query="product catalog")

    assert [region["name"] for region in report["regions"]] == ["h2 Product catalog"]
    unfiltered = inspect_file(tmp_path, "index.htm")
    assert [region["name"] for region in unfiltered["regions"]] == [
        "script",
        "h2 Product catalog",
        "input#search",
    ]


def test_source_navigation_keeps_recovered_regions_readable(tmp_path):
    (tmp_path / "app.js").write_text("function first() {}\nconst incomplete =\n")
    report = inspect_file(tmp_path, "app.js", query="first")

    assert report["analysis"]["syntax_recovered"]
    assert report["warnings"] == [
        "The parser recovered from syntax errors; some navigation regions may be missing."
    ]
    assert report["regions"][0]["name"] == "first"
    with file_read_context((report.evidence,)):
        assert (
            read_file(tmp_path, "app.js", 1, 1).selected_text() == "function first() {}"
        )


def test_source_navigation_obeys_the_index_budget(tmp_path):
    (tmp_path / "style.css").write_text(".first {}\n.second {}\n")
    report = inspect_file(
        tmp_path, "style.css", settings=replace(FILE_INSPECTION, max_regions=1)
    )

    assert report["selection"]["index_truncated"]
    assert [item["name"] for item in report["regions"]] == [".first"]


def test_css_indexes_named_at_rules(tmp_path):
    (tmp_path / "style.css").write_text(
        '@import "base.css";\n@layer theme {\n.card {}\n}\n'
    )
    report = inspect_file(tmp_path, "style.css", query="theme")

    assert report["regions"][0] == {
        "kind": "at_rule",
        "name": "@layer theme",
        "start_line": 2,
        "end_line": 4,
    }
    unfiltered = inspect_file(tmp_path, "style.css")
    assert {region["name"] for region in unfiltered["regions"]} == {
        '@import "base.css"',
        "@layer theme",
        ".card",
    }


def test_c_distinguishes_prototypes_from_function_pointer_variables(tmp_path):
    (tmp_path / "api.c").write_text(
        "#define VALUE 42\nint declared(void);\nint (*callback)(void);\n"
    )

    report = inspect_file(tmp_path, "api.c")

    assert [(region["kind"], region["name"]) for region in report["regions"]] == [
        ("macro", "VALUE"),
        ("function", "declared"),
    ]


def test_json_lines_indexes_keys_and_keeps_bad_records_locatable(tmp_path):
    (tmp_path / "events.ndjson").write_text(
        '{"eventType":"secret-value"}\n\n{invalid\n{"traceId":"private"}\n'
    )
    report = inspect_file(tmp_path, "events.ndjson", query="trace id")

    assert report["format"] == "jsonl"
    assert report["analysis"]["records_scanned"] == 3
    assert report["analysis"]["invalid_records"] == 1
    assert report["warnings"]
    assert [(region["kind"], region["start_line"]) for region in report["regions"]] == [
        ("record", 4),
    ]
    invalid = inspect_file(tmp_path, "events.ndjson", query="invalid record")
    assert [
        (region["kind"], region["start_line"]) for region in invalid["regions"]
    ] == [("invalid_record", 3)]
    assert "private" not in str(report)
    assert "secret-value" not in str(report)
    assert report["read_plan"]["ranges"][0]["start_index"] == 4
