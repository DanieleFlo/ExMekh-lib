"""Parse structured formats without evaluating or returning their values."""

import csv
import io
import json
from collections import Counter
from dataclasses import replace
from xml.parsers import expat

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT

from .lines import SourceLines
from .models import Analysis, Region


def analyze_mapping(text: str, result: Analysis) -> None:
    if result.format == "json":
        # YAML supplies source marks; JSON validation preserves JSON's grammar.
        json.loads(text)
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    lines = SourceLines(text)
    counts: Counter[str] = Counter()
    visited = 0

    def visit(node: Node, name: str, ancestors: frozenset[int], depth: int) -> None:
        nonlocal visited
        visited += 1
        if visited > result.index.settings.max_structure_nodes:
            result.index.mark_truncated()
            return
        counts[getattr(node, "id", "unknown")] += 1
        if id(node) in ancestors:
            warning = FILE_INSPECTION_TEXT.recursive_yaml_warning
            if warning not in result.warnings:
                result.warnings.append(warning)
            return
        if depth > result.index.settings.max_structure_depth:
            result.index.mark_truncated()
            return
        lineage = ancestors | {id(node)}
        if isinstance(node, MappingNode):
            for key, value in node.value:
                if visited > result.index.settings.max_structure_nodes:
                    break
                label = (
                    key.value
                    if isinstance(key, ScalarNode)
                    else FILE_INSPECTION_TEXT.complex_key_label
                )
                child_name = f"{name}/{label}"
                start = lines.line_number(key.start_mark.index)
                end = max(start, lines.line_number(value.end_mark.index - 1))
                result.index.add("key", child_name, start, end)
                visit(value, child_name, lineage, depth + 1)
        elif isinstance(node, SequenceNode):
            for number, child in enumerate(node.value):
                if visited > result.index.settings.max_structure_nodes:
                    break
                visit(child, f"{name}[{number}]", lineage, depth + 1)

    if root is not None:
        visit(root, "", frozenset(), 1)
    result.details.update(
        root_kind=root.id if root else "empty", node_counts=dict(counts)
    )


def analyze_json_lines(lines: list[str], result: Analysis) -> None:
    records = invalid = 0
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if records == result.index.settings.max_structure_nodes:
            result.index.mark_truncated()
            break
        records += 1
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            invalid += 1
            result.index.add("invalid_record", f"record {records}", number, number)
            continue
        name = f"record {records}"
        if isinstance(value, dict) and value:
            name += " · " + ", ".join(value)
        result.index.add("record", name, number, number)
    result.details.update(records_scanned=records, invalid_records=invalid)
    if invalid:
        result.warnings.append(FILE_INSPECTION_TEXT.invalid_json_lines_warning)


def analyze_xml(text: str, result: Analysis) -> None:
    parser = expat.ParserCreate()
    stack: list[int | None] = []
    regions: list[Region] = []
    element_count = 0

    def reject_doctype(*args: object) -> None:
        raise ValueError(FILE_INSPECTION_TEXT.xml_entities_forbidden)

    def start(name: str, attributes: dict[str, str]) -> None:
        nonlocal element_count
        del attributes
        element_count += 1
        if (
            len(stack) >= result.index.settings.max_structure_depth
            or len(regions) >= result.index.settings.max_regions
        ):
            result.index.mark_truncated()
            stack.append(None)
            return
        stack.append(len(regions))
        regions.append(Region("element", name, parser.CurrentLineNumber))

    def end(name: str) -> None:
        del name
        position = stack.pop()
        if position is not None:
            regions[position] = replace(
                regions[position], end_line=parser.CurrentLineNumber
            )

    parser.StartDoctypeDeclHandler = reject_doctype
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.Parse(text, True)
    for region in regions:
        result.index.add(region.kind, region.name, region.start_line, region.end_line)
    result.details["element_count"] = element_count


def analyze_table(text: str, result: Analysis) -> None:
    sample = text[:8192]
    delimiter = "\t" if result.format == "tsv" else ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    rows = 0
    columns = 0
    widths: set[int] = set()
    chunk_start = 1
    previous_end = 0
    for row in reader:
        start = previous_end + 1
        previous_end = reader.line_num
        columns = max(columns, len(row))
        widths.add(len(row))
        rows += 1
        if rows == 1 and has_header:
            result.index.add(
                "header",
                FILE_INSPECTION_TEXT.table_header_label,
                start,
                reader.line_num,
            )
        if rows % result.index.settings.table_rows_per_region == 0:
            result.index.add(
                "rows",
                FILE_INSPECTION_TEXT.table_records_label.format(
                    start=rows - result.index.settings.table_rows_per_region + 1,
                    end=rows,
                ),
                chunk_start,
                reader.line_num,
            )
            chunk_start = reader.line_num + 1
    if chunk_start <= previous_end:
        first = rows - rows % result.index.settings.table_rows_per_region + 1
        result.index.add(
            "rows",
            FILE_INSPECTION_TEXT.table_records_label.format(start=first, end=rows),
            chunk_start,
            previous_end,
        )
    result.details.update(
        records=rows,
        columns=columns,
        inconsistent_columns=len(widths) > 1,
        delimiter=delimiter,
        header_detected=has_header,
    )
