"""Orchestrate bounded inspection and select a page of navigation regions."""

import csv
import tarfile
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf.errors import PdfReadError

from exmekh_lib.config.file_inspection import (
    FILE_INSPECTION,
    FILE_INSPECTION_TEXT,
    FileInspectionSettings,
)
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.tokenizers.text_budget import TokenTextBudget
from exmekh_lib.interfaces.tokenizer import Tokenizer

from .containers import analyze_image, analyze_tar, analyze_zip
from .errors import FileInspectionError
from .evidence import InspectionReport
from .models import Analysis, RegionIndex, RegionPage
from .pdf import analyze_pdf
from .read_plan import ReadPlan, build_read_plan
from .reporting import InspectionPresentation
from .selection import validate_selection
from .text import analyze_text
from .workspace import FileSnapshot, read_snapshot


def inspect_file(
    work_root: Path,
    path: str,
    *,
    tokenizer: Tokenizer,
    detailed: bool = False,
    max_items: int | None = None,
    offset: int = 0,
    query: str | None = None,
    read_from_line: int | None = None,
    settings: FileInspectionSettings = FILE_INSPECTION,
    output_settings: ToolOutputSettings | None = None,
) -> dict[str, object]:
    if max_items is None:
        max_items = settings.default_max_items
    validate_selection(detailed, max_items, offset, query, read_from_line, settings)
    snapshot = read_snapshot(work_root, path, settings)
    analysis = _analyze(snapshot, settings, tokenizer)
    if detailed and analysis.index.is_empty() and analysis.lines:
        analysis.index.add(
            "content", FILE_INSPECTION_TEXT.content_region_label, 1, len(analysis.lines)
        )
    page = analysis.index.select(max_items=max_items, offset=offset, query=query)
    output_limit = (output_settings or ToolOutputSettings.from_environment()).max_tokens
    read_plan = _plan_reads(
        analysis, page, read_from_line, TokenTextBudget(tokenizer, output_limit)
    )
    report_budget = TokenTextBudget(
        tokenizer,
        output_limit if detailed else min(output_limit, settings.max_report_tokens),
    )
    report = InspectionPresentation(snapshot, analysis, detailed, report_budget).fit(
        page, read_plan
    )
    if snapshot.content is not None:
        return InspectionReport(report, snapshot)
    return report


def _plan_reads(
    analysis: Analysis,
    page: RegionPage,
    start_line: int | None,
    budget: TokenTextBudget,
) -> ReadPlan | None:
    lines = analysis.lines
    if lines is None or analysis.status != "complete":
        return None
    end_line = len(lines)
    if start_line is not None and start_line > max(1, end_line):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.read_from_line_outside_file.format(
                total_lines=end_line
            ),
            code="invalid_input",
        )
    match = page.first_match()
    if (
        match is not None
        and match.start_line is not None
        and match.end_line is not None
    ):
        if start_line is None or match.start_line <= start_line <= match.end_line:
            start_line = start_line or match.start_line
            end_line = match.end_line
    return build_read_plan(
        lines,
        analysis.index.line_boundaries(),
        start_line or 1,
        end_line,
        analysis.index.settings.read_ranges_per_page,
        budget,
    )


def _analyze(
    snapshot: FileSnapshot, settings: FileInspectionSettings, tokenizer: Tokenizer
) -> Analysis:
    label_budget = TokenTextBudget(tokenizer, settings.max_label_tokens)
    index = RegionIndex(settings, label_budget)
    extension = snapshot.path.suffix.casefold()
    raw = snapshot.content
    if raw is None:
        return Analysis(
            extension.lstrip(".") or "unknown",
            "unknown",
            index,
            status="skipped_limit",
            warnings=[FILE_INSPECTION_TEXT.metadata_only_warning],
        )
    try:
        if raw.startswith(b"PK\x03\x04") or extension in settings.zip_extensions:
            return analyze_zip(raw, index)
        if extension in settings.archive_extensions or raw[257:262] == b"ustar":
            return analyze_tar(raw, index)
        if extension in settings.image_extensions:
            return analyze_image(raw, index)
        if raw.startswith(b"%PDF-") or extension == ".pdf":
            return analyze_pdf(raw, index, label_budget)
        return analyze_text(raw, extension, index)
    except (
        ValueError,
        SyntaxError,
        UnicodeError,
        csv.Error,
        OSError,
        KeyError,
        EOFError,
        tarfile.TarError,
        zipfile.BadZipFile,
        PdfReadError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        return Analysis(
            extension.lstrip(".") or "unknown",
            "unknown",
            index,
            status="invalid_format",
            warnings=[
                FILE_INSPECTION_TEXT.analysis_failed_warning.format(
                    error_type=type(error).__name__
                )
            ],
        )
