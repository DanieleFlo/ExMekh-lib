"""Read images directly or inspection-gated text ranges and PDF pages."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from exmekh_lib.config.file_inspection import FILE_INSPECTION, FILE_INSPECTION_TEXT

from .decoding import decode_text
from .errors import FileInspectionError
from .evidence import InspectionEvidence, ReadReport, hash_content
from .image_reading import read_image
from .lines import physical_lines
from .pdf_reading import read_pdf, validate_pdf_read_arguments
from .workspace import FileSnapshot, read_snapshot, validate_file_path


class FileReadContext:
    """Expose visible proof and an explicit request for runtime recovery."""

    def __init__(self, evidence: tuple[InspectionEvidence, ...]) -> None:
        self._evidence = evidence
        self._inspection_path: str | None = None

    def visible_inspections(self) -> tuple[InspectionEvidence, ...]:
        return tuple(item for item in self._evidence if item.source == "inspect")

    def visible_evidence(self) -> tuple[InspectionEvidence, ...]:
        return self._evidence

    def request_inspection(self, path: str) -> None:
        self._inspection_path = path

    def pending_inspection_path(self) -> str | None:
        return self._inspection_path


_read_context: ContextVar[FileReadContext | None] = ContextVar(
    "visible_file_inspections", default=None
)


@contextmanager
def file_read_context(
    evidence: tuple[InspectionEvidence, ...],
) -> Generator[FileReadContext]:
    context = FileReadContext(evidence)
    token = _read_context.set(context)
    try:
        yield context
    finally:
        _read_context.reset(token)


def require_inspection(path: str) -> dict[str, object]:
    """Request runtime recovery separately from the result passed through hooks."""
    context = _read_context.get()
    if context is not None:
        context.request_inspection(path)
    indices = (
        FILE_INSPECTION_TEXT.page_indices_label
        if Path(path).suffix.casefold() == ".pdf"
        else FILE_INSPECTION_TEXT.line_indices_label
    )
    return {
        "ok": False,
        "error_kind": "file_inspection_required",
        "error": FILE_INSPECTION_TEXT.inspection_required.format(indices=indices),
    }


def validate_read_arguments(path: str, start_index: int, end_index: int) -> None:
    validate_file_path(path)
    for name, value in (("start_index", start_index), ("end_index", end_index)):
        if type(value) is not int or (value < 1 and (start_index, end_index) != (0, 0)):
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.positive_index.format(name=name),
                code="invalid_selection",
            )
    if start_index > end_index:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.ordered_indices, code="invalid_selection"
        )


def read_file(
    work_root: Path, path: str, start_index: int, end_index: int
) -> dict[str, object]:
    validate_file_path(path)
    extension = Path(path).suffix.casefold()
    if extension in FILE_INSPECTION.image_extensions:
        return read_image(
            read_snapshot(work_root, path, FILE_INSPECTION), FILE_INSPECTION
        )
    if extension == ".pdf":
        validate_pdf_read_arguments(start_index, end_index, FILE_INSPECTION)
    else:
        validate_read_arguments(path, start_index, end_index)
    # No content is decoded or selected before the prerequisite is satisfied.
    context = _read_context.get()
    evidence = context.visible_inspections() if context is not None else ()
    if not evidence:
        return require_inspection(path)
    try:
        snapshot = read_snapshot(work_root, path, FILE_INSPECTION)
    except FileInspectionError:
        return require_inspection(path)
    inspection = _find_matching_inspection(snapshot, evidence)
    if inspection is None:
        return require_inspection(path)
    if inspection.format == "pdf":
        return read_pdf(snapshot, start_index, end_index, FILE_INSPECTION)
    return _read_line_range(snapshot, inspection, start_index, end_index)


def _find_matching_inspection(
    snapshot: FileSnapshot, evidence: tuple[InspectionEvidence, ...]
) -> InspectionEvidence | None:
    if snapshot.content is None:
        return None
    digest = hash_content(snapshot.content)
    return next(
        (item for item in evidence if item.matches(snapshot.path, digest)), None
    )


def _read_line_range(
    snapshot: FileSnapshot,
    inspection: InspectionEvidence,
    start_index: int,
    end_index: int,
) -> dict[str, object]:
    if not inspection.readable:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.no_readable_lines, code="unsupported_file_format"
        )
    assert snapshot.content is not None
    decoded = decode_text(snapshot.content, snapshot.path.suffix.casefold())
    if decoded is None:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.no_readable_lines, code="unsupported_file_format"
        )
    text, encoding = decoded
    lines = physical_lines(text)
    if not lines:
        if (start_index, end_index) != (0, 0):
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.empty_file_range, code="invalid_selection"
            )
    elif start_index < 1 or end_index > len(lines):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.line_range_outside_file.format(total_lines=len(lines)),
            code="range_out_of_bounds",
        )
    return ReadReport(
        {
            "ok": True,
            "path": snapshot.relative_path,
            "encoding": encoding,
            "start_index": start_index,
            "end_index": end_index,
            "total_lines": len(lines),
            "lines": [
                {"line": number, "text": lines[number - 1]}
                for number in range(start_index, end_index + 1)
            ]
            if lines
            else [],
        },
        snapshot,
    )
