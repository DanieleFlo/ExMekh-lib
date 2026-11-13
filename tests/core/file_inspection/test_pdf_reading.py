"""Inspection-gated PDF page rendering and bounded multimodal delivery."""

import builtins
from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock

import pypdfium2 as pdfium
import pytest
from PIL import Image
from pypdf import PageObject, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection import reading
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.pdf_reading import read_pdf
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.file_inspection.workspace import read_snapshot
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool import AttachedToolResult
from tests.core.file_inspection.test_reading import (
    assert_paired,
    batch,
    call,
    make_node,
    open_files,
    read_call,
    replies,
    result_for,
)
from tests.file_inspection_support import inspect_file
from tests.tool_result_support import parse_tool_result


def write_pdf(path, count=7, password=None):
    writer = PdfWriter()
    for number in range(count):
        page = PageObject.create_blank_page(width=72, height=144)
        contents = DecodedStreamObject()
        contents.set_data(
            f"{number % 2} 0 {1 - number % 2} rg 0 0 72 144 re f".encode()
        )
        page[NameObject("/Contents")] = contents
        writer.add_page(page)
    if password is not None:
        writer.encrypt(password)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def images(result):
    return [
        p for p in result.attachments.iter_content_parts() if isinstance(p, Image.Image)
    ]


def inspected_read(path, start, end):
    report = inspect_file(path.parent, path.name)
    with file_read_context((report.evidence,)):
        return read_file(path.parent, path.name, start_index=start, end_index=end)


def test_pdf_reads_only_requested_pages_at_configured_dpi(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "report.PDF")
    rendered = []
    original = pdfium.PdfPage.render

    def render(page, **options):
        rendered.append(options["scale"])
        return original(page, **options)

    monkeypatch.setattr(pdfium.PdfPage, "render", render)
    result = inspected_read(path, 2, 6)
    assert isinstance(result, AttachedToolResult)
    assert result["start_index"] == 2 and result["end_index"] == 6
    assert result["total_pages"] == 7 and result["rendered_pages"] == 5
    assert result["dpi"] == 120
    assert rendered == [120 / 72] * 5
    assert [p["page"] for p in result["pages"]] == [2, 3, 4, 5, 6]
    frames = images(result)
    assert [im.size for im in frames] == [(120, 240)] * 5
    assert [im.getpixel((1, 1))[:3] for im in frames] == [
        (255, 0, 0),
        (0, 0, 255),
        (255, 0, 0),
        (0, 0, 255),
        (255, 0, 0),
    ]
    path.unlink()
    for image in frames:
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # Native PDF handles are already closed.


def test_pdf_dpi_and_page_limit_are_configurable(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "report.pdf")
    monkeypatch.setattr(
        reading,
        "FILE_INSPECTION",
        replace(FILE_INSPECTION, pdf_render_dpi=72, max_pdf_render_pages=2),
    )
    result = inspected_read(path, 3, 4)
    assert result["dpi"] == 72
    assert [im.size for im in images(result)] == [(72, 144)] * 2
    with pytest.raises(FileInspectionError, match="at most 2 pages"):
        inspected_read(path, 1, 3)


@pytest.mark.parametrize(
    "start,end", [(0, 1), (1, 0), (True, 1), (1, 1.5), (None, 1), (3, 2), (1, 6)]
)
def test_invalid_pdf_intervals_fail_before_reading_or_rendering(
    tmp_path, monkeypatch, start, end
):
    monkeypatch.setattr(
        reading, "read_snapshot", lambda *a: pytest.fail("early file read")
    )
    with pytest.raises(FileInspectionError, match="start_index|end_index|at most 5"):
        read_file(tmp_path, "report.pdf", start, end)


@pytest.mark.parametrize("setting", ["pdf_render_dpi", "max_pdf_render_pages"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), "120"])
def test_invalid_pdf_render_settings_are_explicit(tmp_path, setting, value):
    path = write_pdf(tmp_path / "report.pdf")
    snapshot = read_snapshot(tmp_path, path.name, FILE_INSPECTION)
    with pytest.raises(FileInspectionError, match="must be a positive integer"):
        read_pdf(snapshot, 1, 1, replace(FILE_INSPECTION, **{setting: value}))


def test_pdf_missing_inspection_does_not_access_file_or_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reading, "read_snapshot", lambda *a: pytest.fail("early file read")
    )
    monkeypatch.setattr(pdfium, "PdfDocument", lambda *a: pytest.fail("early render"))
    result = read_file(tmp_path, "report.pdf", 1, 5)
    assert result["error_kind"] == "file_inspection_required"
    assert "page indices" in result["error"]


def test_pdf_changes_invalidate_inspection_without_rendering(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "report.pdf")
    report = inspect_file(tmp_path, path.name)
    write_pdf(path, count=2)
    monkeypatch.setattr(pdfium, "PdfDocument", lambda *a: pytest.fail("stale render"))
    with file_read_context((report.evidence,)):
        assert (
            read_file(tmp_path, path.name, 1, 1)["error_kind"]
            == "file_inspection_required"
        )


def test_read_uses_the_inspected_pdf_format_without_a_pdf_extension(tmp_path):
    path = write_pdf(tmp_path / "report.dat", count=2)
    report = inspect_file(tmp_path, path.name)
    assert report["format"] == "pdf"

    with file_read_context((report.evidence,)):
        result = read_file(tmp_path, path.name, 2, 2)

    assert result["format"] == "pdf"
    assert result["total_pages"] == 2


def test_pdf_page_range_must_fit_the_actual_document(tmp_path):
    path = write_pdf(tmp_path / "report.pdf", count=2)
    with pytest.raises(FileInspectionError, match=r"outside.*1\.\.2"):
        inspected_read(path, 2, 3)


def test_pdf_byte_and_total_pixel_budgets_are_checked(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "report.pdf", count=2)
    snapshot = read_snapshot(
        tmp_path, path.name, replace(FILE_INSPECTION, max_file_bytes=1)
    )
    with pytest.raises(FileInspectionError, match="byte limit"):
        read_pdf(snapshot, 1, 1, FILE_INSPECTION)
    monkeypatch.setattr(
        reading, "FILE_INSPECTION", replace(FILE_INSPECTION, max_image_pixels=30_000)
    )
    with pytest.raises(FileInspectionError, match="total pixel limit"):
        inspected_read(path, 1, 2)


@pytest.mark.parametrize("kind", ["corrupt", "encrypted"])
def test_unreadable_pdfs_are_explicit(tmp_path, kind):
    path = tmp_path / "report.pdf"
    if kind == "corrupt":
        path.write_bytes(b"not a PDF")
    else:
        write_pdf(path, password="secret")
    with pytest.raises(FileInspectionError, match="PDF could not be read or rendered"):
        read_pdf(
            read_snapshot(tmp_path, path.name, FILE_INSPECTION), 1, 1, FILE_INSPECTION
        )


def test_empty_pdf_is_an_explicit_error(tmp_path, monkeypatch):
    path = write_pdf(tmp_path / "report.pdf", count=0)
    document = MagicMock()
    document.__enter__.return_value = document
    document.__len__.return_value = 0
    monkeypatch.setattr(pdfium, "PdfDocument", lambda *a: document)
    with pytest.raises(FileInspectionError, match="PDF is empty"):
        read_pdf(
            read_snapshot(tmp_path, path.name, FILE_INSPECTION), 1, 1, FILE_INSPECTION
        )
    document.__exit__.assert_called_once()


def test_pdf_render_failure_returns_one_error_without_attachments(
    tmp_path, monkeypatch
):
    path = write_pdf(tmp_path / "report.pdf")

    def fail(*args, **kwargs):
        raise OSError("render failed")

    monkeypatch.setattr(pdfium.PdfPage, "render", fail)
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(call("inspect-file", "inspect", path=path.name)),
            batch(read_call(path=path.name, start_index=1, end_index=2)),
            AssistantMessage("done"),
        ],
    )
    node.run("read PDF")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"] == "invalid_file_read"
    )
    assert "render failed" in result_for(provider.requests[-1], "read")["error"]
    assert replies(provider.requests[-1])[-1].get_attachments() is None
    assert_paired(provider.requests[-1].get_messages())


def test_pdf_automatic_inspection_then_retry_delivers_ordered_cached_pages(tmp_path):
    path = write_pdf(tmp_path / "report.pdf", count=3)
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call(path=path.name, start_index=1, end_index=2)),
            batch(read_call("retry", path=path.name, start_index=2, end_index=3)),
            AssistantMessage("done"),
        ],
    )
    node.run("read PDF")
    assert (
        result_for(provider.requests[2], "read")["error_kind"]
        == "file_inspection_required"
    )
    assert replies(provider.requests[2])[-1].get_name() == "inspect-file"
    assert result_for(provider.requests[-1], "retry")["rendered_pages"] == 2
    response = replies(provider.requests[-1])[-1]
    parts = list(response.get_attachments().iter_content_parts())
    assert parts[0] == "PDF report.pdf, page 2"
    assert parts[2] == "PDF report.pdf, page 3"
    record = next(
        r for r in node._runtime._tool_cache.get_all() if r.tool_call_id == "retry"
    )
    path.unlink()
    saved = node.read_tool_output(record.id)
    assert parse_tool_result(saved.content) == result_for(
        provider.requests[-1], "retry"
    )
    assert [image.getpixel((1, 1))[:3] for image in images(saved)] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert_paired(provider.requests[-1].get_messages())


@pytest.mark.parametrize("inspected", [False, True])
def test_missing_native_renderer_does_not_block_pdf_prerequisite_recovery(
    tmp_path, monkeypatch, inspected
):
    path = write_pdf(tmp_path / "report.pdf", count=2)
    original_import = builtins.__import__

    def import_without_pdfium(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ModuleNotFoundError("PDFium unavailable", name="pypdfium2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pdfium)
    responses = [open_files()]
    if inspected:
        responses.append(batch(call("inspect-file", "inspect", path=path.name)))
    responses.extend(
        [
            batch(read_call(path=path.name, start_index=1, end_index=2)),
            AssistantMessage("done"),
        ]
    )
    node, provider = make_node(tmp_path, responses)
    node.run("read PDF")
    assert result_for(provider.requests[-1], "read")["error_kind"] == (
        "missing_dependency" if inspected else "file_inspection_required"
    )
    if not inspected:
        assert replies(provider.requests[-1])[-1].get_name() == "inspect-file"
        assert parse_tool_result(replies(provider.requests[-1])[-1].get_message())["ok"]
    assert_paired(provider.requests[-1].get_messages())
