"""Supported file extensions and budgets for workspace inspection and reading."""

from dataclasses import dataclass

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES


@dataclass(frozen=True, slots=True)
class FileInspectionSettings:
    max_file_bytes: int = 8 * 1024 * 1024
    max_regions: int = 10_000
    default_max_items: int = 12
    max_items: int = 500
    max_label_tokens: int = 200
    read_ranges_per_page: int = 5
    max_report_tokens: int = 1600
    max_query_chars: int = 1_000
    max_structure_nodes: int = 100_000
    max_pdf_pages: int = 2_000
    max_structure_depth: int = 64
    max_archive_entries: int = 10_000
    max_expanded_bytes: int = 64 * 1024 * 1024
    table_rows_per_region: int = 100
    max_image_pixels: int = 40_000_000
    image_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".avif",
        ".ico",
    )
    pdf_render_dpi: int = 120
    max_pdf_render_pages: int = 5
    archive_extensions: tuple[str, ...] = (".tar", ".tgz", ".gz", ".bz2", ".xz")
    zip_extensions: tuple[str, ...] = (".zip", ".docx", ".xlsx", ".xlsm", ".pptx")


FILE_INSPECTION = FileInspectionSettings()


@dataclass(frozen=True, slots=True)
class FileInspectionText:
    """Descriptions and diagnostics for workspace file tools."""

    tokenizer_required: str = (
        f"File inspection requires the model tokenizer. Run {DEFAULT_CAPABILITIES.inspect_file} through an "
        "initialized node, or call inspect_file with an explicit tokenizer."
    )
    no_readable_lines: str = f"This format has no readable physical text lines; use {DEFAULT_CAPABILITIES.inspect_file} metadata."
    inspection_required: str = (
        f"{DEFAULT_CAPABILITIES.inspect_file}(path) and its successful result are missing from the "
        "visible context, or the file has changed. Inspect the current file, "
        "then choose start_index and end_index as {indices} again from that analysis "
        f"and make a new {DEFAULT_CAPABILITIES.read_file} call. This read was not performed."
    )
    image_attachment_label: str = "Image read from {path}"
    pdf_attachment_label: str = "PDF {path}, page {page}"
    metadata_only_warning: str = (
        "File exceeds the configured byte limit; only metadata is available."
    )
    analysis_failed_warning: str = "Format analysis failed: {error_type}."
    invalid_text_warning: str = "Invalid {format_name}: {error_type}."

    path_required: str = "path must contain non-space text inside the workroot."
    path_nul: str = "path must not contain NUL characters."
    path_unicode: str = (
        "path must contain valid Unicode text without surrogate code points."
    )
    root_not_directory: str = "workroot must identify a directory."
    root_unavailable: str = (
        "workroot does not exist or cannot be resolved; restore it before retrying."
    )
    path_outside_root: str = "path must remain inside the workroot."
    path_unavailable: str = "path does not exist or cannot be resolved."
    regular_file_required: str = "path must identify a regular file."
    file_changed: str = "File changed during inspection; retry."
    file_unreadable: str = "File could not be read; check permissions and retry."
    positive_index: str = (
        "{name} must be a positive integer (1-based); choose it from "
        f"{DEFAULT_CAPABILITIES.inspect_file}."
    )
    ordered_indices: str = (
        "start_index must be <= end_index; supply an inclusive interval from "
        f"{DEFAULT_CAPABILITIES.inspect_file}."
    )
    empty_file_range: str = (
        "File is empty; use start_index=0 and end_index=0 to read it."
    )
    line_range_outside_file: str = (
        "start_index/end_index are outside the file (1..{total_lines}); choose both "
        f"inclusive indices from {DEFAULT_CAPABILITIES.inspect_file}."
    )
    positive_page_index: str = "{name} must be a positive PDF page number (1-based)."
    ordered_page_indices: str = (
        "start_index must be <= end_index for the inclusive PDF page range."
    )
    positive_setting: str = "{name} must be a positive integer."
    page_limit: str = (
        "PDF reads allow at most {max_pdf_render_pages} pages per call; choose a "
        "smaller start_index/end_index interval."
    )
    pdf_byte_limit: str = (
        "PDF exceeds the configured file byte limit; use a smaller PDF."
    )
    empty_pdf: str = "PDF is empty; no pages can be read."
    page_range_outside_pdf: str = (
        "start_index/end_index are outside the PDF page range (1..{page_count})."
    )
    pdf_render_failed: str = (
        "PDF could not be read or rendered ({error_type}: {error}); check the file, "
        "encryption and render settings, then retry."
    )
    pdf_pixel_limit: str = (
        "PDF pages exceed the configured total pixel limit; read fewer pages or lower "
        "the configured DPI."
    )
    image_byte_limit: str = (
        "Image exceeds the configured file byte limit; use a smaller image."
    )
    image_pixel_limit: str = (
        "Image dimensions exceed the configured pixel limit; use a smaller image."
    )
    image_decode_failed: str = (
        "Image could not be read or decoded ({error_type}); check the file and its "
        "format, then retry."
    )
    detailed_boolean: str = "detailed must be a boolean."
    max_items_range: str = "max_items must be an integer between 1 and {max_items}."
    offset_nonnegative: str = "offset must be a nonnegative integer."
    read_from_line_positive: str = "read_from_line must be a positive integer."
    read_from_line_outside_file: str = (
        "read_from_line is outside the file ({total_lines} physical lines)."
    )
    query_required: str = (
        "query must contain searchable words and at most {max_query_chars} characters."
    )

    archive_declared_size_warning: str = "Declared expanded size exceeds the configured limit; no members were extracted."
    archive_decompression_warning: str = (
        "Decompressed archive exceeds the configured byte limit; scan skipped."
    )
    archive_partial_warning: str = (
        "Archive scan stopped at its entry or expanded-size limit; totals are partial."
    )
    image_dimensions_warning: str = (
        "Image dimensions exceed the configured pixel limit."
    )
    invalid_xz: str = "Invalid XZ archive data."
    recursive_yaml_warning: str = "Recursive YAML alias omitted from navigation."
    xml_entities_forbidden: str = "XML document types and entities are not supported."
    unsupported_binary_warning: str = (
        "No text analyzer is available for this binary or non-Unicode file."
    )
    parser_depth_warning: str = "Parser nesting limit exceeded."
    recovered_syntax_warning: str = "The parser recovered from syntax errors; some navigation regions may be missing."
    invalid_json_lines_warning: str = "Some JSON Lines records could not be parsed; their physical lines are still readable."
    encrypted_pdf_warning: str = (
        "Encrypted PDF: page count and document properties are unavailable."
    )
    invalid_pdf_tree: str = "Invalid PDF page tree."
    invalid_pdf_count: str = "Invalid PDF page count."
    pdf_traversal_warning: str = (
        "Declared page count exceeds the configured limit; page traversal was skipped."
    )
    physical_line_outside_source: str = "Physical line is outside the source"
    content_region_label: str = "file content"
    complex_key_label: str = "<complex key>"
    table_header_label: str = "table header"
    table_records_label: str = "records {start}-{end}"
    page_indices_label: str = "page indices"
    line_indices_label: str = "line indices"

    @property
    def inspect_description(self) -> str:
        return (
            "Inspect a workroot file: metadata, text line count and bounded navigation ranges "
            "(1-based, inclusive), not full content. PDF/image metadata and archive indexes only; "
            "no OCR/extraction. Text read_plan shows token-sized reads; read_from_line resumes it. "
            "query finds matching regions; offset/max_items paginate "
            f"(default {FILE_INSPECTION.default_max_items}); detailed adds technical limits. Check truncation/warnings."
        )

    @property
    def read_description(self) -> str:
        return (
            "Read explicit inclusive start_index/end_index: text lines or PDF pages from 1. "
            f"Text/PDF requires visible {DEFAULT_CAPABILITIES.inspect_file} of unchanged content; "
            "missing/stale evidence triggers inspection, then a new read is required. "
            f"PDF returns <={FILE_INSPECTION.max_pdf_render_pages} labelled page images. "
            "Images need no inspection (first frame, indices ignored); use 0/0, also for empty text. "
            "No implicit full read or range clamping."
        )

    @property
    def read_guidance(self) -> str:
        return (
            "Supply path, start_index and end_index. Use 0/0 for images or empty text; "
            f"otherwise choose inclusive 1-based line or page indices from {DEFAULT_CAPABILITIES.inspect_file} "
            f"(at most {FILE_INSPECTION.max_pdf_render_pages} pages for PDF)."
        )


FILE_INSPECTION_TEXT = FileInspectionText()
