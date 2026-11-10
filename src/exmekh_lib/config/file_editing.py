"""Shared literal edit contract and bounded filesystem operations."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES


@dataclass(frozen=True, slots=True)
class FileEditingSettings:
    temporary_file_prefix: str = ".exmekh-edit-"
    temporary_file_mode: int = 0o600
    min_new_file_characters: int = 2
    max_creation_document_bytes: int = 1024 * 1024
    max_creation_elements: int = 64
    creation_directory_mode: int = 0o777
    max_snapshot_bytes: int = 1024 * 1024
    max_generated_bytes: int = 1024 * 1024
    protocol_timeout_seconds: float = 300
    max_exchanges: int = 64
    max_response_bytes: int = 16 * 1024 * 1024
    inspection_required: str = (
        "Inspect the current file first. The visible inspection is missing or stale; "
        f"this edit was rejected. Read the current file, then make a new {DEFAULT_CAPABILITIES.edit_file} call."
    )
    read_required: str = (
        f"Read the current file with {DEFAULT_CAPABILITIES.read_file} after its inspection. A complete, "
        "unchanged successful read must remain visible. This edit was rejected."
    )
    changed: str = "File or path changed during generation; inspect and read again."
    selection_error: str = (
        "old_text must occur exactly once globally and start at a physical line boundary, "
        "including indentation. It may be empty only when the existing file is empty."
    )
    selection_empty: str = "old_text is empty, but the existing file is not empty."
    selection_missing: str = "old_text does not occur in the current initial snapshot."
    selection_ambiguous: str = "old_text occurs more than once in the initial snapshot."
    selection_midline: str = (
        "old_text does not start at a physical line boundary, including indentation."
    )
    conflict_error: str = "Edits of the same canonical file must use the initial snapshot without overlapping selections."
    document_description: str = (
        'One string: <exmekh_edit path="file.txt" id="123" />old text'
        '<exmekh_new_text id="123"/>new text. The server supplies the opening ID; '
        "copy it into the separator; omit IDs only for unambiguous framing. "
        "Bodies are exact literal text: preserve whitespace/newlines; no entity or CDATA decoding. "
        "Append edits with fresh IDs; new text ends at the next opening or string end. "
        "Empty new text removes the selection, not the file. Insert by retaining an anchor "
        "in new text. Empty old text requires an existing empty file. No command tags."
    )

    noneditable_inspection: str = (
        "Inspection identifies a format without editable physical text."
    )
    unsupported_text_format: str = (
        "This file format is not supported by the text edit tool."
    )
    existing_file_required: str = f"{DEFAULT_CAPABILITIES.edit_file} requires an existing file; use {DEFAULT_CAPABILITIES.new_file}."
    unicode_text_required: str = (
        "Only supported Unicode text without NUL can be edited."
    )
    decoded_snapshot_limit: str = (
        "Decoded snapshot exceeds the UTF-8 control byte limit."
    )
    authorization_mismatch: str = (
        "Every path and edit index must match its authorization, including after "
        "Before hooks."
    )
    alias_snapshot_mismatch: str = "Aliases must share the same initial snapshot."
    replacement_byte_limit: str = (
        "Result exceeds the configured file byte limit; no edit was applied."
    )
    interactive_exchange_required: str = (
        "An accepted interactive snapshot exchange is required; ordinary tool-call "
        "fallback is forbidden."
    )
    parent_required: str = "The parent directory must already exist."
    snapshot_byte_limit: str = (
        "File exceeds the edit snapshot byte limit; no truncated snapshot is allowed."
    )
    minimum_content: str = "content must contain at least {minimum} Unicode characters."
    content_nul: str = "content must not contain NUL characters."
    content_unicode: str = (
        "content must contain valid Unicode text representable in UTF-8."
    )
    creation_byte_limit: str = "Content exceeds the configured file byte limit."
    path_exists: str = (
        f"The path already exists; {DEFAULT_CAPABILITIES.new_file} cannot overwrite it."
    )
    creation_document_string: str = "The creation document must be a string."
    creation_document_byte_limit: str = "Creation document exceeds the limit of {maximum} UTF-8 bytes; no resources were created."
    creation_document_element_limit: str = "Creation document exceeds the limit of {maximum} creation markers; no resources were created."
    creation_document_limit_hint: str = "Split the document into smaller creation calls within the reported limits; no resources were created."
    creation_marker_required: str = (
        'Expected <new-exmekh-file path="<path>"/> or '
        '<new-exmekh-dir path="<path>"/>; only the path attribute is supported.'
    )
    empty_creation_document: str = (
        "Creation document must contain at least one creation."
    )
    directory_content_ignored: str = (
        "Directory content was ignored; directory markers do not accept content."
    )
    deletion_target: str = f"{DEFAULT_CAPABILITIES.delete_file} requires an existing regular file or directory; top-level symbolic links are not followed."
    deletion_paths_required: str = (
        "paths must be a nonempty list of workroot-relative paths."
    )
    deletion_relative_path: str = "Deletion paths must be relative to the workroot."
    deletion_workroot: str = "The workroot itself cannot be deleted."
    changed_before_deletion: str = "File or path changed before deletion."
    unsupported_platform: str = (
        "This platform lacks descriptor-relative atomic edit operations; use a "
        "supported filesystem/runtime."
    )
    file_not_writable: str = "File is not writable; check permissions before retrying."
    parent_not_writable: str = (
        "Parent directory is not writable/searchable; check permissions before "
        "retrying."
    )
    selection_required: str = "At least one file edit is required."
    document_string: str = "The edit document must be an exact string"
    document_unicode: str = "The edit document contains invalid Unicode text"
    document_byte_limit: str = "Edit document exceeds the generated UTF-8 byte limit"
    opening_marker_required: str = "Expected an exmekh_edit opening marker"
    duplicate_edit_id: str = "Duplicate edit ID"
    separator_required: str = "Missing matching exmekh_new_text separator"
    separator_ambiguous: str = (
        "Ambiguous exmekh_new_text separator; include a distinct ID"
    )
    document_element_limit: str = "Edit document exceeds the element limit"
    empty_document: str = "Edit document must contain at least one edit"
    runtime_handler_required: str = (
        "Interactive behavior has no registered runtime handler"
    )
    authorization_sequence: str = "Missing or duplicate edit authorization"
    cancelled_before_dispatch: str = (
        "Interactive call invalidated by steering before dispatch"
    )
    cancelled_before_write: str = (
        "Interactive edit invalidated by cancellation or scope change before writing."
    )
    definition_required: str = (
        "Literal edit behavior requires exactly one required document string"
    )

    @property
    def description(self) -> str:
        return (
            "Replace literal selections in existing text files. Requires visible inspect then read "
            "of unchanged content. Select unique text from line start, including indentation; "
            "initial-snapshot selections cannot overlap. Rejected resources prevent writes. "
            f'<if-tool name="{DEFAULT_CAPABILITIES.new_file}" scope="selected">'
            f"Use {DEFAULT_CAPABILITIES.new_file} to create a file.</if-tool> "
            f'<if-tool name="{DEFAULT_CAPABILITIES.delete_file}" scope="selected">'
            f"Use {DEFAULT_CAPABILITIES.delete_file} to remove a file.</if-tool> "
        )

    delete_file_description: str = (
        "Delete a nonempty list of workroot-relative paths; directories are recursive. "
        "Reject the workroot and directly named symlinks; never follow nested symlinks. "
        "No inspect/read required. Ordered per-path results; failures keep earlier effects."
    )

    argument_guidance: str = (
        "Supply one literal two-marker edits string. Inspect and read existing files "
        "first."
    )

    @property
    def new_file_description(self) -> str:
        return (
            "Create absent files/directories in order; parents must exist or be created earlier. "
            "No inspect/read or overwrite. Per-entry errors preserve successes; retry only failures."
        )

    @property
    def new_document_description(self) -> str:
        return (
            '<new-exmekh-dir path="folder"/><new-exmekh-file path="folder/file.txt"/>content. '
            "Self-closing markers have only a double-quoted path, entity-decoded. "
            "Each body ends at the next reserved new-exmekh-file/new-exmekh-dir opening prefix "
            "or string end; no closing tags or escaping those prefixes. "
            f"File bodies need >={self.min_new_file_characters} Unicode characters, "
            "preserved literally without trimming, entity decoding or added newlines. "
            "Directory bodies are ignored with warnings."
        )

    @classmethod
    def from_environment(cls) -> FileEditingSettings:
        value = os.getenv("EXMEKH_INTERACTIVE_TIMEOUT_SECONDS")
        if value is None:
            return cls()
        error_text = "EXMEKH_INTERACTIVE_TIMEOUT_SECONDS must be finite and positive"
        try:
            timeout = float(value)
        except ValueError as error:
            raise ValueError(error_text) from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(error_text)
        return cls(protocol_timeout_seconds=timeout)


FILE_EDITING = FileEditingSettings.from_environment()


@dataclass(frozen=True, slots=True)
class FileEditFormat:
    """Fixed framing and behavior negotiated with the interactive provider."""

    behavior: str = "edit_file.v3"
    uninspected_error_code: str = "uninspected"
    unread_error_code: str = "unread"
    opening_pattern: str = r'<exmekh_edit path="([^"<>]+)"(?: id="([0-9]{3})")?\s*/>'
    separator_pattern: str = r'<exmekh_new_text(?: id="([0-9]{3})")?\s*/>'


FILE_EDIT_FORMAT = FileEditFormat()


@dataclass(frozen=True, slots=True)
class FileCreationFormat:
    """Reserved creation boundaries; only path attributes have XML semantics."""

    marker_pattern: str = r"<new-exmekh-(?:file|dir)\b"
    opening_pattern: str = r'<new-exmekh-(file|dir)\s+path="([^"<>]*)"\s*/>'


FILE_CREATION_FORMAT = FileCreationFormat()
