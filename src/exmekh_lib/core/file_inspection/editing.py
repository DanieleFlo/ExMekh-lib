"""Authorize snapshots and revalidate every edit after execution hooks."""

from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from exmekh_lib.config.file_editing import FILE_EDIT_FORMAT, FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.config.tool_errors import ERROR_HINTS, PARTIAL_EDIT_MESSAGE
from exmekh_lib.core.models.edit_document import EditDocument, EditElement
from exmekh_lib.utils.tool_errors import error_diagnostic

from .decoding import decode_text
from .edit_filesystem import (
    FileWriteTarget,
    edit_snapshot,
    encode_replacement,
    resolve_edit_target,
    resource_lock,
)
from .edit_selection import select_replacements
from .errors import FileInspectionError
from .evidence import InspectionEvidence, hash_content
from .lines import physical_lines
from .workspace import FileSnapshot


@dataclass(frozen=True, slots=True)
class EditApproval:
    resource: str
    target: Path
    snapshot: FileSnapshot
    text: str
    encoding: str

    def authorize_alias(self, work_root: Path, resource: str) -> Self:
        """Reuse the initial snapshot only while the alias resolves to its target."""
        self.validate_target(work_root, resource)
        return replace(self, resource=resource)

    def validate_target(self, work_root: Path, resource: str) -> None:
        FileWriteTarget(work_root, resource, self.target, self.snapshot).validate()

    def shares_snapshot(self, other: Self) -> bool:
        return (
            self.snapshot.matches(other.snapshot)
            and self.text == other.text
            and self.encoding == other.encoding
        )

    def prepare_replacement(self, elements: tuple[EditElement, ...]) -> bytes:
        replacement = select_replacements(self.text, elements)
        assert self.snapshot.content is not None
        encoded = encode_replacement(replacement, self.snapshot.content, self.encoding)
        if len(encoded) > FILE_EDITING.max_snapshot_bytes:
            raise FileInspectionError(
                FILE_EDITING.replacement_byte_limit, code="file_limit_exceeded"
            )
        return encoded

    def commit_replacement(
        self, work_root: Path, content: bytes, evidence: tuple[InspectionEvidence, ...]
    ) -> None:
        self.require_evidence(evidence)
        FileWriteTarget(work_root, self.resource, self.target, self.snapshot).commit(
            content
        )

    def reply(self) -> dict[str, object]:
        return {
            "status": "provide",
            "text": self.text,
            "encoding": self.encoding,
            "line_count": len(physical_lines(self.text)),
        }

    def require_evidence(self, evidence: tuple[InspectionEvidence, ...]) -> None:
        """Existing snapshots require current, visible inspect/read authorization."""
        assert self.snapshot.content is not None
        require_edit_evidence(self.target, self.snapshot.content, evidence)


class EditPrerequisiteError(FileInspectionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message, code=code, hint=message)


def require_edit_evidence(
    target: Path, content: bytes, evidence: tuple[InspectionEvidence, ...]
) -> None:
    digest = hash_content(content)
    inspected = False
    for item in evidence:
        if not item.matches(target, digest):
            continue
        if item.source == "inspect" and not item.readable:
            raise FileInspectionError(
                FILE_EDITING.noneditable_inspection, code="unsupported_file_format"
            )
        if not item.readable:
            continue
        if item.source == "inspect":
            inspected = True
        elif item.source == "read" and inspected:
            return
    if not inspected:
        raise EditPrerequisiteError(
            FILE_EDIT_FORMAT.uninspected_error_code, FILE_EDITING.inspection_required
        )
    raise EditPrerequisiteError(
        FILE_EDIT_FORMAT.unread_error_code, FILE_EDITING.read_required
    )


def approve_edit(
    work_root: Path, resource: str, evidence: tuple[InspectionEvidence, ...]
) -> EditApproval:
    target = resolve_edit_target(work_root, resource)
    if target.suffix.casefold() in {
        *FILE_INSPECTION.image_extensions,
        *FILE_INSPECTION.archive_extensions,
        *FILE_INSPECTION.zip_extensions,
        ".pdf",
    }:
        raise FileInspectionError(
            FILE_EDITING.unsupported_text_format, code="unsupported_file_format"
        )
    with resource_lock(target):
        try:
            target.lstat()
        except FileNotFoundError as error:
            raise FileInspectionError(
                FILE_EDITING.existing_file_required, code="file_not_found"
            ) from error
        snapshot = edit_snapshot(work_root, resource)
        assert snapshot.content is not None
        require_edit_evidence(target, snapshot.content, evidence)
        decoded = decode_text(snapshot.content, target.suffix.casefold())
        if decoded is None or "\x00" in decoded[0]:
            raise FileInspectionError(
                FILE_EDITING.unicode_text_required, code="unsupported_file_format"
            )
        text, encoding = decoded
        if len(text.encode("utf-8")) > FILE_EDITING.max_snapshot_bytes:
            raise FileInspectionError(
                FILE_EDITING.decoded_snapshot_limit, code="file_limit_exceeded"
            )
        return EditApproval(resource, target, snapshot, text, encoding)


type EditEvidence = (
    tuple[InspectionEvidence, ...] | Callable[[], tuple[InspectionEvidence, ...]]
)

_approval: ContextVar[tuple[tuple[EditApproval, ...], EditEvidence] | None] = (
    ContextVar("edit_approvals", default=None)
)


@contextmanager
def file_edit_context(
    approvals: tuple[EditApproval, ...], evidence: EditEvidence
) -> Generator[None]:
    token = _approval.set((approvals, evidence))
    try:
        yield
    finally:
        _approval.reset(token)


@dataclass(frozen=True, slots=True)
class _PreparedEdit:
    approval: EditApproval
    resources: tuple[str, ...]
    content: bytes

    @property
    def resource(self) -> str:
        return self.approval.resource

    def validate(self, work_root: Path) -> None:
        """Preflight every alias before any canonical file is changed."""
        for resource in self.resources:
            self.approval.validate_target(work_root, resource)

    def commit(self, work_root: Path, evidence: tuple[InspectionEvidence, ...]) -> None:
        self.approval.commit_replacement(work_root, self.content, evidence)

    def result(self, work_root: Path) -> dict[str, object]:
        return {
            "path": self.approval.target.relative_to(work_root.resolve()).as_posix(),
            "operation": "edit",
            "size_bytes": len(self.content),
        }


def _prepare_edits(
    document: EditDocument,
    approvals: tuple[EditApproval, ...],
    evidence: tuple[InspectionEvidence, ...],
) -> tuple[_PreparedEdit, ...]:
    if len(document.elements) != len(approvals) or any(
        element.path != approval.resource
        for element, approval in zip(document.elements, approvals, strict=True)
    ):
        raise FileInspectionError(
            FILE_EDITING.authorization_mismatch, code="edit_authorization_mismatch"
        )
    groups: dict[Path, list[tuple[EditElement, EditApproval]]] = {}
    for element, approval in zip(document.elements, approvals, strict=True):
        groups.setdefault(approval.target, []).append((element, approval))
    prepared: list[_PreparedEdit] = []
    for group in groups.values():
        approval = group[0][1]
        if any(not approval.shares_snapshot(other) for _, other in group):
            raise FileInspectionError(
                FILE_EDITING.alias_snapshot_mismatch, code="file_changed"
            )
        approval.require_evidence(evidence)
        encoded = approval.prepare_replacement(tuple(element for element, _ in group))
        prepared.append(
            _PreparedEdit(
                approval, tuple(element.path for element, _ in group), encoded
            )
        )
    return tuple(prepared)


def edit_files(work_root: Path, edits: str) -> dict[str, object]:
    """Validate the entire post-hook document, then commit each canonical file once."""
    context = _approval.get()
    if context is None or not context[0]:
        raise FileInspectionError(
            FILE_EDITING.interactive_exchange_required,
            code="interactive_exchange_required",
        )
    approvals, evidence_source = context
    document = EditDocument.parse(
        edits, snapshots=tuple(item.text for item in approvals)
    )
    committed: list[dict[str, object]] = []
    with ExitStack() as locks:
        for target in sorted({approval.target for approval in approvals}):
            locks.enter_context(resource_lock(target))
        evidence = evidence_source() if callable(evidence_source) else evidence_source
        prepared = _prepare_edits(document, approvals, evidence)
        # Validate all resources before the first write, then recheck each commit.
        for item in prepared:
            item.validate(work_root)
        for item in prepared:
            try:
                # Steering can arrive while waiting on locks or between commits.
                current_evidence = (
                    evidence_source() if callable(evidence_source) else evidence_source
                )
                item.commit(work_root, current_evidence)
            except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
                if not committed:
                    raise
                return {
                    "ok": False,
                    "error_kind": "partial_file_edit",
                    "error": PARTIAL_EDIT_MESSAGE.format(error=error),
                    **error_diagnostic(error),
                    "hint": ERROR_HINTS["partial_file_edit"],
                    "failed_path": item.resource,
                    "files": committed,
                }
            committed.append(item.result(work_root))
    return {"ok": True, "files": committed}
