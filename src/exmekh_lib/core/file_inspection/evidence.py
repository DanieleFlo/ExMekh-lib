"""Private proof of the exact bytes and report produced by an inspection."""

from dataclasses import dataclass, replace
from pathlib import Path

from exmekh_lib.utils.xml import stringify_tool_result

from .lines import selected_line_text
from .workspace import FileSnapshot


def hash_content(content: bytes) -> bytes:
    """Load the fast hash dependency at execution, like the inspection parsers."""
    from xxhash import xxh3_128_digest

    return xxh3_128_digest(content)


@dataclass(frozen=True, slots=True)
class InspectionEvidence:
    path: Path
    digest: bytes
    report_content: str
    readable: bool
    successful: bool
    source: str = "inspect"
    format: str = ""

    def matches(self, path: Path, digest: bytes) -> bool:
        return self.successful and self.path == path and self.digest == digest


class InspectionReport(dict[str, object]):
    """A normal structured report; proof is never serialized to the model/cache."""

    def __init__(self, report: dict[str, object], snapshot: FileSnapshot) -> None:
        super().__init__(report)
        assert snapshot.content is not None
        self._evidence = InspectionEvidence(
            snapshot.path,
            hash_content(snapshot.content),
            stringify_tool_result(report),
            report.get("family") == "text",
            report.get("analysis_status") in {"complete", "unsupported"},
            format=str(report.get("format", "")),
        )

    @property
    def evidence(self) -> InspectionEvidence:
        return self._evidence


class ReadReport(InspectionReport):
    """Successful text reading, with proof of the snapshot actually delivered."""

    def __init__(self, report: dict[str, object], snapshot: FileSnapshot) -> None:
        super().__init__(report, snapshot)
        self._evidence = replace(
            self._evidence, source="read", readable=True, successful=True
        )

    def selected_text(self) -> str | None:
        """Text delivered by the read, without presentation metadata or line labels."""
        lines = self.get("lines", [])
        if not isinstance(lines, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("text"), str)
            for item in lines
        ):
            return None
        return selected_line_text(item["text"] for item in lines)
