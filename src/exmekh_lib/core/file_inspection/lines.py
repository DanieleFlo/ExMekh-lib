"""Physical CR/LF lines, independent of a parser's Unicode newline rules."""

import re
from bisect import bisect_right
from collections.abc import Iterable

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT


class SourceLines:
    """Own physical line positions, including the insertion point at EOF."""

    def __init__(self, text: str) -> None:
        self._length = len(text)
        self._starts = [0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", text))]

    @property
    def physical_count(self) -> int:
        """A final newline ends a physical line without creating another one."""
        return len(self._starts) - int(self._starts[-1] == self._length)

    def start_offset(self, line: int) -> int:
        """Return a physical line start, or EOF for physical_count + 1."""
        if not 1 <= line <= self.physical_count + 1:
            raise ValueError(FILE_INSPECTION_TEXT.physical_line_outside_source)
        return self._starts[line - 1] if line <= self.physical_count else self._length

    def is_line_start(self, character_offset: int) -> bool:
        """EOF is an insertion point, not the start of literal source text."""
        return (
            0 <= character_offset < self._length
            and self._starts[bisect_right(self._starts, character_offset) - 1]
            == character_offset
        )

    def line_number(self, character_offset: int) -> int:
        return bisect_right(self._starts, character_offset)


def physical_lines(text: str) -> list[str]:
    """A trailing CR/LF ends a line without creating an extra physical line."""
    if not text:
        return []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines[-1]:
        lines.pop()
    return lines


def selected_line_text(lines: Iterable[str]) -> str:
    """Join a text read's physical lines for output budgeting."""
    return "\n".join(lines)
