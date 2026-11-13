"""Suggest physical line ranges using the same text budget as read-file."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace

from exmekh_lib.core.tokenizers.text_budget import TokenTextBudget

from .lines import selected_line_text


@dataclass(frozen=True, slots=True)
class ReadRange:
    start_index: int
    end_index: int
    tokens: int

    def to_dict(self, limit: int) -> dict[str, object]:
        result: dict[str, object] = {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "tokens": self.tokens,
        }
        if self.tokens > limit:
            result["over_limit"] = True
        return result


@dataclass(frozen=True, slots=True)
class ReadPlan:
    max_tokens: int
    ranges: tuple[ReadRange, ...]
    next_line: int | None

    def first_range_only(self) -> ReadPlan:
        if len(self.ranges) < 2:
            return self
        return replace(
            self, ranges=self.ranges[:1], next_line=self.ranges[1].start_index
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "ranges": [item.to_dict(self.max_tokens) for item in self.ranges],
        }
        if self.next_line is not None:
            result["next_line"] = self.next_line
        return result


def build_read_plan(
    lines: list[str],
    boundaries: list[int],
    start_line: int,
    end_line: int,
    max_ranges: int,
    budget: TokenTextBudget,
) -> ReadPlan:
    ranges: list[ReadRange] = []
    start = start_line - 1
    if not lines and start == 0:
        ranges.append(ReadRange(0, 0, 0))
    while start < end_line and len(ranges) < max_ranges:
        end, tokens = _fit_range(lines, start, end_line, budget)
        if tokens <= budget.max_tokens and end < end_line:
            boundary_index = bisect_right(boundaries, end)
            if boundary_index:
                boundary = boundaries[boundary_index - 1]
                # Favor nearby structural breaks without producing tiny reads.
                if start < boundary < end:
                    boundary_tokens = budget.count(
                        selected_line_text(lines[start:boundary])
                    )
                    if tokens // 2 <= boundary_tokens <= budget.max_tokens:
                        end, tokens = boundary, boundary_tokens
        ranges.append(ReadRange(start + 1, end, tokens))
        start = end

    return ReadPlan(
        budget.max_tokens, tuple(ranges), start + 1 if start < end_line else None
    )


def _fit_range(
    lines: list[str], start: int, stop: int, budget: TokenTextBudget
) -> tuple[int, int]:
    end = start + 1
    tokens = budget.count(lines[start])
    if tokens > budget.max_tokens:
        return end, tokens

    # Grow the prefix first so paging never tokenizes the entire remaining file
    # merely to suggest a small read. Keep only endpoints tested with this tokenizer.
    while end < stop:
        candidate = min(stop, start + 2 * (end - start))
        count = budget.count(selected_line_text(lines[start:candidate]))
        if count > budget.max_tokens:
            break
        end, tokens = candidate, count
    else:
        return end, tokens

    while candidate - end > 1:
        middle = (end + candidate) // 2
        count = budget.count(selected_line_text(lines[start:middle]))
        if count <= budget.max_tokens:
            end, tokens = middle, count
        else:
            candidate = middle
    return end, tokens
