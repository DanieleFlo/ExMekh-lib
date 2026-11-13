"""Analysis records and an encapsulated, bounded navigation index."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from exmekh_lib.config.file_inspection import FileInspectionSettings
from exmekh_lib.core.tokenizers.text_budget import TokenTextBudget

from .selection import navigation_words


@dataclass(frozen=True, slots=True)
class Region:
    kind: str
    name: str
    start_line: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind, "name": self.name}
        if self.start_line is not None:
            result.update(start_line=self.start_line, end_line=self.end_line)
        return result


@dataclass(frozen=True, slots=True)
class RegionPage:
    regions: tuple[Region, ...]
    offset: int
    max_items: int
    indexed_items: int
    index_truncated: bool
    labels_truncated: bool
    ranking: Literal["lexical", "document_order"]
    matching_items: int | None = None

    def first_match(self) -> Region | None:
        return self.regions[0] if self.matching_items and self.regions else None

    @property
    def size(self) -> int:
        return len(self.regions)

    @property
    def first_label(self) -> str:
        return self.regions[0].name

    def prefix(self, count: int) -> RegionPage:
        return replace(self, regions=self.regions[:count])

    def with_first_label(self, label: str) -> RegionPage:
        return replace(
            self,
            regions=(replace(self.regions[0], name=label), *self.regions[1:]),
            labels_truncated=True,
        )

    def to_dict(self, *, detailed: bool = False) -> dict[str, object]:
        next_offset = self.offset + len(self.regions)
        total = (
            self.indexed_items if self.matching_items is None else self.matching_items
        )
        has_more = next_offset < total
        selection: dict[str, object] = {
            "indexed_items": self.indexed_items,
            "index_truncated": self.index_truncated,
            "labels_truncated": self.labels_truncated,
            "has_more": has_more,
            "ranking": self.ranking,
        }
        if self.matching_items is not None:
            selection["matching_items"] = self.matching_items
        if has_more and self.regions:
            selection["next_offset"] = next_offset
        if detailed:
            selection.update(offset=self.offset, max_items=self.max_items)
        return {
            "regions": [region.to_dict() for region in self.regions],
            "selection": selection,
        }


@dataclass(frozen=True, slots=True)
class _IndexedRegion:
    region: Region
    name_words: tuple[str, ...]
    search_terms: frozenset[str]


class RegionIndex:
    """Keep navigation bounded; expose selection without mutable list access."""

    def __init__(
        self, settings: FileInspectionSettings, label_budget: TokenTextBudget
    ) -> None:
        self._settings = settings
        self._label_budget = label_budget
        self._regions: list[_IndexedRegion] = []
        self._truncated = False
        self._labels_truncated = False

    @property
    def settings(self) -> FileInspectionSettings:
        return self._settings

    def is_empty(self) -> bool:
        return not self._regions

    def mark_truncated(self) -> None:
        self._truncated = True

    def add(
        self, kind: str, name: str, start: int | None = None, end: int | None = None
    ) -> bool:
        """Add a region in discovery order, retaining at most max_regions."""
        if len(self._regions) >= self._settings.max_regions:
            self.mark_truncated()
            return False
        label = self._label_budget.truncate(name)
        self._labels_truncated |= label != name
        words = navigation_words(name)
        self._regions.append(
            _IndexedRegion(
                Region(kind, label, start, end),
                words,
                frozenset((*words, *navigation_words(kind))),
            )
        )
        return True

    def select(self, *, max_items: int, offset: int, query: str | None) -> RegionPage:
        regions = sorted(
            self._regions,
            key=lambda item: (item.region.start_line or 0, item.region.end_line or 0),
        )
        indexed_items = len(regions)
        matching_items = None
        if query is not None:
            words = navigation_words(query)
            terms = set(words)
            matches = [(len(terms & item.search_terms), item) for item in regions]
            # Prefer full-term matches when available, and exact names within them.
            min_terms = (
                len(terms) if any(score == len(terms) for score, _ in matches) else 1
            )
            matches = [(score, item) for score, item in matches if score >= min_terms]
            matches.sort(
                key=lambda match: (match[0], match[1].name_words == words), reverse=True
            )
            matching_items = len(matches)
            regions = [item for _, item in matches]
        return RegionPage(
            tuple(item.region for item in regions[offset : offset + max_items]),
            offset,
            max_items,
            indexed_items,
            self._truncated,
            self._labels_truncated,
            "lexical" if query is not None else "document_order",
            matching_items,
        )

    def line_boundaries(self) -> list[int]:
        """Return lines after which a read can end at a region boundary."""
        boundaries: set[int] = set()
        for item in self._regions:
            region = item.region
            if region.start_line is not None and region.end_line is not None:
                boundaries.update((region.start_line - 1, region.end_line))
        return sorted(boundaries)


@dataclass(slots=True)
class Analysis:
    format: str
    family: str
    index: RegionIndex
    status: Literal["complete", "invalid_format", "unsupported", "skipped_limit"] = (
        "complete"
    )
    details: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] | None = field(default=None, repr=False)
