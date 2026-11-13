"""Build and size structured inspection reports at the presentation boundary."""

from dataclasses import asdict

from exmekh_lib.core.tokenizers.text_budget import TokenTextBudget
from exmekh_lib.utils.xml import stringify_tool_result

from .models import Analysis, RegionPage
from .read_plan import ReadPlan
from .workspace import FileSnapshot


class InspectionPresentation:
    def __init__(
        self,
        snapshot: FileSnapshot,
        analysis: Analysis,
        detailed: bool,
        budget: TokenTextBudget,
    ) -> None:
        self._budget = budget
        self._detailed = detailed
        details = analysis.details.copy()
        base: dict[str, object] = {
            "ok": True,
            "path": snapshot.relative_path,
            "format": analysis.format,
            "family": analysis.family,
            "size_bytes": snapshot.stat.st_size,
            "analysis_status": analysis.status,
            "analysis": details,
            "warnings": analysis.warnings,
        }
        if analysis.lines is not None:
            base["total_lines"] = len(analysis.lines)
        if detailed:
            base["metadata"] = snapshot.metadata()
            base["limits"] = asdict(analysis.index.settings)
        else:
            for field in (
                "total_lines",
                "characters",
                "non_empty_lines",
                "maximum_line_length",
                "newline_style",
            ):
                details.pop(field, None)
        self._base = base

    def fit(self, page: RegionPage, plan: ReadPlan | None) -> dict[str, object]:
        report = self._report(page, plan)
        if self._fits(report):
            return report

        minimum = min(1, page.size)
        first = self._report(page.prefix(minimum), plan)
        if not self._fits(first) and plan is not None:
            plan = plan.first_range_only()
            first = self._report(page.prefix(minimum), plan)

        compact = not self._detailed and not self._fits(first)
        if compact:
            first = self._report(page.prefix(minimum), plan, compact=True)

        if minimum and not self._fits(first):
            page = self._shorten_first_label(page, plan, compact=compact)
            first = self._report(page.prefix(minimum), plan, compact=compact)

        lower, upper = minimum, page.size + 1
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            candidate = self._report(page.prefix(middle), plan, compact=compact)
            if self._fits(candidate):
                lower, first = middle, candidate
            else:
                upper = middle
        return first

    def _report(
        self, page: RegionPage, plan: ReadPlan | None, *, compact: bool = False
    ) -> dict[str, object]:
        report = {**self._base, **page.to_dict(detailed=self._detailed)}
        if plan is not None:
            report["read_plan"] = plan.to_dict()
        if compact:
            _keep_report_essentials(report)
        return report

    def _fits(self, report: dict[str, object]) -> bool:
        return self._budget.fits(stringify_tool_result(report))

    def _shorten_first_label(
        self, page: RegionPage, plan: ReadPlan | None, *, compact: bool
    ) -> RegionPage:
        name = page.first_label
        label_tokens = self._budget.count(name) // 2
        while label_tokens:
            page = page.with_first_label(
                self._budget.with_limit(label_tokens).truncate(name)
            )
            if self._fits(self._report(page.prefix(1), plan, compact=compact)):
                break
            label_tokens //= 2
        return page


def _keep_report_essentials(report: dict[str, object]) -> None:
    """Preserve read coordinates and paging when the configured limit is tight."""
    report.pop("size_bytes", None)
    if not report.get("warnings"):
        report.pop("warnings", None)
    if not report.get("regions"):
        report.pop("regions", None)
    selection = report.get("selection")
    if isinstance(selection, dict):
        selection.pop("ranking", None)
        selection.pop("indexed_items", None)
        for key in ("has_more", "index_truncated", "labels_truncated"):
            if selection.get(key) is False:
                selection.pop(key)
        if not selection:
            report.pop("selection")
    if report.get("read_plan") is not None:
        details = report.get("analysis")
        if isinstance(details, dict):
            details = details.copy()
            report["analysis"] = details
            if details.get("encoding") == "utf-8":
                details.pop("encoding")
            details.pop("language", None)
            if details.get("syntax_valid") is True:
                details.pop("syntax_valid")
            if details.get("syntax_recovered") is False:
                details.pop("syntax_recovered")
            details.pop("root_kind", None)
    if not report.get("analysis"):
        report.pop("analysis", None)
