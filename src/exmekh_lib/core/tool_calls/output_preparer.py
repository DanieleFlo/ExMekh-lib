"""Materialize oversized text before publication or persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.tool_errors import ERROR_HINTS
from exmekh_lib.config.tool_output import TOOL_OUTPUT_TEXT
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.file_inspection.lines import SourceLines
from exmekh_lib.core.models.external_tool_output import (
    ExternalToolOutput,
    OutputTextFile,
)
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tokenizers.text_budget import TokenTextBudget
from exmekh_lib.interfaces.tokenizer import Tokenizer
from exmekh_lib.utils.tool_errors import error_diagnostic
from exmekh_lib.utils.xml import stringify_tool_result

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedToolOutput:
    content: str
    external_output: ExternalToolOutput | None = None
    failure: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return self.external_output is not None or self.failure is not None

    def as_result(self, original: PreformattedToolResult) -> PreformattedToolResult:
        return PreformattedToolResult(
            self.content,
            attachments=original.attachments,
            external_output=self.external_output,
            delivery_failed=original.delivery_failed or self.failure is not None,
        )


class ToolOutputPreparer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        files: OutputFileWorkspace,
    ) -> None:
        self._tokenizer = tokenizer
        self.files = files
        self.settings = files.settings
        self._budget = TokenTextBudget(tokenizer, self.settings.max_tokens)

    def prepare(
        self,
        output: ToolOutputPresentation,
    ) -> PreparedToolOutput:
        return self._prepare(output, self.settings.max_tokens)

    def externalize(self, output: ToolOutputPresentation) -> PreparedToolOutput:
        """Export a nonempty body independently of the ordinary inline threshold."""
        return self._prepare(output, 0)

    def shorten(self, output: ToolOutputPresentation) -> PreparedToolOutput:
        """Use a file reference only if it reduces the displayed token cost."""
        prepared = self.externalize(output)
        if prepared.failure is not None or self._budget.count(
            prepared.content
        ) < self._budget.count(output.result.content):
            return prepared
        return PreparedToolOutput(output.result.content, output.result.external_output)

    def _prepare(
        self, output: ToolOutputPresentation, inline_limit: int
    ) -> PreparedToolOutput:
        if output.result.external_output is not None:
            # References may list many files; never recursively export their manifest.
            return PreparedToolOutput(
                output.result.content, output.result.external_output
            )
        text = output.body
        try:
            count = self._budget.count(
                text if output.budget_text is None else output.budget_text
            )
            if count <= inline_limit:
                return PreparedToolOutput(output.result.content)
            reference = self._export(text, output.identity.id)
            return PreparedToolOutput(
                output.replace_body(reference.render()),
                reference,
            )
        except Exception as error:
            logger.exception(TOOL_OUTPUT_TEXT.export_failed_log)
            diagnostic = error_diagnostic(error)
            failure = {
                "ok": False,
                "error_kind": "tool_output_export_failed",
                "error": TOOL_OUTPUT_TEXT.failed.format(error=error),
                **diagnostic,
                "hint": ERROR_HINTS["tool_output_export_failed"]
                + " "
                + str(diagnostic["hint"]),
            }
            rendered = output.replace_body(stringify_tool_result(failure))
            rendered = XmlHeader.add_field_to_content(
                rendered, "response-type", "failed"
            )
            return PreparedToolOutput(rendered, failure=failure)

    def _export(self, text: str, output_id: str) -> ExternalToolOutput:
        parts = self._parts(text)
        counts = tuple(self._budget.count(part) for part in parts)
        digest = sha256()
        for component in (output_id, str(self.settings.max_tokens), text):
            digest.update(
                component.encode(
                    CACHE_FORMAT.output_text_encoding, CACHE_FORMAT.output_text_errors
                )
            )
            digest.update(CACHE_FORMAT.output_digest_separator)
        paths = self.files.write(digest.hexdigest(), parts)
        files = tuple(
            OutputTextFile(
                path,
                tokens,
                SourceLines(part).physical_count,
            )
            for path, tokens, part in zip(paths, counts, parts, strict=True)
        )
        return ExternalToolOutput(
            self._budget.count(text),
            SourceLines(text).physical_count,
            self.settings.max_tokens,
            files,
        )

    def _parts(self, text: str) -> tuple[str, ...]:
        if SourceLines(text).physical_count != 1:
            return (text,)
        parts = self._tokenizer.split_text(text, self.settings.max_tokens)
        if (
            not parts
            or "".join(parts) != text
            or any(not part or not self._budget.fits(part) for part in parts)
        ):
            raise ValueError(TOOL_OUTPUT_TEXT.invalid_prefix)
        return parts
