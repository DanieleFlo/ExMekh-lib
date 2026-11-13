"""User-configurable budgets and storage for large tool output text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ToolOutputSettings:
    max_tokens: int = 5000
    directory: Path = Path(".exmekh_lib/tool_outputs")

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.directory, Path):
            raise TypeError("directory must be a Path")
        if self.directory.is_absolute() or ".." in self.directory.parts:
            raise ValueError("directory must be a relative path inside the workroot")
        if not self.directory.parts:
            raise ValueError("directory must not be the workroot itself")

    @classmethod
    def from_environment(cls) -> ToolOutputSettings:
        default = cls()
        return cls(
            max_tokens=int(
                os.getenv("EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS", str(default.max_tokens))
            ),
            directory=Path(
                os.getenv("EXMEKH_LIB_TOOL_OUTPUT_DIR", str(default.directory))
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolOutputText:
    export_failed_log: str = "Tool output export failed"
    exported: str = (
        "The output exceeded {limit} tokens and was written to temporary text files. "
        "Inspect and read the listed files in order with the file tools. "
        "read_tool_output returns this reference, not the original text."
    )
    failed: str = "Large output could not be exported: {error}. The original text was not delivered."
    access_denied: str = (
        "This temporary tool output is not available to this node/session."
    )
    invalid_prefix: str = "Tokenizer could not produce a non-empty, lossless prefix within the token limit."


TOOL_OUTPUT_TEXT = ToolOutputText()
