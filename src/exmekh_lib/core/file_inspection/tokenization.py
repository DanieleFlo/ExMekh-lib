"""Bind model tokenization to file tools in their execution thread."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.tokenizers.text_budget import TextTokenizationError
from exmekh_lib.interfaces.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True)
class InspectionTokenization:
    tokenizer: Tokenizer
    output_settings: ToolOutputSettings | None


_tokenization: ContextVar[InspectionTokenization | None] = ContextVar(
    "inspection_tokenizer", default=None
)


@contextmanager
def inspection_tokenizer_context(
    tokenizer: Tokenizer, output_settings: ToolOutputSettings | None = None
) -> Generator[None]:
    token = _tokenization.set(InspectionTokenization(tokenizer, output_settings))
    try:
        yield
    finally:
        _tokenization.reset(token)


def current_inspection_tokenization() -> InspectionTokenization:
    context = _tokenization.get()
    if context is None:
        raise TextTokenizationError(FILE_INSPECTION_TEXT.tokenizer_required)
    return context


def current_inspection_tokenizer() -> Tokenizer:
    return current_inspection_tokenization().tokenizer
