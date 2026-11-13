"""Bound displayed text using only the tokenizer's public interface."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from exmekh_lib.config.tokenizer_text import TOKENIZER_TEXT
from exmekh_lib.interfaces.tokenizer import Tokenizer


class TextTokenizationError(RuntimeError):
    """A tokenizer failure, distinct from invalid source content."""


@dataclass(frozen=True, slots=True)
class TokenTextBudget:
    tokenizer: Tokenizer = field(repr=False)
    max_tokens: int

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 0:
            raise ValueError(TOKENIZER_TEXT.invalid_text_budget)

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.max_tokens

    def with_limit(self, max_tokens: int) -> TokenTextBudget:
        return replace(self, max_tokens=max_tokens)

    def truncate(self, text: str, *, suffix: str = "") -> str:
        """Include any suffix in the budget and preserve untruncated text exactly."""
        try:
            if self.fits(text):
                return text
            if not suffix:
                return self._prefix(text, self.max_tokens)
            suffix = self._prefix(suffix, self.max_tokens)
            prefix_tokens = self.max_tokens - self.count(suffix)
            for limit in range(prefix_tokens, -1, -1):
                prefix = self._prefix(text, limit).rstrip()
                candidate = prefix + suffix
                if self.count(candidate) <= self.max_tokens:
                    return candidate
        except TextTokenizationError:
            raise
        except Exception as error:
            raise TextTokenizationError(
                TOKENIZER_TEXT.tokenization_failed.format(error=error)
            ) from error
        raise TextTokenizationError(TOKENIZER_TEXT.invalid_prefix)

    def count(self, text: str) -> int:
        try:
            count = self.tokenizer.count_tokens(text)
        except TextTokenizationError:
            raise
        except Exception as error:
            raise TextTokenizationError(
                TOKENIZER_TEXT.tokenization_failed.format(error=error)
            ) from error
        if type(count) is not int or count < 0:
            raise TextTokenizationError(TOKENIZER_TEXT.invalid_token_count)
        return count

    def _prefix(self, text: str, limit: int) -> str:
        prefix = self.tokenizer.truncate_text(text, limit)
        if (
            not isinstance(prefix, str)
            or not text.startswith(prefix)
            or self.count(prefix) > limit
        ):
            raise TextTokenizationError(TOKENIZER_TEXT.invalid_prefix)
        return prefix
