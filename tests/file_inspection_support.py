"""Deterministic tokenization for standalone file inspection fixtures."""

from dataclasses import replace
from functools import partial

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection.service import (
    inspect_file as inspect_with_tokenizer,
)
from tests.tokenizer_support import StubTokenizer

# The character-counting stub is for parser assertions. Tests that exercise
# presentation use an explicit test budget or a real model tokenizer.
inspect_file = partial(
    inspect_with_tokenizer,
    tokenizer=StubTokenizer(),
    settings=replace(FILE_INSPECTION, max_report_tokens=5000),
)
