"""Versioned operation feedback, independent of model and compute backend."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exmekh_lib.config.native_suite import NATIVE_SUITE


class OperationProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    type: Literal[
        "operation.started",
        "operation.progress",
        "operation.heartbeat",
        "operation.completed",
        "operation.failed",
        "operation.cancelled",
    ]
    operation_id: str
    request_id: str
    seq: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    state: Literal["running", "completed", "failed", "cancelled"]
    phase: str
    item_index: int = Field(default=0, ge=0)
    item_count: int = Field(default=1, ge=1)
    active_items: int = Field(default=1, ge=1)
    completed_units: int = Field(default=0, ge=0)
    total_units: int | None = Field(default=None, ge=1)
    unit: str | None = None
    overall_fraction: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    overall_kind: Literal["work", "estimated", "indeterminate"] = "indeterminate"

    @model_validator(mode="after")
    def _validate_snapshot(self) -> OperationProgress:
        expected = self.type.removeprefix("operation.")
        if expected in {"started", "progress", "heartbeat"}:
            expected = "running"
        if (
            self.state != expected
            or self.item_index + self.active_items > self.item_count
        ):
            raise ValueError(NATIVE_SUITE.invalid_response)
        return self

    def follows(self, previous: OperationProgress | None) -> bool:
        """Accept a later snapshot only from the same running operation."""
        return previous is None or (
            previous.state == "running"
            and self.operation_id == previous.operation_id
            and self.request_id == previous.request_id
            and self.seq > previous.seq
        )

    def is_active_item(self, index: int) -> bool:
        return self.item_index <= index < self.item_index + self.active_items

    def item_fraction(self, index: int) -> float | None:
        """Resolve batch progress to the work for one item in the snapshot."""
        if not 0 <= index < self.item_count:
            return None
        if self.state == "completed" or index < self.item_index:
            return 1.0
        if not self.is_active_item(index) or self.overall_fraction is None:
            return None
        completed = self.overall_fraction * self.item_count - self.item_index
        return min(1.0, max(0.0, completed / self.active_items))


_observer: ContextVar[Callable[[OperationProgress], None] | None] = ContextVar(
    "operation_progress_observer", default=None
)


def operation_progress_observer() -> Callable[[OperationProgress], None] | None:
    return _observer.get()


@contextmanager
def operation_progress_context(
    callback: Callable[[OperationProgress], None] | None,
) -> Iterator[None]:
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)
