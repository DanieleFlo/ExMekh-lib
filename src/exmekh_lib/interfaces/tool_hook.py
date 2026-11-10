from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exmekh_lib.config.interface_text import INTERFACE_TEXT
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.operation_progress import OperationProgress


@dataclass(frozen=True, slots=True)
class ToolContext:
    arguments: dict[str, Any]
    work_root: Path
    history: tuple[Message[object], ...]
    call_id: str = ""
    on_progress: Callable[[OperationProgress], None] | None = None


@dataclass(frozen=True, slots=True)
class ToolArguments:
    arguments: dict[str, Any]
    on_progress: Callable[[OperationProgress], None] | None = None


@dataclass(frozen=True, slots=True)
class ToolResponse:
    result: Any


class BeforeTool(ABC):
    @abstractmethod
    def handle(self, context: ToolContext) -> ToolArguments | ToolResponse | None:
        """Change arguments, answer in place of the tool, or continue unchanged."""


class AfterTool(ABC):
    @abstractmethod
    def handle(self, context: ToolContext, result: Any) -> Any:
        """Return the result that should be exposed to the agent."""


def validate_tool_hooks(before: BeforeTool | None, after: AfterTool | None) -> None:
    if before is not None and not isinstance(before, BeforeTool):
        raise TypeError(INTERFACE_TEXT.before_must_be_a_beforetool_instance)
    if after is not None and not isinstance(after, AfterTool):
        raise TypeError(INTERFACE_TEXT.after_must_be_an_aftertool_instance)
    for hook, argument_count in ((before, 1), (after, 2)):
        if hook is None:
            continue
        method = hook.handle
        if (
            inspect.iscoroutinefunction(method)
            or inspect.isgeneratorfunction(method)
            or inspect.isasyncgenfunction(method)
        ):
            raise TypeError(INTERFACE_TEXT.tool_hook_handle_must_be_synchronous)
        parameters = tuple(inspect.signature(method).parameters.values())
        if len(parameters) != argument_count or any(
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        ):
            raise TypeError(
                INTERFACE_TEXT.tool_hook_handle_must_accept_argument_s.format(
                    argument_count=argument_count
                )
            )
