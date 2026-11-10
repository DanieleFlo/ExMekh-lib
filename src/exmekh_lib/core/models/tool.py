from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_type_hints

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    validate_interactive_tool,
)
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.operation_progress import operation_progress_context
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
    validate_tool_hooks,
)
from exmekh_lib.utils.tool_errors import (
    ToolArgumentError,
    ToolInputError,
    ToolTypeError,
)


@dataclass(frozen=True, slots=True)
class PreformattedToolResult:
    """Preformatted text with an optional multimodal companion for the provider."""

    content: str
    attachments: UserMessage | None = None
    external_output: ExternalToolOutput | None = None
    delivery_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(MODEL_TEXT.preformatted_tool_content_must_be_a_string)
        if self.external_output is not None and not isinstance(
            self.external_output, ExternalToolOutput
        ):
            raise TypeError(
                MODEL_TEXT.external_output_must_be_externaltooloutput_or_none
            )
        if not isinstance(self.delivery_failed, bool):
            raise TypeError(MODEL_TEXT.delivery_failed_must_be_a_boolean)
        if self.attachments is not None and not isinstance(
            self.attachments, UserMessage
        ):
            raise TypeError(
                MODEL_TEXT.preformatted_tool_attachments_must_be_a_usermessage_or_none
            )


class AttachedToolResult(dict[str, object]):
    """Structured fields with a multimodal companion outside the serialized result."""

    def __init__(self, result: dict[str, object], attachments: UserMessage) -> None:
        if not isinstance(attachments, UserMessage):
            raise TypeError(MODEL_TEXT.attached_tool_results_require_a_usermessage)
        super().__init__(result)
        self._attachments = attachments

    @property
    def attachments(self) -> UserMessage:
        return self._attachments


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    mode: Literal["wait", "background"] = "wait"
    wait_seconds: float = RUNTIME.default_tool_wait_seconds

    def __post_init__(self) -> None:
        if self.mode not in ("wait", "background"):
            raise ToolInputError(
                MODEL_TEXT.execution_mode_must_be_wait_or_background,
                code="invalid_execution_options",
            )
        self.validate_wait_seconds(self.wait_seconds)

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> ExecutionOptions:
        mode = arguments.get("execution_mode", "wait")
        wait_seconds = arguments.get("wait_seconds", RUNTIME.default_tool_wait_seconds)
        mode = "wait" if mode is None else mode
        wait_seconds = (
            RUNTIME.default_tool_wait_seconds if wait_seconds is None else wait_seconds
        )
        return cls(mode=mode, wait_seconds=wait_seconds)

    @staticmethod
    def validate_wait_seconds(wait_seconds: float) -> float:
        if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)):
            raise ToolTypeError(
                MODEL_TEXT.wait_seconds_must_be_a_number,
                code="invalid_execution_options",
            )
        if not 0 < wait_seconds <= RUNTIME.max_tool_wait_seconds:
            raise ToolInputError(
                MODEL_TEXT.wait_seconds_must_be_between_and.format(
                    runtime_max_tool_wait_seconds=RUNTIME.max_tool_wait_seconds
                ),
                code="invalid_execution_options",
            )
        return float(wait_seconds)


@dataclass(slots=True)
class ToolExecutionOperation:
    """One tool invocation; specialized tools may supply cancellation and output reads."""

    run: Callable[[], Any]
    cancel: Callable[[], bool] | None = None
    read_partial: Callable[[int], tuple[str, int]] | None = None
    _arguments: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def record_arguments(self, arguments: dict[str, Any]) -> None:
        """Freeze selected work inputs before the callable can mutate them."""
        self._arguments = deepcopy(arguments)

    def get_executed_arguments(self) -> dict[str, Any] | None:
        """Snapshot the work arguments selected by Before, once it has run."""
        return deepcopy(self._arguments)

    def add_run_context(self, context: AbstractContextManager[object]) -> None:
        """Wrap execution without losing cancellation/progress updates on this handle."""
        run = self.run

        def run_in_context() -> Any:
            with context:
                return run()

        self.run = run_in_context


class Tool:
    def __init__(
        self,
        tool: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        cache_result: bool = True,
        background_enabled: bool = True,
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
        argument_guidance: str = "",
        interactive: InteractiveDefinition | None = None,
    ) -> None:
        self.tool = tool
        self._tool_type: Literal["function"] = "function"
        self._check_callable_tool()
        self.name = self._resolve_tool_name(tool, name)
        self.description = (
            (inspect.getdoc(tool) or "") if description is None else description
        )
        self._description_template = self.description
        self._result_validator: Callable[[Any], Any] | None = None
        if not isinstance(cache_result, bool):
            raise TypeError(MODEL_TEXT.cache_result_must_be_a_bool)
        self.cache_result = cache_result
        if not isinstance(background_enabled, bool):
            raise TypeError(MODEL_TEXT.background_enabled_must_be_a_bool)
        self.background_enabled = background_enabled
        if (
            background_enabled
            and RUNTIME.execution_argument_names
            & self.get_work_signature().parameters.keys()
        ):
            raise ValueError(
                MODEL_TEXT.tool_parameters_cannot_use_reserved_execution_names
            )
        validate_tool_hooks(before, after)
        self.before = before
        self.after = after
        if not isinstance(argument_guidance, str):
            raise TypeError(MODEL_TEXT.argument_guidance_must_be_a_string)
        self.argument_guidance = argument_guidance
        validate_interactive_tool(interactive, background_enabled=background_enabled)
        self.interactive = interactive
        self._parameter_descriptions_visible = True
        self._check_inputs()

    def _resolve_tool_name(self, tool: Callable[..., Any], name: str | None) -> str:
        resolved_name = getattr(tool, "__name__", "") if name is None else name
        if not self.is_valid_tool_name(resolved_name):
            raise ValueError(
                MODEL_TEXT.tool_name_must_contain_to_letters_digits_underscores_or
            )
        return resolved_name

    def _check_callable_tool(self) -> None:
        if not callable(self.tool):
            raise TypeError(MODEL_TEXT.tool_must_be_callable)

    def _check_inputs(self):
        if not isinstance(self.description, str):
            raise TypeError(MODEL_TEXT.tool_description_must_be_a_string)

    def is_valid_tool_name(self, name: str) -> bool:
        _TOOL_NAME_PATTERN = re.compile(VALIDATION_PATTERNS.tool_name)
        return bool(_TOOL_NAME_PATTERN.fullmatch(name))

    def execute(self, **arguments: Any) -> Any:
        return self.tool(**arguments)

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate the final arguments before invoking the tool."""
        signature = self.get_work_signature()
        try:
            signature.bind(**arguments)
        except TypeError as error:
            raise ToolArgumentError.from_binding_error(
                error, signature, arguments
            ) from error

    def is_execution_argument(self, name: str) -> bool:
        return self.background_enabled and name in RUNTIME.execution_argument_names

    def normalize_call_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Null execution options use defaults; null business arguments remain intact."""
        return {
            name: value
            for name, value in arguments.items()
            if value is not None or not self.is_execution_argument(name)
        }

    def get_work_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in arguments.items()
            if not self.is_execution_argument(name)
        }

    def execute_with_hooks(
        self,
        arguments: dict[str, Any],
        *,
        work_root: Path,
        history: tuple[Message[object], ...],
        call_id: str = "",
    ) -> Any:
        return self.create_execution_with_hooks(
            arguments, call_id=call_id, work_root=work_root, history=history
        ).run()

    def _finalize_result(self, context: ToolContext, result: Any) -> Any:
        result = (
            self.after.handle(context, result) if self.after is not None else result
        )
        return (
            self._result_validator(result)
            if self._result_validator is not None
            else result
        )

    def _prepare_context(
        self,
        arguments: dict[str, Any],
        work_root: Path,
        history: tuple[Message[object], ...],
        call_id: str = "",
    ) -> tuple[ToolContext, ToolResponse | None]:
        context = ToolContext(deepcopy(dict(arguments)), work_root, history, call_id)
        decision = self.before.handle(context) if self.before is not None else None
        if isinstance(decision, ToolArguments):
            context = replace(
                context,
                arguments=deepcopy(dict(decision.arguments)),
                on_progress=decision.on_progress,
            )
            return context, None
        if decision is not None and not isinstance(decision, ToolResponse):
            raise TypeError(
                MODEL_TEXT.beforetool_handle_must_return_toolarguments_toolresponse_or_none
            )
        return context, decision

    def create_execution(
        self,
        arguments: dict[str, Any],
        *,
        call_id: str,
        work_root: Path,
        history: tuple[Message[object], ...],
    ) -> ToolExecutionOperation:
        """Create an execution handle; subclasses can add cancellation or progress."""
        return ToolExecutionOperation(run=lambda: self.execute(**arguments))

    def create_execution_with_hooks(
        self,
        arguments: dict[str, Any],
        *,
        call_id: str,
        work_root: Path,
        history: tuple[Message[object], ...],
    ) -> ToolExecutionOperation:
        """Apply hooks once to a private snapshot, including for custom handles."""
        arguments = deepcopy(arguments)
        history = tuple(message.copy() for message in history)

        def run() -> Any:
            context, decision = self._prepare_context(
                arguments, work_root, history, call_id
            )
            operation.record_arguments(context.arguments)
            if isinstance(decision, ToolResponse):
                result = decision.result
            else:
                try:
                    self.validate_arguments(context.arguments)
                except (TypeError, ValueError) as error:
                    guidance = self.argument_guidance or (
                        RUNTIME.argument_correction.format(tool_name=self.get_name())
                    )
                    raise ToolArgumentError.from_error(
                        error,
                        RUNTIME.invalid_arguments.format(
                            error=error, guidance=guidance
                        ),
                        hint=guidance,
                    ) from error
                work = self.create_execution(
                    deepcopy(context.arguments),
                    call_id=call_id,
                    work_root=work_root,
                    history=history,
                )
                operation.cancel = work.cancel
                # AfterTool filters complete results; raw progress must not bypass it.
                if self.after is None:
                    operation.read_partial = work.read_partial
                with operation_progress_context(context.on_progress):
                    result = work.run()
            return self._finalize_result(context, result)

        operation = ToolExecutionOperation(run)
        return operation

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_description_template(self) -> str:
        return self._description_template

    def render_description(self, renderer: Callable[[str], str]) -> None:
        description = renderer(self.get_description_template())
        if not isinstance(description, str):
            raise TypeError(MODEL_TEXT.tool_description_must_be_a_string)
        self.description = description.strip()

    def get_work_signature(self) -> inspect.Signature:
        """Business arguments accepted by this tool, before execution options."""
        return inspect.signature(self.tool)

    def get_signature(self) -> inspect.Signature:
        signature = self.get_work_signature()
        if not self.background_enabled:
            return signature
        parameters = [
            inspect.Parameter(
                "execution_mode",
                inspect.Parameter.KEYWORD_ONLY,
                default="wait",
                annotation=Literal["wait", "background"],
            ),
            inspect.Parameter(
                "wait_seconds",
                inspect.Parameter.KEYWORD_ONLY,
                default=RUNTIME.default_tool_wait_seconds,
                annotation=float,
            ),
        ]
        original = list(signature.parameters.values())
        insert_at = next(
            (
                index
                for index, value in enumerate(original)
                if value.kind == inspect.Parameter.VAR_KEYWORD
            ),
            len(original),
        )
        return signature.replace(
            parameters=[*original[:insert_at], *parameters, *original[insert_at:]]
        )

    def get_type_hints(self) -> dict[str, Any]:
        callable_tool = self.tool.func if isinstance(self.tool, partial) else self.tool
        hints = get_type_hints(callable_tool, include_extras=True)
        if self.background_enabled:
            hints["execution_mode"] = Annotated[
                Literal["wait", "background"],
                RUNTIME.execution_mode_guidance,
            ]
            hints["wait_seconds"] = Annotated[
                float,
                RUNTIME.initial_wait_guidance,
            ]
        return hints

    def with_description(self, description: str) -> Self:
        copied = self._copy(name=self.get_name(), description=description)
        copied._description_template = description
        return copied

    def copy(self) -> Self:
        """Copy the definition before any scope-specific description rendering."""
        return self.with_description(self.get_description_template())

    def exposes_parameter_descriptions(self) -> bool:
        """Whether provider schemas should include parameter documentation."""
        return self._parameter_descriptions_visible

    def without_parameter_descriptions(self) -> Self:
        """Copy the tool for a scope that loads its documentation separately."""
        copied = copy(self)
        copied._parameter_descriptions_visible = False
        return copied

    def with_name(self, name: str) -> Self:
        """Copy a tool under another registered name, preserving its behavior."""
        return self._copy(name=name, description=self.get_description())

    def with_result_validator(self, validator: Callable[[Any], Any]) -> Self:
        copied = copy(self)
        copied._result_validator = validator
        return copied

    def with_hooks(self, hooks: ToolHooks) -> Self:
        """Copy a tool with selected hooks replaced, preserving its behavior."""
        if not isinstance(hooks, ToolHooks):
            raise TypeError(MODEL_TEXT.hooks_must_be_a_toolhooks_instance)
        copied = copy(self)
        copied.before, copied.after = hooks.resolve(self.before, self.after)
        return copied

    def _copy(self, *, name: str, description: str) -> Self:
        copied = copy(self)
        copied.name = self._resolve_tool_name(self.tool, name)
        copied.description = description
        copied._check_inputs()
        return copied

    def without_background_execution(self) -> Self:
        """Keep structural tool overrides synchronous without mutating the source."""
        copied = copy(self)
        copied.background_enabled = False
        return copied

    def with_execution_policy_of(self, original: Tool) -> Self:
        """Copy with the original's interactive contract and synchronous requirements."""
        copied = copy(self)
        if not original.background_enabled or original.interactive is not None:
            copied.background_enabled = False
        if original.interactive is not None:
            copied.interactive = original.interactive
        return copied

    def with_description_suffix(self, suffix: str) -> Self:
        description = " ".join(
            part for part in (self.get_description(), suffix) if part
        )
        return self.with_description(description)

    def __str__(self) -> str:
        return MODEL_TEXT.name_type_tool_description.format(
            name=self.name,
            description=self.description[
                : CAPABILITY_FORMAT.representation_preview_chars
            ],
        )

    def __repr__(self) -> str:
        return "\n---\n" + str(self)


class CalledTool:
    def __init__(self, tool: Tool, tool_call: ToolCall) -> None:
        if not isinstance(tool, Tool):
            raise TypeError(MODEL_TEXT.tool_must_be_a_tool_instance)
        if not isinstance(tool_call, ToolCall):
            raise TypeError(MODEL_TEXT.tool_call_must_be_a_toolcall_instance)

        self._tool = tool
        self._tool_call = tool_call
        self._execution: ToolExecutionOperation | None = None
        self._validate_matching()

    @classmethod
    def from_tool_call(cls, tool: Tool, tool_call: ToolCall) -> CalledTool:
        return cls(tool, tool_call)

    @property
    def name(self) -> str:
        return self._tool.get_name()

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    @property
    def cache_result(self) -> bool:
        return self._tool.cache_result

    @property
    def background_enabled(self) -> bool:
        return self._tool.background_enabled

    @property
    def execution_options(self) -> ExecutionOptions:
        return ExecutionOptions.from_arguments(self.arguments)

    @property
    def work_arguments(self) -> dict[str, Any]:
        return self._tool.get_work_arguments(self.arguments)

    def execute(
        self,
        *,
        work_root: Path | None = None,
        history: tuple[Message[object], ...] = (),
    ) -> Any:
        return self.create_execution(
            work_root=work_root if work_root is not None else Path.cwd(),
            history=history,
        ).run()

    @property
    def executed_arguments(self) -> dict[str, Any] | None:
        return self._execution.get_executed_arguments() if self._execution else None

    def create_execution(
        self,
        *,
        work_root: Path,
        history: tuple[Message[object], ...],
    ) -> ToolExecutionOperation:
        self._execution = self._tool.create_execution_with_hooks(
            self.work_arguments,
            call_id=self.call_id,
            work_root=work_root,
            history=history,
        )
        return self._execution

    def without_background_execution(self) -> CalledTool:
        """Execute a runtime follow-up to completion before the next inference."""
        return CalledTool(self._tool.without_background_execution(), self._tool_call)

    def _validate_matching(self) -> None:
        call_name = self._tool_call.get_name()
        if call_name != self.name:
            raise ValueError(
                MODEL_TEXT.tool_call_name_does_not_match_tool_name.format(
                    call_name=call_name, name=self.name
                )
            )
