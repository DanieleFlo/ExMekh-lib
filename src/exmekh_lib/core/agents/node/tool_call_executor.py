from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.config.tool_errors import ERROR_HINTS
from exmekh_lib.core.agents.capabilities.prompts import CapabilityPrompts
from exmekh_lib.core.agents.deep_skill.deep_skill import CalledDeepSkill
from exmekh_lib.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from exmekh_lib.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from exmekh_lib.core.agents.exit_tool.items import ExitItemError
from exmekh_lib.core.file_inspection.evidence import ReadReport
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.output_access import output_file_access
from exmekh_lib.core.file_inspection.reading import FileReadContext
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.file_inspection.tokenization import inspection_tokenizer_context
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.message_snapshot import snapshot_message
from exmekh_lib.core.models.native_suite import native_suite_context
from exmekh_lib.core.models.tool import CalledTool, PreformattedToolResult
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.tool_result import ToolResult
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.execution_manager import (
    ExecutionLaunch,
    ExecutionSnapshot,
    ToolExecutionManager,
)
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.core.tool_calls.output_runtime import (
    ToolOutputDelivery,
    ToolOutputRuntime,
)
from exmekh_lib.utils.tool_errors import (
    ToolArgumentError,
    tool_failure_result,
)

from .agentic_scope import AgenticScope
from .tool_call_resolver import ResolvedToolCall
from .types import (
    CacheIdentity,
    NodeExecutionSnapshot,
    ToolCallOutcome,
    ToolExecutionResult,
)

logger = logging.getLogger(__name__)

type ExitContentResolver = Callable[[str, list[str], str, int | None], str]


class ToolCallExecutor:
    """Execute resolved tool calls and track nested deep-skill state."""

    def __init__(
        self,
        client: Client,
        exit_tool: ExitTool,
        initial_depth: int,
        agentic_scope: AgenticScope,
        exit_content_resolver: ExitContentResolver,
        work_root: Path | None = None,
        execution_manager: ToolExecutionManager | None = None,
        scope_prefix: str = "",
        output_preparer: ToolOutputPreparer | None = None,
        capability_prompts: CapabilityPrompts | None = None,
    ) -> None:
        self._client = client
        self._output_formatter = ToolOutputFormatter(client)
        self._output_preparer = output_preparer
        self._exit_tool = exit_tool
        self._depth = initial_depth
        self._agentic_scope = agentic_scope
        self._exit_content_resolver = exit_content_resolver
        self._work_root = work_root if work_root is not None else Path.cwd()
        self._skill_executions = DeepSkillExecutionTracker()
        self._execution_manager = execution_manager
        self._output_runtime = (
            ToolOutputRuntime(execution_manager, output_preparer)
            if execution_manager is not None
            else None
        )
        self._scope_prefix = scope_prefix
        self._file_tools = FileToolRuntime(execution_manager)
        self._capability_prompts = capability_prompts
        self._interactive_calls = FileInteractiveCalls(
            self._file_tools,
            self._work_root,
            self._client.get_history,
            lambda: False,
            lambda: self.skill_scope_ids,
        )

    def configure_interactive_calls(self, cancelled: Callable[[], bool]) -> None:
        self._interactive_calls.set_cancellation_check(cancelled)
        self._client.interactive_handler = self._interactive_calls

    def pending_interactive_inspections(self) -> tuple[ToolCall, ...]:
        return self._interactive_calls.pending_inspections()

    def begin_tool_batch(self) -> None:
        """Freeze prerequisites from context available when this batch was generated."""
        self._file_tools.begin_batch(self._client.get_history())

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def pending_return_call_id(self) -> str | None:
        return self._agentic_scope.pending_return_call_id

    @property
    def skill_path(self) -> str:
        return self._skill_executions.active_path

    @property
    def skill_executions(self) -> tuple[DeepSkillExecution, ...]:
        return self._skill_executions.roots

    @property
    def skill_scope_ids(self) -> tuple[str, ...]:
        return (self._scope_prefix, *self._skill_executions.active_scope_ids)

    @property
    def protected_history_indices(self) -> tuple[int, ...]:
        return self._skill_executions.protected_history_indices

    def rebase_active_history(self, index_map: tuple[int, ...]) -> None:
        self._skill_executions.rebase_active_history(index_map)

    def is_directly_in_skill(self, name: str) -> bool:
        return self._skill_executions.is_current_skill(name)

    def snapshot(self) -> NodeExecutionSnapshot:
        messages = self._client.copy_history()
        return NodeExecutionSnapshot(
            depth=self._depth,
            skill_path=self._skill_executions.active_path,
            history=[snapshot_message(message) for message in messages],
            history_tree=self._skill_executions.history_tree(messages),
        )

    def build_agentic_extra(self) -> dict[str, dict[str, object]]:
        extra = self._agentic_scope.build_extra(
            self._skill_executions.active_skills,
            self._depth,
        )
        logger.info(
            RUNTIME_TEXT.agentic_scope_path_depth_scope,
            self.skill_path,
            self.depth,
            extra["agentic"],
        )
        return extra

    def acknowledge_agentic_extra(self) -> None:
        self._agentic_scope.acknowledge_extra()

    def validate_tool_calls(self, calls: list[ToolCall]) -> None:
        self._agentic_scope.validate_tool_calls(
            calls, self._skill_executions.active_skills, self._depth
        )

    def record_automatic_tool_call(
        self, call: ToolCall
    ) -> AbstractContextManager[None]:
        # Native skill returns consume their latest call batch and final reply.
        return self._client.record_tool_call(
            call, batch_call_id=self._agentic_scope.pending_return_call_id
        )

    def record_steering_output(self) -> str:
        """Expose saved guidance in every enclosing skill's output references."""
        return self.record_runtime_output(RUNTIME.steering_output_name)

    def record_runtime_output(self, name: str) -> str:
        output_id = self._output_formatter.generate_output_id()
        self._skill_executions.record_tool_output(name, output_id)
        return output_id

    def execute(
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
        history_position: int = 0,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        supported_types = (CalledDeepSkill, CalledExitTool, CalledTool)
        if not isinstance(tool_call, supported_types):
            raise TypeError(
                RUNTIME_TEXT.node_supports_only_deep_skills_exit_tools_and_tools
            )

        if isinstance(tool_call, CalledExitTool):
            return self._prepare_result(
                self._run_exit_tool(
                    tool_call,
                    thinking=thinking,
                    close_scope=close_scope,
                    thinking_annotation_provider=thinking_annotation_provider,
                )
            )
        if isinstance(tool_call, CalledDeepSkill):
            return self._run_deep_skill(
                tool_call, thinking=thinking, history_position=history_position
            )
        result = self._run_tool(
            tool_call, thinking_annotation_provider=thinking_annotation_provider
        )
        if result.output_id is not None:
            self._skill_executions.record_tool_output(tool_call.name, result.output_id)
        return result

    def _run_deep_skill(
        self,
        called_deep_skill: CalledDeepSkill,
        *,
        thinking: str | None,
        history_position: int,
    ) -> ToolExecutionResult:
        self._open_deep_skill(
            called_deep_skill, thinking=thinking, history_position=history_position
        )
        context = None
        if self._capability_prompts is not None:
            context = self._capability_prompts.skill_context(
                called_deep_skill.name,
                self._agentic_scope.get_allowed_tool_names(
                    self._skill_executions.active_skills, self._depth
                ),
            )
        content = called_deep_skill.render_instructions(
            self._exit_tool.render_prompt_section(context=context),
            context=context,
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_deep_skill.call_id,
            name=called_deep_skill.name,
        )
        return ToolExecutionResult(response=tool_response)

    def _run_tool(
        self,
        called_tool: CalledTool,
        *,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        logger.info(
            RUNTIME_TEXT.executing_tool_name_call_id_depth,
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        snapshot = None
        elapsed_seconds = None
        inspection_path = None
        failed = False
        started_clock = monotonic()
        try:
            if called_tool.background_enabled and self._execution_manager is not None:
                result, snapshot = self._run_managed_tool(
                    called_tool, self._execution_manager
                )
            else:
                with (
                    self._tool_run_context() as context,
                    self._interactive_calls.execution_context(called_tool.call_id),
                ):
                    try:
                        result = called_tool.execute(
                            work_root=self._work_root,
                            history=tuple(self._client.get_history()),
                        )
                    finally:
                        inspection_path = context.pending_inspection_path()
                elapsed_seconds = monotonic() - started_clock
        except Exception as error:
            result = tool_failure_result(error)
            failed = True
            elapsed_seconds = monotonic() - started_clock
        return self._format_tool_result(
            called_tool,
            ToolCallOutcome(
                value=result,
                execution=snapshot,
                elapsed_seconds=elapsed_seconds,
                inspection_path=inspection_path,
                failed=failed,
            ),
            thinking_annotation_provider=thinking_annotation_provider,
        )

    def _tool_run_context(self) -> AbstractContextManager[FileReadContext]:
        # Capture dependencies now; enter their contexts in the execution thread.
        tokenizer_context = inspection_tokenizer_context(
            self._client.get_tokenizer(),
            self._output_preparer.settings
            if self._output_preparer is not None
            else None,
        )
        output_context = output_file_access(
            self._output_preparer.files if self._output_preparer is not None else None
        )
        read_context = self._file_tools.read_context(self._client.get_history())

        @contextmanager
        def enter_contexts() -> Iterator[FileReadContext]:
            with (
                tokenizer_context,
                output_context,
                read_context as context,
                native_suite_context(self._client.native_suite),
            ):
                yield context

        return enter_contexts()

    def _run_managed_tool(
        self, called_tool: CalledTool, manager: ToolExecutionManager
    ) -> tuple[Any, ExecutionSnapshot]:
        output_id = self._output_formatter.generate_output_id()
        operation = called_tool.create_execution(
            work_root=self._work_root,
            history=tuple(self._client.get_history()),
        )
        operation.add_run_context(self._tool_run_context())
        try:
            options = called_tool.execution_options
        except (TypeError, ValueError) as error:
            raise ToolArgumentError.from_error(
                error,
                RUNTIME_TEXT.invalid_arguments.format(error=error),
                code="invalid_execution_options",
            ) from error
        snapshot = manager.launch(
            ExecutionLaunch(
                call=called_tool.tool_call,
                operation=operation,
                owner_scope=self.skill_scope_ids,
                options=options,
                output_id=output_id,
                skill_path=self.skill_path,
            )
        )
        result = snapshot.output_result()
        if snapshot.is_finished:
            return result, snapshot
        try:
            partial = manager.partial_output(output_id, 0)
            if partial is not None:
                result = {
                    **result,
                    "partial_output": partial[0],
                    "next_cursor": partial[1],
                }
        except Exception as error:
            result = {
                **result,
                "partial_output_error": str(error),
                "partial_output_hint": ERROR_HINTS["partial_output_failed"],
            }
        return result, snapshot

    def _format_tool_result(
        self,
        called_tool: CalledTool,
        outcome: ToolCallOutcome,
        *,
        thinking_annotation_provider: Callable[[], str | None] | None,
    ) -> ToolExecutionResult:
        result = outcome.value
        snapshot = outcome.execution
        output = ToolResult(result)
        content, output_id = self._output_formatter.format_tool_result(
            called_tool.name,
            result,
            annotation_provider=thinking_annotation_provider,
            execution=snapshot,
            elapsed_seconds=outcome.elapsed_seconds,
            failed=outcome.delivery_failed,
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_tool.call_id,
            name=called_tool.name,
            attachments=output.attachments(),
            external_output=output.external_output(),
        )
        execution_result = ToolExecutionResult(
            response=tool_response,
            raw_result=output.storage_value(),
            executed_arguments=(
                snapshot.output_arguments()
                if snapshot is not None
                else called_tool.executed_arguments
            ),
            save_result=called_tool.cache_result,
            output_id=output_id,
            execution=snapshot,
            elapsed_seconds=outcome.elapsed_seconds,
            inspection_path=outcome.inspection_path,
            failed=outcome.delivery_failed,
            external_output=output.external_output(),
        )
        # File reads measure selected text, excluding XML line labels and framing.
        budget_text = result.selected_text() if isinstance(result, ReadReport) else None
        execution_result = self._prepare_result(
            execution_result, budget_text=budget_text
        )
        self._file_tools.record_result(
            called_tool.tool_call, result, execution_result.response
        )
        return execution_result

    def _prepare_result(
        self, result: ToolExecutionResult, *, budget_text: str | None = None
    ) -> ToolExecutionResult:
        message = result.response
        output_id = result.output_id or message.get_tool_call_id()
        presentation = ToolOutputPresentation(
            ToolOutputReference(message.get_name(), output_id),
            PreformattedToolResult(
                message.get_message(),
                attachments=message.get_attachments(),
                external_output=result.external_output,
                delivery_failed=result.failed,
            ),
            budget_text=budget_text,
        )
        if (
            result.execution is not None
            and result.execution.is_finished
            and self._output_runtime is not None
        ):
            delivery = self._output_runtime.publish_final(
                result.execution, presentation
            )
            if (
                delivery.final_execution is not None
                and self._execution_manager is not None
            ):
                self._execution_manager.acknowledge_output(delivery.final_execution)
            return result.with_delivery(delivery, output_id)
        if self._output_preparer is None or result.external_output is not None:
            return result
        prepared = self._output_preparer.prepare(presentation)
        if not prepared.changed:
            return result
        output_id = result.output_id or self._output_formatter.generate_output_id()
        content = prepared.content
        if result.output_id is None:
            content = XmlHeader.add_field_to_content(content, "id", output_id)
        response = replace(prepared.as_result(presentation.result), content=content)
        return result.with_delivery(
            ToolOutputDelivery(response, failure=prepared.failure), output_id
        )

    def _running_tool_outputs(self) -> tuple[ExecutionSnapshot, ...]:
        if self._execution_manager is None:
            return ()
        return self._execution_manager.running_outputs(self.skill_scope_ids)

    def _run_exit_tool(
        self,
        called_exit_tool: CalledExitTool,
        *,
        thinking: str | None,
        close_scope: bool = True,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        logger.info(
            RUNTIME_TEXT.executing_exit_tool_name_call_id_active_skills_depth,
            called_exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        try:
            result = called_exit_tool.execute(
                work_root=self._work_root,
                history=tuple(self._client.get_history()),
            )
        except (TypeError, ValueError) as error:
            tool_name = called_exit_tool.name
            diagnostic = (
                str(error)
                if isinstance(error, ToolArgumentError)
                else RUNTIME_TEXT.invalid_arguments.format(error=error)
            )
            return self._recoverable_error(
                called_exit_tool,
                RUNTIME_TEXT.use_text_result_items_or_text_items_id_output.format(
                    diagnostic=diagnostic, tool_name=tool_name, tool_name_2=tool_name
                ),
            )
        content = result["text"]
        items = result["items"]
        opening_position = (
            self._skill_executions.current.opening_history_position
            if self._skill_executions.depth
            else None
        )
        try:
            content = self._exit_content_resolver(
                content, items, self.skill_path, opening_position
            )
        except ExitItemError as error:
            return self._recoverable_error(called_exit_tool, str(error))
        result = {**result, "text": content}

        if not close_scope:
            return ToolExecutionResult(
                response=ToolMessage(
                    content, called_exit_tool.call_id, called_exit_tool.name
                ),
                raw_result=result,
                save_result=True,
            )

        if self._skill_executions.depth:
            elapsed_seconds = self._skill_executions.current.elapsed_seconds
            content, output_id = self._output_formatter.format_skill_return(
                self._skill_executions.current.name,
                content,
                thinking=thinking,
                annotation_provider=thinking_annotation_provider,
                background_tools=self._running_tool_outputs(),
                used_tools=self._skill_executions.current.used_tools,
                elapsed_seconds=elapsed_seconds,
            )
            return self._close_deep_skill(
                ToolExecutionResult(
                    response=ToolMessage(
                        content,
                        self._skill_executions.current.opening_tool_call_id,
                        self._skill_executions.current.name,
                    ),
                    raw_result=result,
                    output_id=output_id,
                    elapsed_seconds=elapsed_seconds,
                )
            )

        self._depth = max(0, self._depth - 1)
        logger.warning(
            RUNTIME_TEXT.exit_tool_executed_without_an_active_deep_skill_call,
            called_exit_tool.call_id,
            self._depth,
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_exit_tool.call_id,
            name=called_exit_tool.name,
        )
        return ToolExecutionResult(
            response=tool_response, raw_result=result, save_result=True
        )

    def _recoverable_error(
        self, called_tool: CalledTool | CalledExitTool, message: str
    ) -> ToolExecutionResult:
        result = {
            "ok": False,
            "error_kind": "invalid_tool_arguments",
            "error": (
                RUNTIME_TEXT.error_correct_the_error_and_call_again.format(
                    called_tool_name=called_tool.name,
                    message=message,
                    called_tool_name_2=called_tool.name,
                )
            ),
        }
        content, output_id = self._output_formatter.format_tool_result(
            called_tool.name, result, failed=True
        )
        return ToolExecutionResult(
            response=ToolMessage(
                content=content,
                tool_call_id=called_tool.call_id,
                name=called_tool.name,
            ),
            raw_result=result,
            save_result=True,
            output_id=output_id,
            failed=True,
        )

    def _open_deep_skill(
        self,
        called_deep_skill: CalledDeepSkill,
        *,
        thinking: str | None,
        history_position: int,
    ) -> None:
        batch = self._client.get_latest_tool_batch()
        called_deep_skill.open_execution(
            self._skill_executions,
            batch.opening_index,
            thinking,
            batch.preceding_responses,
            history_position,
        )
        self._depth += 1
        logger.info(
            RUNTIME_TEXT.opened_deep_skill_name_call_id_start_index_depth,
            called_deep_skill.name,
            called_deep_skill.call_id,
            batch.opening_index,
            self._depth,
        )

    def _close_deep_skill(self, result: ToolExecutionResult) -> ToolExecutionResult:
        prepared_result = self._prepare_result(result)
        output = prepared_result.response.get_message()
        history = self._client.copy_history()
        start_index = self._skill_executions.get_current_start_index()
        execution = self._skill_executions.close(
            end_index=len(history) - 1,
            output=output,
            history=history[start_index:],
        )
        response = ToolMessage(
            content=output,
            tool_call_id=execution.opening_tool_call_id,
            name=execution.name,
            external_output=prepared_result.external_output,
        )
        self._depth = max(0, self._depth - 1)
        self._agentic_scope.register_skill_return(execution.opening_tool_call_id)
        logger.info(
            RUNTIME_TEXT.closed_deep_skill_name_opening_call_id_start_index,
            execution.name,
            execution.opening_tool_call_id,
            execution.start_index,
            execution.end_index,
            len(execution.history),
            self._depth,
        )
        return ToolExecutionResult(
            response=response,
            raw_result=prepared_result.raw_result,
            save_result=True,
            replace_history_from=execution.start_index + 1,
            preserved_responses=execution.preceding_tool_responses,
            output_id=prepared_result.output_id,
            used_tools=execution.used_tools,
            external_output=prepared_result.external_output,
            failed=prepared_result.failed,
            elapsed_seconds=prepared_result.elapsed_seconds,
            cache_identity=CacheIdentity(
                name=execution.name,
                source="DeepSkill",
                tool_call_id=execution.opening_tool_call_id,
                args=execution.opening_arguments,
                thinking=execution.opening_thinking,
                history_position=execution.opening_history_position,
            ),
        )
