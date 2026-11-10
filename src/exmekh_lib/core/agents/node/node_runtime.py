from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Final, Literal
from uuid import uuid4

from exmekh_lib.capabilities.tools.bash.bash_tool import BashTool
from exmekh_lib.capabilities.tools.control_tool_execution.execution_control_tool import (
    ExecutionControlTool,
)
from exmekh_lib.capabilities.tools.read_tool_output.summary import SummaryMode
from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.inventory import (
    CapabilityInventory,
    build_capability_inventory,
)
from exmekh_lib.core.agents.capabilities.prompts import (
    prepare_node_prompts,
)
from exmekh_lib.core.agents.capabilities.tool_composer import _DefaultToolComposer
from exmekh_lib.core.agents.deep_agent.agent_response import AgentResponse
from exmekh_lib.core.agents.deep_agent.stream_event import (
    DeepAgentCompactionProgressEvent,
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentHistoryCompactionEvent,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.agents.deep_skill.deep_skill import CalledDeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import CalledExitTool
from exmekh_lib.core.agents.exit_tool.items import ExitItemResolver
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.file_inspection.runtime import FileInspectionRequests
from exmekh_lib.core.history_compaction import HistoryCompactionPolicy, HistoryCompactor
from exmekh_lib.core.history_compaction.projection import compaction_spans
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.request_budget import RequestBudget
from exmekh_lib.core.models.tool import CalledTool, PreformattedToolResult
from exmekh_lib.core.models.user_message import ErrorUserMessage, UserMessage
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.provider import ProviderRequestRejectedError

from .node_runtime_config import NodeRuntimeConfig
from .thinking_annotation import ThinkingAnnotationResolver
from .tool_call_executor import ToolCallExecutor
from .tool_call_resolver import ResolvedToolCall, ToolCallResolver
from .tool_result_recorder import ToolResultRecorder
from .turn_control import TurnControl
from .types import NodeUserMessage, ToolExecutionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DeferredToolBatch:
    depth: int
    calls: list[ResolvedToolCall]
    thinking: str | None


class NodeRuntime(ProviderEventHandler):
    """Own the conversational lifecycle of an initialized node."""

    REQUIRED_TOOL_CHOICE: Final[str] = RUNTIME.required_tool_choice
    EMPTY_RESPONSE_MESSAGE: Final[str] = RUNTIME.empty_response_message

    def __init__(self, config: NodeRuntimeConfig) -> None:
        self._config = config
        self._history_compaction_policy: HistoryCompactionPolicy | None = None
        self._compaction_budget: RequestBudget | None = None
        self._node_id = f"{config.get_hds_name()}-{uuid4().hex}"
        self._tool_cache: NodeCache | None = None
        self._execution_history = ExecutionHistory()
        self._output_preparer = ToolOutputPreparer(
            config.provider.get_tokenizer(),
            OutputFileWorkspace(self._get_tool_cache, config.work_root, None),
        )
        self._execution_manager = ToolExecutionManager()
        self._result_recorder = ToolResultRecorder(
            self._execution_history,
            self._get_tool_cache,
            self._execution_manager,
            self._output_preparer,
        )
        self._turn_control = TurnControl(self._execution_manager)
        self._memory_tools = HarnessMemoryTools(
            self._execution_history,
            self._get_tool_cache,
            self._execution_manager,
            config.work_root,
            provider=config.provider,
            output_preparer=self._output_preparer,
        )
        self._exit_items = ExitItemResolver(
            self._execution_history,
            self._get_tool_cache,
        )
        self._run_start_position = 0
        self._thinking_annotation_resolver = ThinkingAnnotationResolver(config.provider)
        defaults = config.default_capabilities
        self._execution_control = ExecutionControlTool(
            self._execution_manager, config.work_root
        )
        harness_tools = (
            *self._memory_tools.tools,
            self._execution_control,
            BashTool(work_root=config.work_root),
        )
        harness_tools = _DefaultToolComposer(defaults).compose_runtime_tools(
            harness_tools
        )
        self._runtime_tools = (*config.runtime_tools, *harness_tools)
        self._validate_runtime_tool_names()
        prepared = prepare_node_prompts(
            config.capabilities,
            config.get_hds_name(),
            self._runtime_tools,
            config.initial_depth,
        )
        self._capability_prompts = prepared.prompts
        capabilities = prepared.capabilities
        self._config = replace(
            config,
            capabilities=capabilities,
            hds=prepared.hds,
            exit_tool=capabilities.get_exit_tool(prepared.hds.get_exit_tool_name()),
            runtime_tools=self._runtime_tools,
        )
        self._resolver = ToolCallResolver(
            capabilities,
            self._runtime_tools,
        )
        self._client: Client
        self._executor: ToolCallExecutor
        self._reset_session()

    @property
    def hds(self) -> HDS:
        return self._config.hds

    @property
    def capabilities(self) -> CapabilitiesData:
        return self._config.capabilities

    def get_capability_inventory(self) -> CapabilityInventory:
        context = self._config.hds.prompt_context or self._capability_prompts.context(
            self._config.hds
        )
        defaults = self._config.default_capabilities
        return build_capability_inventory(
            self.capabilities,
            self._runtime_tools,
            context,
            frozenset(defaults.overrides) if defaults is not None else frozenset(),
        )

    @property
    def depth(self) -> int:
        return self._executor.depth

    @property
    def agent_context_id(self) -> str:
        return self._config.resolved_agent_context_id

    def get_tools_text(self) -> str:
        return self._config.provider.get_tools_text(self._client.tools)

    def control_tool_execution(
        self,
        tool_call_id: str,
        action: Literal["wait", "cancel"] = "wait",
        wait_seconds: float = RUNTIME.default_tool_wait_seconds,
    ) -> dict[str, Any]:
        return self._execution_control.control_tool_execution(
            tool_call_id, action, wait_seconds
        )

    def read_tool_output(
        self,
        output_id: str,
        cursor: int | None = None,
        summary_mode: SummaryMode = "none",
    ) -> PreformattedToolResult:
        return self._memory_tools.read_tool_output(output_id, cursor, summary_mode)

    def _validate_runtime_tool_names(self) -> None:
        capability_names = {
            tool.get_name() for tool in self._config.capabilities.get_all_tools()
        }
        runtime_names = tuple(tool.get_name() for tool in self._runtime_tools)
        duplicate_runtime_names = {
            name for name in runtime_names if runtime_names.count(name) > 1
        }
        if duplicate_runtime_names:
            names = ", ".join(sorted(duplicate_runtime_names))
            raise ValueError(
                RUNTIME_TEXT.harness_runtime_tool_names_must_be_unique.format(
                    names=names
                )
            )

        collisions = capability_names.intersection(runtime_names)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                RUNTIME_TEXT.harness_runtime_tool_names_are_reserved.format(names=names)
            )

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        with self._turn_control.run():
            return self._run_turn(user_message)

    def compact_history(self) -> None:
        """Compact now while idle, or at the next safe boundary of an active run."""
        self._turn_control.compact_history(self._compact_history)

    def set_history_compaction_policy(
        self,
        policy: HistoryCompactionPolicy | None,
        *,
        budget: RequestBudget | None = None,
    ) -> None:
        self._history_compaction_policy = policy
        self._compaction_budget = budget if policy is not None else None

    def _prepare_inference(self, request: ProviderRequest) -> ProviderRequest:
        # Client has admitted user input and completed tool replies here. Rebuild
        # the request only after the owner has replaced history and rebased scopes.
        policy = self._history_compaction_policy
        if policy is None:
            return request
        budget = self._compaction_budget
        oversized = budget is not None and not budget.fits(request)
        eligible = compaction_spans(
            self._client.get_history(), self._executor.protected_history_indices
        )
        requested = bool(eligible) and policy.before_inference(request)
        # A hard capacity check must also override a disarmed threshold policy
        # and cover histories made entirely of protected user/system messages.
        if not requested and not oversized:
            return request
        try:
            self._compact_history(request)
        except Exception as error:
            # Sending the unmodified request after the policy asked to compact
            # can exhaust the context before there is another safe boundary.
            logger.exception(HISTORY_COMPACTION.automatic_failure)
            raise RuntimeError(HISTORY_COMPACTION.automatic_failure) from error
        steering = (
            self._turn_control.take_user_messages()
            if self._executor.pending_return_call_id is None
            else ()
        )
        for message in steering:
            self._client.add_user_message(message)
        compacted_request = self._request_with_history(
            request, self._client.get_history()
        )
        compacted_request.notify_steering_added(steering)
        policy.after_compaction(compacted_request)
        if budget is not None:
            budget.require_fits(compacted_request)
        return compacted_request

    def _request_with_history(
        self, request: ProviderRequest, history: list[Message[object]]
    ) -> ProviderRequest:
        extra = {**(request.get_extra() or {}), **self._executor.build_agentic_extra()}
        return request.with_messages(history).with_extra(extra)

    def _compact_pending_history(self) -> None:
        if self._turn_control.take_compaction_request():
            self._compact_history()

    def _compact_history(self, request: ProviderRequest | None = None) -> None:
        original = self._client.get_history()
        records = {record.id: record for record in self._get_tool_cache().get_all()}
        records.update(
            {
                record.id: record
                for record in self._execution_manager.output_records(records)
            }
        )
        compactor = HistoryCompactor(
            self._config.provider,
            output_lookup=records.get,
            files=self._output_preparer.files,
            budget=self._compaction_budget,
            on_progress=lambda progress: self._config.emit_stream(
                DeepAgentCompactionProgressEvent(
                    depth=self._executor.depth,
                    skill_path=self._executor.skill_path,
                    agent_context_id=self.agent_context_id,
                    progress=progress,
                )
            ),
        )
        protected = self._executor.protected_history_indices
        if self._executor.pending_return_call_id is not None:
            # The native return protocol still needs the original final batch.
            protected = (*protected, len(original) - 1)
        result = compactor.compact(
            self._client.create_history_checkpoint(),
            tools=self._client.tools,
            protected_indices=protected,
        )
        if self._client.get_history() != original:
            raise RuntimeError(HISTORY_COMPACTION.changed_history)
        replacement = result.replacement_for(original)
        budget = self._compaction_budget
        if request is not None and budget is not None:
            candidate = self._request_with_history(request, replacement)
            # Short retained outputs can grow when wrapped in a synthetic batch.
            # Commit only a smaller request that fits, keeping originals and
            # scope anchors available when compaction cannot satisfy the budget.
            candidate_tokens = budget.count_tokens(candidate)
            if (
                candidate_tokens > budget.prompt_limit
                or candidate_tokens >= budget.count_tokens(request)
            ):
                return
        self._client.replace_history_from(0, replacement)
        self._executor.rebase_active_history(result.index_map)
        if result.replaced_history:
            snapshot = self._executor.snapshot()
            event = DeepAgentHistoryCompactionEvent(
                depth=snapshot.depth,
                skill_path=snapshot.skill_path,
                history=snapshot.history,
                history_tree=snapshot.history_tree,
                replaced_history=result.replaced_history,
                agent_context_id=self.agent_context_id,
            )
            try:
                self._config.emit_stream(event)
            except Exception:
                # Applied history must never be paired with the stale request
                # because an optional frontend observer failed to render it.
                logger.exception(HISTORY_COMPACTION.observer_failure)

    def steer(self, user_message: NodeUserMessage) -> None:
        """Accept guidance without mutating history or invoking the provider."""
        self._turn_control.steer(user_message)

    def _continue_with_steering(self) -> ProviderResponse | None:
        self._compact_pending_history()
        if self._executor.pending_return_call_id is not None:
            return None
        messages = self._turn_control.take_user_messages()
        if not messages:
            return None
        return self._client.continue_with_user_messages(
            messages,
            extra=self._executor.build_agentic_extra(),
            tool_choice=self._current_tool_choice(),
            streaming=self._config.is_streaming(),
            on_event=self,
        )

    def _continue_after_tools(self) -> ProviderResponse:
        steered_response = self._continue_with_steering()
        if steered_response is not None:
            return steered_response
        return self._client.continue_after_tools(
            extra=self._executor.build_agentic_extra(),
            tool_choice=self._current_tool_choice(),
            streaming=self._config.is_streaming(),
            on_event=self,
        )

    def _finish_return_before_deferred_calls(self) -> None:
        self._client.complete_tool_transition(
            extra=self._executor.build_agentic_extra()
        )
        self._executor.acknowledge_agentic_extra()

    def _run_turn(self, user_message: NodeUserMessage) -> AgentResponse:
        if not self._config.is_persistent_history():
            self._reset_session()

        self._thinking_annotation_resolver.begin_turn()
        self._run_start_position = self._execution_history.count
        logger.info(
            RUNTIME_TEXT.starting_node_run_node_depth,
            self._config.get_hds_name(),
            self.depth,
        )
        self._result_recorder.sync_completed()
        pending_result = self._execution_manager.next_delivery(
            self._executor.skill_scope_ids
        )
        if pending_result is not None and self._has_read_tool():
            input_message = UserMessage(content=user_message)
            self._client.add_user_message(input_message)
            self._deliver_completed_tool_outputs()
            try:
                response = self._continue_after_tools()
            except ProviderRequestRejectedError:
                # Keep completed background output; remove only the rejected input.
                self._client.remove_rejected_input([input_message])
                raise
        else:
            response = self._client.continue_with_user_messages(
                self._turn_control.take_user_messages(),
                initial_message=UserMessage(content=user_message),
                extra=self._executor.build_agentic_extra(),
                tool_choice=self._current_tool_choice(),
                streaming=self._config.is_streaming(),
                on_event=self,
            )
        response = self._route_until_complete(response)
        logger.info(
            RUNTIME_TEXT.completed_node_run_node_depth_root_executions,
            self._config.get_hds_name(),
            self.depth,
            len(self._executor.skill_executions),
        )
        return AgentResponse(
            response=response,
            usage=response.usage,
            thinking=response.thinking,
            history=self._client.dump_history(),
            skill_executions=self._executor.skill_executions,
        )

    def restore_history(
        self,
        messages: Sequence[Message[Any]],
        restore_execution_state: Callable[[ExecutionHistory, NodeCache], None]
        | None = None,
    ) -> None:
        """Restore messages and tool state for a persistent node."""
        if not self._config.is_persistent_history():
            raise ValueError(
                RUNTIME_TEXT.cannot_restore_history_for_a_non_persistent_node
            )
        self._client.replace_history_from(1, list(messages))
        if restore_execution_state is not None:
            restore_execution_state(
                self._execution_history,
                self._get_tool_cache(),
            )

    def _route_until_complete(self, response: ProviderResponse) -> ProviderResponse:
        while True:
            tool_calls = response.get_tool_calls()
            if tool_calls:
                logger.info(
                    RUNTIME_TEXT.tool_calls.format(
                        names=[tool.get_name() for tool in tool_calls]
                    )
                )
                resolved_tool_calls = self._prepare_tool_batch(tool_calls)
                thinking = response.thinking.thoughts if response.thinking else None
                self._execute_batch(resolved_tool_calls, thinking)
                self._execute_deferred_batches()

                if response.interruptions:
                    self._record_interruptions(response)
                    response = self._continue_after_interruption()
                    continue

                self._result_recorder.sync_completed()
                self._deliver_completed_tool_outputs()

                response = self._continue_after_tools()
                self._executor.acknowledge_agentic_extra()
                continue

            if response.interruptions:
                self._record_interruptions(response)
                response = self._continue_after_interruption()
                continue

            content = response.get_message()
            if isinstance(content, str) and content.strip():
                self._result_recorder.sync_completed()
                if self._deliver_completed_tool_outputs():
                    response = self._continue_after_tools()
                    continue
                steered_response = self._continue_with_steering()
                if steered_response is not None:
                    response = steered_response
                    continue
                if self._turn_control.try_close():
                    return response
                continue

            self._result_recorder.sync_completed()
            self._deliver_completed_tool_outputs()
            steered_response = self._continue_with_steering()
            if steered_response is not None:
                response = steered_response
                continue
            response = self._client.call(
                ClientRequest(
                    message=ErrorUserMessage(self.EMPTY_RESPONSE_MESSAGE),
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=self._current_tool_choice(),
                    streaming=self._config.is_streaming(),
                    on_event=self,
                )
            )

    def _prepare_tool_batch(self, tool_calls: list[ToolCall]) -> list[ResolvedToolCall]:
        """Validate against the generating scope before any execution changes it."""
        self._executor.validate_tool_calls(tool_calls)
        return self._resolver.resolve_all(tool_calls)

    def _record_interruptions(self, response: ProviderResponse) -> None:
        if self._executor.pending_return_call_id is not None:
            self._finish_return_before_deferred_calls()
        for message in response.interruptions:
            self._client.add_user_message(message)
            self._result_recorder.persist_steering(
                message,
                self._executor.record_runtime_output("interactive_call_interrupted"),
                self._executor.skill_path,
                name="interactive_call_interrupted",
            )
        calls = self._executor.pending_interactive_inspections()
        if calls:
            self._inspect_missing_files(calls)

    def _continue_after_interruption(self) -> ProviderResponse:
        steered = self._continue_with_steering()
        if steered is not None:
            return steered
        return self._client.continue_after_interruption(
            extra=self._executor.build_agentic_extra(),
            tool_choice=self._current_tool_choice(),
            streaming=self._config.is_streaming(),
            on_event=self,
        )

    def _execute_deferred_batches(self) -> None:
        while self._pending_batches and self._pending_batches[-1].depth == self.depth:
            self._finish_return_before_deferred_calls()
            batch = self._pending_batches.pop()
            self._client.add_tool_calls([call.tool_call for call in batch.calls])
            self._execute_batch(batch.calls, batch.thinking)

    def _execute_batch(
        self, calls: list[ResolvedToolCall], thinking: str | None
    ) -> None:
        tools = [
            call
            for call in calls
            if not isinstance(call, (CalledDeepSkill, CalledExitTool))
        ]
        skills = [call for call in calls if isinstance(call, CalledDeepSkill)]
        exits = [call for call in calls if isinstance(call, CalledExitTool)]
        active_calls = tools + skills[:1] if skills else tools + exits
        deferred_calls: list[ResolvedToolCall] = [*skills[1:], *exits] if skills else []

        annotation_provider = (
            partial(self._thinking_annotation_resolver.resolve, thinking)
            if self._executor.is_directly_in_skill(RUNTIME.deep_thinking_skill_name)
            else None
        )

        if deferred_calls:
            # Deferred siblings must not appear as unanswered calls in the child scope.
            self._client.replace_latest_tool_calls(
                [call.tool_call for call in active_calls]
            )
            self._pending_batches.append(
                _DeferredToolBatch(self.depth, deferred_calls, thinking)
            )

        inspections = FileInspectionRequests(
            self._config.work_root, self._executor.skill_scope_ids
        )
        self._executor.begin_tool_batch()
        for index, call in enumerate(active_calls):
            result = self._execute_tool_call(
                call,
                thinking=thinking,
                close_scope=index == len(active_calls) - 1,
                thinking_annotation_provider=annotation_provider,
            )
            if result.inspection_path is not None:
                inspections.add(result.inspection_path)
        if inspection_calls := inspections.create_calls(self._executor.skill_scope_ids):
            self._inspect_missing_files(inspection_calls)

    def _inspect_missing_files(self, calls: tuple[ToolCall, ...]) -> None:
        # Only append: all original results precede this new assistant batch.
        self._client.add_tool_calls(calls)
        for call in calls:
            resolved = self._resolver.resolve(call)
            assert isinstance(resolved, CalledTool)
            self._execute_tool_call(resolved.without_background_execution())

    def _execute_tool_call(
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        self._result_recorder.sync_completed()
        skill_path = self._executor.skill_path
        identity = self._result_recorder.record_call(tool_call, thinking)
        self._emit_tool_state(
            tool_call.name, tool_call.call_id, DeepAgentExecutionEventType.TOOL_START
        )
        failed = True
        try:
            logger.info(
                RUNTIME_TEXT.starting_tool_execution_name_call_id_path,
                tool_call.name,
                tool_call.call_id,
                self._executor.skill_path,
            )
            result = self._executor.execute(
                tool_call,
                thinking=thinking,
                close_scope=close_scope,
                history_position=identity.history_position,
                thinking_annotation_provider=thinking_annotation_provider,
            )
            if result.replace_history_from is None:
                self._client.add_tool_response(result.response)
            else:
                self._client.replace_history_from(
                    result.replace_history_from,
                    messages=[*result.preserved_responses, result.response],
                )
            self._result_recorder.persist_result(result, identity, skill_path)
            failed = result.is_error
            return result
        finally:
            self._emit_tool_state(
                tool_call.name,
                tool_call.call_id,
                DeepAgentExecutionEventType.TOOL_END,
                failed=failed,
            )

    def _deliver_completed_tool_outputs(self) -> bool:
        if not self._has_read_tool():
            return False
        delivered = False
        while snapshot := self._execution_manager.next_delivery(
            self._executor.skill_scope_ids
        ):
            output_id = snapshot.output_id
            if output_id is None:
                break
            call = ToolCall(
                id=uuid4().hex,
                function=FunctionCall(
                    name=RUNTIME.read_tool_name,
                    arguments={"output_id": output_id},
                ),
            )
            result = self._execute_automatic_output_read(call)
            if result.is_error:
                break
            self._execution_manager.acknowledge_output(snapshot)
            delivered = True
        return delivered

    def _execute_automatic_output_read(self, call: ToolCall) -> ToolExecutionResult:
        with self._executor.record_automatic_tool_call(call):
            return self._execute_tool_call(self._resolver.resolve(call))

    def _has_read_tool(self) -> bool:
        return any(
            tool.get_name() == RUNTIME.read_tool_name for tool in self._runtime_tools
        )

    def _get_tool_cache(self) -> NodeCache:
        if self._tool_cache is None or not self._tool_cache.is_active:
            self._tool_cache = self._config.build_node_cache(self._node_id)
        return self._tool_cache

    def _reset_session(self) -> None:
        self._scope_prefix = uuid4().hex
        self._pending_batches: list[_DeferredToolBatch] = []
        tools = self._config.build_tools(self._runtime_tools)
        self._client = Client(
            provider=self._config.provider,
            system_message=self._config.build_system_message(),
            tools=tools,
            before_request=self._prepare_inference,
        )

        self._executor = self._config.build_executor(
            self._client,
            tuple(tool.get_name() for tool in self._runtime_tools),
            exit_content_resolver=self._compose_exit_content,
            execution_manager=self._execution_manager,
            scope_prefix=self._scope_prefix,
            output_preparer=self._output_preparer,
            capability_prompts=self._capability_prompts,
        )
        self._executor.configure_interactive_calls(
            self._turn_control.has_pending_messages
        )

    def _compose_exit_content(
        self,
        content: str,
        items: list[str],
        skill_path: str,
        opening_position: int | None,
    ) -> str:
        return self._exit_items.compose(
            content,
            items,
            skill_path,
            self._run_start_position if opening_position is None else opening_position,
        )

    def _current_tool_choice(self) -> str | None:
        if self.depth > 0:
            return self.REQUIRED_TOOL_CHOICE
        return None

    def handle(self, event: ProviderStreamEvent) -> None:
        if event.type is ProviderStreamEventType.STEERING_ADDED and self.depth > 0:
            assert event.message is not None
            self._result_recorder.persist_steering(
                event.message,
                self._executor.record_steering_output(),
                self._executor.skill_path,
            )
        self._config.emit_stream(
            DeepAgentStreamEvent(
                provider_event=event,
                depth=self.depth,
                skill_path=self._executor.skill_path,
            )
        )

    def _emit_tool_state(
        self,
        name: str,
        call_id: str,
        event_type: DeepAgentExecutionEventType,
        *,
        failed: bool = False,
    ) -> None:
        if not self._config.is_streaming():
            return

        snapshot = self._executor.snapshot()
        self._config.emit_stream(
            DeepAgentExecutionEvent(
                type=event_type,
                failed=failed,
                depth=snapshot.depth,
                skill_path=snapshot.skill_path,
                history=snapshot.history,
                history_tree=snapshot.history_tree,
                tool_name=name,
                tool_call_id=call_id,
            )
        )
