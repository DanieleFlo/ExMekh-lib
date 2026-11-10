from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import Self

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.interactive import (
    InteractiveHandler,
    InteractiveProtocolError,
)
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.message import Message


class ProviderRequest:
    def __init__(
        self,
        messages: list[Message[object]],
        *,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        tools: Tools | None = None,
        stream: bool = False,
        on_event: ProviderEventHandler | None = None,
        reasoning_effort_override: str | None = None,
    ) -> None:
        self._messages = messages
        self._extra = extra
        self._tool_choice = tool_choice
        self._tools = tools
        self.stream = stream
        self.on_event = on_event
        self._reasoning_effort_override = reasoning_effort_override
        self.interactive_handler: InteractiveHandler | None = None
        self._check_provider_inputs()
        self._messages = self._include_tool_attachments(messages)
        self._extra = dict(extra) if extra is not None else None

    @staticmethod
    def _check_messages(messages: list[Message[object]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(MODEL_TEXT.expected_a_non_empty_list_of_messages)
        if not all(isinstance(message, Message) for message in messages):
            raise TypeError(MODEL_TEXT.all_messages_must_be_message_instances)

    def _check_provider_inputs(self) -> None:
        self._check_messages(self._messages)
        if self._extra is not None and not isinstance(self._extra, Mapping):
            raise TypeError(MODEL_TEXT.extra_provider_data_must_be_a_mapping)
        if self._tool_choice is not None and not isinstance(self._tool_choice, str):
            raise TypeError(MODEL_TEXT.tool_choice_must_be_a_string)
        if self._tools is not None and not isinstance(self._tools, Tools):
            raise TypeError(MODEL_TEXT.tools_must_be_a_tools_instance)
        if not isinstance(self.stream, bool):
            raise TypeError(MODEL_TEXT.streaming_must_be_a_boolean)
        if self.on_event is not None:
            ProviderEventHandler.validate_instance(self.on_event)
        if self._reasoning_effort_override is not None and (
            not isinstance(self._reasoning_effort_override, str)
            or not self._reasoning_effort_override.strip()
        ):
            raise ValueError(
                MODEL_TEXT.reasoning_effort_override_must_be_a_non_empty_string
            )

    @staticmethod
    def _include_tool_attachments(
        source: list[Message[object]],
    ) -> list[Message[object]]:
        # Providers receive multimodal companions only after every reply in a batch.
        messages: list[Message[object]] = []
        attachments: list[UserMessage] = []
        for message in source:
            if not isinstance(message, ToolMessage):
                messages.extend(attachments)
                attachments.clear()
            messages.append(message)
            if isinstance(message, ToolMessage):
                companion = message.get_attachments()
                if companion is not None:
                    attachments.append(companion)
        return [*messages, *attachments]

    def get_messages(self) -> list[Message[object]]:
        return self._messages.copy()

    def get_extra(self) -> dict[str, object] | None:
        return dict(self._extra) if self._extra is not None else None

    def get_tool_choice(self) -> str | None:
        return self._tool_choice

    def get_tools(self) -> Tools | None:
        return self._tools

    def get_streaming(self) -> bool:
        return self.stream

    def with_event_handler(self, handler: ProviderEventHandler | None) -> Self:
        """Adapt observation without rebuilding an already expanded conversation."""
        if handler is not None:
            ProviderEventHandler.validate_instance(handler)
        adapted = copy(self)
        adapted.on_event = handler
        return adapted

    def with_messages(self, messages: list[Message[object]]) -> Self:
        """Replace raw history and expand its tool attachments once."""
        self._check_messages(messages)
        adapted = copy(self)
        adapted._messages = self._include_tool_attachments(messages)
        return adapted

    def with_system_message(self, content: str) -> Self:
        """Replace the leading system message without expanding attachments again."""
        if not isinstance(self._messages[0], SystemMessage):
            raise ValueError(MODEL_TEXT.request_must_start_with_system_message)
        adapted = copy(self)
        adapted._messages = [
            SystemMessage(content, name=self._messages[0].get_name()),
            *self._messages[1:],
        ]
        return adapted

    def with_extra(self, extra: Mapping[str, object] | None) -> Self:
        """Copy provider options while preserving expanded messages and handlers."""
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError(MODEL_TEXT.extra_provider_data_must_be_a_mapping)
        adapted = copy(self)
        adapted._extra = dict(extra) if extra is not None else None
        return adapted

    def with_streaming(self, enabled: bool) -> Self:
        """Adapt streaming without rebuilding expanded multimodal messages."""
        if not isinstance(enabled, bool):
            raise TypeError(MODEL_TEXT.streaming_must_be_a_boolean)
        adapted = copy(self)
        adapted.stream = enabled
        return adapted

    def require_interactive_handler(self) -> InteractiveHandler:
        if self.interactive_handler is None:
            raise InteractiveProtocolError(
                MODEL_TEXT.interactive_tools_require_a_node_runtime_handler
            )
        return self.interactive_handler

    def get_reasoning_effort_override(self) -> str | None:
        return self._reasoning_effort_override

    def execute_on_stream_event(self, event: ProviderStreamEvent) -> None:
        if self.on_event is not None:
            self.on_event.handle(event)

    def notify_steering_added(self, messages: Sequence[UserMessage]) -> None:
        """Snapshot the committed batch before invoking any event handler."""
        if self.on_event is None:
            return
        events = tuple(
            ProviderStreamEvent(ProviderStreamEventType.STEERING_ADDED, message=message)
            for message in messages
        )
        for event in events:
            self.execute_on_stream_event(event)
