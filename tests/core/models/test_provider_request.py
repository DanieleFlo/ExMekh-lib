from types import MappingProxyType
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from tests.event_handler import as_provider_handler


def test_get_messages_returns_messages_in_order() -> None:
    messages = [
        SystemMessage("non-default-rules"),
        UserMessage("non-default-question"),
        AssistantMessage("non-default-answer"),
    ]
    request = ProviderRequest(messages)

    result = request.get_messages()

    assert result == messages


def test_get_messages_is_isolated_from_external_list_mutation() -> None:
    system_message = SystemMessage("stable-rules")
    source = [system_message]
    request = ProviderRequest(source)

    source.append(UserMessage("source mutation"))
    result = request.get_messages()
    result.append(UserMessage("attempted external mutation"))

    assert request.get_messages() == [system_message]


def test_request_preserves_provider_options() -> None:
    def search(query: str) -> str:
        return query

    tools = Tools([Tool(search)])
    extra = {"temperature": 0.2}
    request = ProviderRequest(
        [SystemMessage("rules")],
        extra=extra,
        tool_choice="auto",
        tools=tools,
    )

    extra["temperature"] = 1
    returned_extra = request.get_extra()
    assert returned_extra is not None
    returned_extra["temperature"] = 2

    assert request.get_extra() == {"temperature": 0.2}
    assert request.get_tool_choice() == "auto"
    assert request.get_tools() is tools


def test_request_preserves_streaming_and_executes_callback() -> None:
    events = []
    request = ProviderRequest(
        [SystemMessage("rules")],
        stream=True,
        on_event=as_provider_handler(events.append),
    )
    event = ProviderStreamEvent(ProviderStreamEventType.START)

    request.execute_on_stream_event(event)

    assert request.get_streaming() is True
    assert events == [event]


def test_request_copies_extra_from_a_read_only_mapping() -> None:
    source = {"temperature": 0.2}
    request = ProviderRequest([SystemMessage("rules")], extra=MappingProxyType(source))
    source["temperature"] = 1

    assert request.get_extra() == {"temperature": 0.2}


def test_execute_on_stream_event_without_callback_is_a_noop() -> None:
    request = ProviderRequest([SystemMessage("rules")])
    event = ProviderStreamEvent(ProviderStreamEventType.START)

    assert request.execute_on_stream_event(event) is None


@pytest.mark.parametrize("messages", [None, [], [object()]])
def test_request_rejects_invalid_messages(messages) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderRequest(messages)


def test_request_rejects_invalid_options() -> None:
    messages = [SystemMessage("rules")]

    with pytest.raises(TypeError, match="mapping"):
        ProviderRequest(messages, extra="invalid")

    with pytest.raises(TypeError, match="Tool choice"):
        ProviderRequest(messages, tool_choice=object())

    with pytest.raises(TypeError, match="Tools instance"):
        ProviderRequest(messages, tools=object())

    with pytest.raises(TypeError, match="boolean"):
        ProviderRequest(messages, stream=1)

    with pytest.raises(TypeError, match="ProviderEventHandler"):
        ProviderRequest(messages, on_event=object())

    with pytest.raises(ValueError, match="Reasoning effort override"):
        ProviderRequest(messages, reasoning_effort_override="  ")


@pytest.mark.parametrize("structured", [False, True])
def test_event_adapter_preserves_conversation_options_and_schema(structured):
    class Answer(BaseModel):
        text: str

    events = []
    original_events = []
    attachment = UserMessage("attachment")
    messages = [
        SystemMessage("rules"),
        ToolMessage("body", "call", "work", attachments=attachment),
    ]
    options = dict(
        extra={"temperature": 0.2},
        tool_choice="none",
        stream=True,
        reasoning_effort_override="none",
        on_event=as_provider_handler(original_events.append),
    )
    request = (
        StructuredProviderRequest(messages, Answer, strict=False, **options)
        if structured
        else ProviderRequest(messages, **options)
    )
    request.interactive_handler = Mock()
    adapted = request.with_event_handler(as_provider_handler(events.append))
    assert type(adapted) is type(request)
    assert adapted is not request
    assert adapted.get_messages() == request.get_messages()
    assert adapted.get_messages().count(attachment) == 1
    assert adapted.get_extra() == request.get_extra()
    assert adapted.get_tool_choice() == "none"
    assert adapted.get_streaming()
    assert adapted.get_reasoning_effort_override() == "none"
    assert adapted.require_interactive_handler() is request.interactive_handler
    if structured:
        assert adapted.get_response_model() is Answer
        assert not adapted.get_strict()
    event = ProviderStreamEvent(ProviderStreamEventType.START)
    adapted.execute_on_stream_event(event)
    assert events == [event]
    assert not original_events
    assert adapted.with_event_handler(None).on_event is None
    with pytest.raises(TypeError, match="ProviderEventHandler"):
        request.with_event_handler(object())


@pytest.mark.parametrize("structured", [False, True])
def test_request_adapters_preserve_expanded_attachments_and_contracts(structured):
    class Answer(BaseModel):
        text: str

    attachment = UserMessage("attachment")
    messages = [
        SystemMessage("old", name="policy"),
        ToolMessage("body", "id", "tool", attachments=attachment),
    ]
    request = (
        StructuredProviderRequest(messages, Answer, strict=False)
        if structured
        else ProviderRequest(messages)
    )
    request.interactive_handler = Mock()
    adapted = (
        request.with_messages(messages)
        .with_system_message("new")
        .with_extra({"max_output_tokens": 10})
        .with_streaming(True)
    )
    assert adapted.get_messages().count(attachment) == 1
    assert adapted.get_messages()[0].get_message() == "new"
    assert adapted.get_messages()[0].get_name() == "policy"
    assert request.get_messages()[0].get_message() == "old"
    assert request.get_extra() is None
    assert not request.get_streaming()
    assert adapted.require_interactive_handler() is request.interactive_handler
    assert type(adapted) is type(request)
    if structured:
        assert adapted.get_response_model() is Answer
        assert not adapted.get_strict()


def test_request_adapters_validate_options():
    request = ProviderRequest([UserMessage("hello")])
    with pytest.raises(ValueError, match="system message"):
        request.with_system_message("rules")
    with pytest.raises(TypeError, match="boolean"):
        request.with_streaming(1)
    with pytest.raises(TypeError, match="mapping"):
        request.with_extra("invalid")
