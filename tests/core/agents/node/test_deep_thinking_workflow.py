"""Full deep-thinking workflow with real tools and a controlled call sequence."""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.web_navigation import crawler, search
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.interfaces.tokenizer import Tokenizer
from tests.tool_result_support import parse_tool_result

STATISTICS_URL = "https://docs.python.org/3/library/statistics.html"
MATH_URL = "https://docs.python.org/3/library/math.html"


class _Tokenizer(Tokenizer):
    def __init__(self, session: SessionCache) -> None:
        self.session = session
        self.next_id = 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        self.next_id += 1
        return f"live-output-{self.next_id}"

    def get_session(self) -> SessionCache:
        return self.session


class _ScriptedProvider(Provider):
    def __init__(
        self,
        session: SessionCache,
        responses: list[tuple[AssistantMessage, str | None]],
    ) -> None:
        super().__init__(_Tokenizer(session))
        self.responses = responses
        self.requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        message, thinking = self.responses.pop(0)
        return ProviderResponse(
            message, ModelThinking(thinking) if thinking is not None else None
        )

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        raise NotImplementedError


def _call(name: str, call_id: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name, arguments))


def _scripted_responses() -> list[tuple[AssistantMessage, str | None]]:
    steps = [
        (_call("deep-thinking", "open"), None),
        (
            _call(
                "search-web",
                "search",
                query="Python statistics mean standard library documentation",
                max_results=5,
            ),
            "Find the official source.",
        ),
        (
            _call("read-web-page", "read", url=STATISTICS_URL),
            "Read the official reference.",
        ),
        (
            _call(
                "interact-web-page",
                "navigate",
                url=STATISTICS_URL,
                actions=[{"kind": "navigate", "url": MATH_URL}],
            ),
            "Check a related page on the same site.",
        ),
        (
            _call(
                "calculate", "calculate", operation="evaluate", expression="(3+5+7)/3"
            ),
            "Compute the sample mean.",
        ),
        (
            _call(
                "logic-consistency",
                "consistency",
                variables={"m": "Real"},
                premises=[{"op": "=", "left": {"var": "m"}, "right": {"const": 5}}],
            ),
            "Check the formal premise is consistent.",
        ),
        (
            _call(
                "logic-check",
                "check",
                variables={"m": "Real"},
                premises=[{"op": "=", "left": {"var": "m"}, "right": {"const": 5}}],
                conclusion={
                    "op": ">",
                    "left": {"var": "m"},
                    "right": {"const": 4},
                },
            ),
            "Check a consequence of the calculated mean.",
        ),
        (
            _call(
                "logic-counterexample",
                "counterexample",
                variables={"m": "Real"},
                premises=[{"op": ">", "left": {"var": "m"}, "right": {"const": 0}}],
                conclusion={
                    "op": ">",
                    "left": {"var": "m"},
                    "right": {"const": 10},
                },
            ),
            "Find a model refuting an overstrong claim.",
        ),
        (_call("exit_agent", "exit", text="The sample mean is 5.", items=[]), None),
    ]
    return [
        (AssistantMessage(None, tool_calls=[tool_call]), thinking)
        for tool_call, thinking in steps
    ] + [(AssistantMessage("The sample mean is 5."), None)]


def _tool_messages(requests: list[ProviderRequest]) -> dict[str, str]:
    return {
        message.get_tool_call_id(): message.get_message()
        for request in requests
        for message in request.get_messages()
        if isinstance(message, ToolMessage)
    }


def _xml_body(content: str) -> dict[str, object]:
    return parse_tool_result(content)


def test_full_skill_uses_every_tool_with_a_mock_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The browser is mocked; URL validation must not depend on external DNS.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    search_web = MagicMock(
        return_value={
            "result_count": 1,
            "results": [{"url": STATISTICS_URL, "title": "Python statistics"}],
        }
    )
    crawl_page = MagicMock(
        side_effect=[
            {"final_url": STATISTICS_URL, "markdown": "Python statistics.mean"},
            {"final_url": MATH_URL, "markdown": "Python math"},
        ]
    )
    monkeypatch.setattr(search, "search_web", search_web)
    monkeypatch.setattr(crawler, "crawl_page", crawl_page)
    with SessionCache(tmp_path / "cache") as session:
        provider = _ScriptedProvider(session, _scripted_responses())
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        response = node.run(
            "Find the official statistics documentation, navigate to the related "
            "math page, calculate the mean of 3, 5, 7, and check formal claims about it."
        )

        assert response.get_message() == "The sample mean is 5."
        assert provider.responses == []
        outputs = _tool_messages(provider.requests)
        assert {
            "search",
            "read",
            "navigate",
            "calculate",
            "consistency",
            "check",
            "counterexample",
            "open",
        } <= outputs.keys()
        assert _xml_body(outputs["search"])["result_count"] > 0
        assert "statistics" in outputs["read"].lower()
        assert MATH_URL in outputs["navigate"]
        assert _xml_body(outputs["calculate"])["result"] == "5.00000000000000"
        assert _xml_body(outputs["consistency"])["status"] == "consistent"
        assert _xml_body(outputs["check"])["status"] == "valid"
        assert _xml_body(outputs["counterexample"])["status"] == "found"
        for call_id in (
            "search",
            "read",
            "navigate",
            "calculate",
            "consistency",
            "check",
            "counterexample",
        ):
            assert "<thinking>" in outputs[call_id]

        search_web.assert_called_once()
        assert search_web.call_args.kwargs["max_results"] == 5
        assert crawl_page.call_count == 2
        assert all(
            call.kwargs["url"] == STATISTICS_URL for call in crawl_page.call_args_list
        )
        assert crawl_page.call_args.kwargs["actions"][0]["url"] == MATH_URL
