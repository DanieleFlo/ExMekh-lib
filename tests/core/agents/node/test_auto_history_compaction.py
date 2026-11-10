import asyncio
from unittest.mock import MagicMock

import pytest

from exmekh_lib.core.agents.deep_agent import DeepAgentHistoryCompactionEvent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.history_compaction import HistoryCompactionPolicy
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.interactive import InterruptedMessage
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.request_budget import (
    ContextBudgetExceededError,
    RequestBudget,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider, ProviderRequestRejectedError
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_history_compaction import runtime_history
from tests.core.history_compaction.support import CompactionProvider, count_request
from tests.event_handler import as_agent_handler
from tests.history_validation import validate_request_history


def automatic_policy(decide=lambda request: True):
    policy = MagicMock(spec=HistoryCompactionPolicy)
    policy.before_inference.side_effect = decide
    return policy


def test_automatic_compaction_prepares_the_next_request_and_emits_archive(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=30
    )
    events = []
    agent = make_agent(
        provider,
        Tool(lambda: "ORIGINAL BODY", name="mock-tool"),
        on_stream=as_agent_handler(events.append),
    )
    policy = automatic_policy()
    agent.root_node.set_history_compaction_policy(policy)
    assert agent.run("task").get_message() == "done"
    request = provider.requests[-1]
    assert "ORIGINAL BODY" not in str(
        [m.message_dump() for m in request.get_messages()]
    )
    event = next(
        event for event in events if isinstance(event, DeepAgentHistoryCompactionEvent)
    )
    assert "ORIGINAL BODY" in str(event.replaced_history)
    assert event.agent_context_id == agent.root_node.agent_context_id
    validate_request_history(request)
    policy.after_compaction.assert_called_once()
    assert policy.after_compaction.call_args.args[0] is request
    assert policy.before_inference.call_count == 1


def test_automatic_policy_is_not_applied_to_internal_compaction_requests(make_agent):
    provider = CompactionProvider([tool_reply(call()), AssistantMessage("done")])
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    observed = []
    agent.root_node.set_history_compaction_policy(
        automatic_policy(lambda request: observed.append(request) or True)
    )
    agent.run("task")
    assert len(provider.score_requests) == 1
    assert all(request.get_reasoning_effort_override() is None for request in observed)
    assert not set(map(id, observed)).intersection(map(id, provider.summary_requests))


def test_automatic_compaction_inside_nested_skills_preserves_native_returns(make_agent):
    provider = CompactionProvider(
        [
            tool_reply(call()),
            tool_reply(call("mock-skill", call_id="open")),
            tool_reply(call("mock-sub-tool", {"text": "parent work"}, "parent")),
            tool_reply(call("nested-skill", call_id="nested-open")),
            tool_reply(call("nested-sub-tool", {"text": "nested work"}, "nested")),
            tool_reply(
                call("exit_agent", {"text": "nested done", "items": []}, "nested-exit")
            ),
            tool_reply(
                call("exit_agent", {"text": "parent done", "items": []}, "parent-exit")
            ),
            AssistantMessage("done"),
        ],
        score=30,
    )
    agent = make_agent(provider, Tool(lambda: "root work", name="mock-tool"))
    agent.root_node.set_history_compaction_policy(automatic_policy())
    response = agent.run("task")
    assert response.get_message() == "done"
    assert agent.root_node.depth == 0
    assert response.skill_executions[0].children[0].output is not None
    assert any(
        (request.get_extra() or {}).get("agentic", {}).get("operation") == "return"
        for request in provider.requests
    )
    for request in provider.requests:
        validate_request_history(request)


def test_rejected_new_input_is_removed_after_prior_history_was_compacted(make_agent):
    provider = CompactionProvider([tool_reply(call()), AssistantMessage("first done")])
    agent = make_agent(provider, Tool(lambda: "old result", name="mock-tool"))
    agent.run("first")
    agent.root_node.set_history_compaction_policy(automatic_policy())
    provider.on_call = lambda request: (_ for _ in ()).throw(
        ProviderRequestRejectedError("rejected")
    )
    with pytest.raises(ProviderRequestRejectedError):
        agent.run("new rejected input")
    history = runtime_history(agent)
    assert any(message.get_message() == "first" for message in history)
    assert not any(message.get_message() == "new rejected input" for message in history)
    assert "compact_history" in str([message.message_dump() for message in history])
    assert agent.root_node.read_tool_output("output-0").content.endswith("old result")


def test_cancellation_during_compaction_preserves_original_history(make_agent):
    provider = CompactionProvider([tool_reply(call()), AssistantMessage("done")])
    provider.summary_error = asyncio.CancelledError()
    agent = make_agent(provider, Tool(lambda: "original body " * 500, name="mock-tool"))
    agent.root_node.set_history_compaction_policy(automatic_policy())
    with pytest.raises(asyncio.CancelledError):
        agent.run("task")
    assert runtime_history(agent)[-1].get_name() == "mock-tool"
    assert agent.root_node.read_tool_output("output-0") is not None
    assert "compact_history" not in str(
        [message.message_dump() for message in runtime_history(agent)]
    )


@pytest.mark.parametrize("enabled", [False, None])
def test_disabling_automatic_policy_keeps_original_requests(make_agent, enabled):
    provider = CompactionProvider([tool_reply(call()), AssistantMessage("done")])
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    agent.root_node.set_history_compaction_policy(
        None if enabled is None else automatic_policy(lambda request: False)
    )
    agent.run("task")
    assert not provider.score_requests


def test_policy_requires_an_initialized_node():
    with pytest.raises(RuntimeError):
        Node(name_hds="main").set_history_compaction_policy(automatic_policy())


def test_failing_compaction_observer_does_not_send_the_stale_request(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=30
    )

    def handle(event):
        if isinstance(event, DeepAgentHistoryCompactionEvent):
            raise RuntimeError("observer failed")

    agent = make_agent(
        provider,
        Tool(lambda: "ORIGINAL BODY", name="mock-tool"),
        on_stream=as_agent_handler(handle),
    )
    agent.root_node.set_history_compaction_policy(automatic_policy())
    assert agent.run("task").get_message() == "done"
    assert "compact_history" in str(
        [m.message_dump() for m in provider.requests[-1].get_messages()]
    )


@pytest.mark.parametrize("phase", ["scoring", "summary"])
def test_failed_automatic_compaction_stops_before_inference_and_preserves_history(
    make_agent, caplog, phase
):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=30
    )
    error = RuntimeError("compaction unavailable")
    if phase == "summary":
        provider.summary_error = error
    else:
        provider.on_score = lambda request: (_ for _ in ()).throw(error)
    agent = make_agent(provider, Tool(lambda: "original body " * 500, name="mock-tool"))
    policy = automatic_policy()
    agent.root_node.set_history_compaction_policy(policy)
    with pytest.raises(
        RuntimeError, match="Automatic history compaction failed"
    ) as failed:
        agent.run("task")
    assert failed.value.__cause__ is error
    assert len(provider.requests) == 1
    assert runtime_history(agent)[-1].get_name() == "mock-tool"
    assert agent.root_node.read_tool_output("output-0") is not None
    assert "compact_history" not in str(
        [message.message_dump() for message in runtime_history(agent)]
    )
    policy.after_compaction.assert_not_called()
    assert "Automatic history compaction failed" in caplog.text
    # Retrying a later turn can compact the retained completed batch.
    provider.summary_error = None
    provider.on_score = None
    assert agent.run("continue").get_message() == "done"
    policy.after_compaction.assert_called_once()


def test_steering_accepted_during_automatic_summary_is_in_the_resumed_request(
    make_agent,
):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=30
    )
    agent = make_agent(provider, Tool(lambda: "original body", name="mock-tool"))
    agent.root_node.set_history_compaction_policy(automatic_policy())
    provider.on_score = lambda request: agent.steer("new constraint")
    assert agent.run("task").get_message() == "done"
    assert len(provider.requests) == 2
    assert provider.requests[-1].get_messages()[-1].get_message() == "new constraint"


def test_return_transition_without_an_assistant_message_keeps_its_history():
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(
        AssistantMessage(None), interruptions=(InterruptedMessage({"phase": "return"}),)
    )
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("task"))
    client.add_tool_calls([call(call_id="call")])
    client.add_tool_response(ToolMessage("result", "call", "mock-tool"))
    before = client.get_history()
    client.complete_tool_transition(
        extra={"agentic": {"operation": "return", "skill_call_id": "call"}}
    )
    assert client.get_history() == before


def test_capacity_forces_compaction_even_when_threshold_policy_is_disarmed(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=9
    )
    body = "evidence line\n" * 300
    agent = make_agent(provider, Tool(lambda: body, name="mock-tool"))
    budget = RequestBudget(11000, count_request)
    provider.on_call = budget.require_fits
    provider.on_score = budget.require_fits
    policy = automatic_policy(lambda request: False)
    agent.root_node.set_history_compaction_policy(policy, budget=budget)
    assert agent.run("user constraint " * 220).get_message() == "done"
    assert len(provider.score_requests) == 1
    assert len(provider.requests) == 2
    assert agent.root_node.read_tool_output("output-0").content.endswith(body)
    assert "compact_history" in str([m.message_dump() for m in runtime_history(agent)])


def test_oversized_protected_input_stops_before_any_inference(make_agent):
    provider = CompactionProvider([AssistantMessage("must not be called")])
    agent = make_agent(provider)
    policy = automatic_policy()
    agent.root_node.set_history_compaction_policy(
        policy, budget=RequestBudget(500, count_request)
    )
    with pytest.raises(ContextBudgetExceededError):
        agent.run("protected user input " * 100)
    assert not provider.requests
    assert not provider.score_requests
    policy.after_compaction.assert_called_once()
    assert any(
        m.get_message() == "protected user input " * 100 for m in runtime_history(agent)
    )


@pytest.mark.parametrize(
    "score,summary", [(100, "unused"), (30, "large summary " * 1000)]
)
def test_larger_automatic_compaction_preserves_fitting_history(
    make_agent, score, summary
):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=score, summary=summary
    )
    events = []
    agent = make_agent(
        provider,
        Tool(lambda: "original body", name="mock-tool"),
        on_stream=as_agent_handler(events.append),
    )
    budget = RequestBudget(8000, count_request)
    policy = automatic_policy()
    provider.on_call = budget.require_fits
    agent.root_node.set_history_compaction_policy(policy, budget=budget)
    assert agent.run("task").get_message() == "done"
    assert len(provider.requests) == 2
    assert len(provider.score_requests) == 1
    assert provider.requests[-1].get_messages()[-1].get_name() == "mock-tool"
    assert "original body" in provider.requests[-1].get_messages()[-1].get_message()
    assert not any(
        isinstance(event, DeepAgentHistoryCompactionEvent) for event in events
    )
    policy.after_compaction.assert_called_once_with(provider.requests[-1])
    assert agent.root_node.read_tool_output("output-0").content.endswith(
        "original body"
    )


@pytest.mark.parametrize("summary", ["large summary " * 1000, "summary\n" * 300])
def test_unusable_compaction_preserves_oversized_originals(make_agent, summary):
    body = "evidence line\n" * 300
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("must not run")],
        score=30,
        summary=summary,
    )
    agent = make_agent(provider, Tool(lambda: body, name="mock-tool"))
    budget = RequestBudget(11000, count_request)
    provider.on_call = budget.require_fits
    provider.on_score = budget.require_fits
    agent.root_node.set_history_compaction_policy(automatic_policy(), budget=budget)
    with pytest.raises(ContextBudgetExceededError):
        agent.run("user constraint " * 220)
    assert len(provider.requests) == len(provider.score_requests) == 1
    assert runtime_history(agent)[-1].get_name() == "mock-tool"
    assert agent.root_node.read_tool_output("output-0").content.endswith(body)
