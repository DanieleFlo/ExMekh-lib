"""Actual compactor inference stays in the ledger after its history disappears."""

from unittest.mock import MagicMock

import pytest

from evaluations.benchmark.accounting import MeteredProvider
from evaluations.benchmark.compaction import ContextCompaction, configure_compaction
from evaluations.benchmark.drivers import run_native
from evaluations.benchmark.models import Criterion, Limits, Task, Variant
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from tests.core.history_compaction.support import CompactionProvider
from tests.evaluations.benchmark.test_accounting import ScriptedProvider, call


class ProviderWithUsage(CompactionProvider):
    def get_tools_text(self, tools):
        return ""

    def call(self, request):
        response = super().call(request)
        response.usage = ModelUsage(10, 5, 15)
        return response

    def call_structured(self, request):
        response = super().call_structured(request)
        response.usage = ModelUsage(10, 5, 15)
        return response


@pytest.mark.parametrize(
    "limits",
    [
        Limits(compact_after_calls=2, max_context_tokens=100_000),
        Limits(compaction_threshold=0.01, max_context_tokens=100_000),
    ],
    ids=["scheduled", "context-threshold"],
)
def test_actual_compaction_keeps_all_internal_calls_in_the_ledger(tmp_path, limits):
    provider = ProviderWithUsage(
        [
            call("file-processing", {}),
            call("run-python", {"path": "answer.py"}),
            call("exit_agent", {"text": "42", "items": []}),
            AssistantMessage("42"),
        ],
        score=30,
    )
    meter = MeteredProvider(provider, limits)
    task = Task(
        id="compact",
        domain="runtime",
        prompt="Run answer.py and return its answer.",
        # The tool output must exceed the overhead of the compacted batch.
        files={"answer.py": "print('intermediate detail ' * 100)\nprint(19 + 23)\n"},
        limits=limits,
        criteria=(
            Criterion(
                id="result",
                description="Correct",
                verification="exact",
                verification_reason="Observable",
                checker="answer",
            ),
        ),
    )
    result = run_native(task, Variant(id="current"), tmp_path, meter)
    assert result.status == "completed"
    assert any(event["kind"] == "history_compaction" for event in result.events)
    totals = meter.get_usage_totals()
    assert totals["compaction_calls"] >= 1
    assert any(record.structured for record in meter.get_inferences())
    assert totals["total_tokens"] == len(meter.get_inferences()) * 15


@pytest.mark.parametrize("size, expected", [(899, False), (900, True), (901, True)])
def test_context_threshold_uses_the_complete_prepared_request(size, expected):
    meter = MeteredProvider(
        ScriptedProvider(),
        Limits(max_context_tokens=1000, max_output_tokens=100),
        Variant(id="candidate", system_addendum="Variant guidance"),
    )
    base = ProviderRequest([SystemMessage("x")])
    padding = 1 + size - meter.estimate_input_tokens(base)
    request = ProviderRequest([SystemMessage("x" * padding)])
    assert meter.estimate_input_tokens(request) == size
    policy = ContextCompaction(meter)
    assert policy.before_inference(request) is expected
    policy.after_compaction(request)
    assert policy.before_inference(request) is expected
    assert meter.get_inferences() == ()
    assert request.get_messages()[0].get_message() == "x" * padding


def test_scheduled_trigger_is_once_but_context_trigger_remains_active():
    meter = MeteredProvider(
        ScriptedProvider(),
        Limits(compact_after_calls=1, max_context_tokens=1000, max_output_tokens=100),
    )
    request = ProviderRequest([SystemMessage("small")])
    policy = ContextCompaction(meter)
    assert not policy.before_inference(request)
    meter.call(request)
    assert policy.before_inference(request)
    policy.after_compaction(request)
    assert not policy.before_inference(request)
    assert policy.before_inference(ProviderRequest([SystemMessage("x" * 1000)]))


def test_compaction_is_installed_on_every_node_and_can_be_forbidden():
    child = MagicMock(children=())
    root = MagicMock(children=(child,))
    meter = MeteredProvider(ScriptedProvider(), Limits())
    configure_compaction(root, meter)
    for node in (root, child):
        node.set_history_compaction_policy.assert_called_once()
        budget = node.set_history_compaction_policy.call_args.kwargs["budget"]
        assert budget.capacity == meter.limits.max_context_tokens
    root.reset_mock()
    configure_compaction(
        root, MeteredProvider(ScriptedProvider(), Limits(compaction="forbid"))
    )
    root.set_history_compaction_policy.assert_not_called()


def test_oversized_protected_input_is_a_budget_failure_without_provider_calls(tmp_path):
    limits = Limits(max_context_tokens=1000, max_output_tokens=100)
    meter = MeteredProvider(ProviderWithUsage(), limits)
    task = Task(
        id="oversized",
        domain="runtime",
        prompt="x" * 10_000,
        limits=limits,
        criteria=(
            Criterion(
                id="answer",
                description="Bounded answer",
                verification="exact",
                verification_reason="Observable",
                checker="answer",
            ),
        ),
    )
    result = run_native(task, Variant(id="current"), tmp_path, meter)
    assert result.status == "budget_exceeded"
    assert "protected" in result.error
    assert meter.get_inferences() == ()


def test_compaction_cannot_hide_call_budget_exhaustion(tmp_path):
    limits = Limits(max_calls=2, compact_after_calls=2, max_context_tokens=100_000)
    meter = MeteredProvider(
        ProviderWithUsage(
            [
                call("deep-thinking", {}),
                call("calculate", {"operation": "evaluate", "expression": "19+23"}),
            ]
        ),
        limits,
    )
    task = Task(
        id="bounded",
        domain="runtime",
        prompt="Compute 19+23",
        limits=limits,
        criteria=(
            Criterion(
                id="answer",
                description="42",
                verification="exact",
                verification_reason="Arithmetic",
                checker="answer",
            ),
        ),
    )
    result = run_native(task, Variant(id="current"), tmp_path, meter)
    assert result.status == "budget_exceeded"
    assert "call budget" in result.error
    assert meter.get_usage_totals()["total_tokens"] == 30
