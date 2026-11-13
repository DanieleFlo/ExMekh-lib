"""Check decision grading and fair prompt comparisons without a model."""

import json
import re
from dataclasses import asdict, replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from evaluations.harness_workflows import (
    BASELINE,
    Trial,
    WorkflowSettings,
    build_payload,
    execute_trial,
    regrade,
    run,
    summarize,
    variant_bundles,
    write_report,
)
from evaluations.tool_argument_grading import RawToolCall
from evaluations.workflow_cases import WorkflowGrade, build_cases, grade_workflow
from evaluations.workflow_prompts import capture_prompts, load_bundle, save_bundle
from exmekh_lib.core.models.json_schema import without_schema_descriptions
from tests.tokenizer_support import StubTokenizer

CASES = {case.name: case for case in build_cases()}


def gold_call(case):
    arguments = dict(case.arguments)
    if case.next_tools[0] == "exit_agent":
        arguments.update(
            text="Completed. " + " ".join(case.contains), items=list(case.items or ())
        )
    elif case.next_tools[0] == "call_sub_agent":
        arguments["text"] = "Reverse AZ-041"
    elif case.next_tools[0] == "new-file":
        arguments["document"] = (
            '<new-exmekh-dir path="missing"/><new-exmekh-file path="missing/next.txt"/>NEXT'
        )
    elif case.category == "mathematics":
        arguments["expression"] = "x**3 + 2*x"
    elif case.category == "date_time":
        arguments.update(
            value="2027-02-14T23:15:00-03:00", other="2027-02-16T02:45:00+09:00"
        )
    elif case.category == "multimodal":
        arguments.update(start_index=0, end_index=0)
    return RawToolCall(case.next_tools[0], json.dumps(arguments))


@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case.name)
def test_known_correct_next_actions_receive_full_credit(case):
    calls = (gold_call(case),) if case.next_tools else ()
    assert grade_workflow(case, calls, " ".join(case.contains)).score == 100


@pytest.mark.parametrize(
    "raw", ["[]", '{"path":"a","path":"b"}', '{"value":NaN}', '{"value":1e400}', "{"]
)
def test_malformed_raw_calls_are_not_repaired(raw):
    case = CASES["inspect_before_read"]
    assert grade_workflow(case, (RawToolCall("inspect-file", raw),), "").score == 0


def test_completed_task_reopening_is_not_fully_correct_even_with_answer_text():
    case = CASES["finish_after_skill_return"]
    grade = grade_workflow(case, (RawToolCall("file-processing", "{}"),), "AZ-041")
    assert grade.checks["answers_without_more_work"] is False
    assert grade.score < 100


def test_background_job_uses_original_call_id_not_output_id():
    case = CASES["wait_original_job"]
    call = RawToolCall(
        "control_tool_execution", '{"tool_call_id":"job-output","action":"wait"}'
    )
    assert grade_workflow(case, (call,), "").checks["argument:tool_call_id"] is False


def test_boolean_line_number_does_not_count_as_integer():
    case = CASES["continue_after_auto_inspection"]
    call = RawToolCall(
        "read-file", '{"path":"notes.md","start_index":true,"end_index":3}'
    )
    assert grade_workflow(case, (call,), "").checks["argument:start_index"] is False


def test_exit_aliases_are_accepted_but_duplicate_aliases_are_not():
    case = CASES["full_details_for_caller"]
    valid = RawToolCall(
        "exit_agent", '{"text":"Rows: {{evidence-17}}","items":["step-2"]}'
    )
    assert grade_workflow(case, (valid,), "").score == 100
    duplicate = RawToolCall(
        "exit_agent", '{"text":"Rows","items":["step-2","evidence-17"]}'
    )
    assert grade_workflow(case, (duplicate,), "").checks["selected_outputs"] is False


def test_complete_inline_evidence_is_equivalent_to_promotion_but_summary_is_not():
    case = CASES["full_details_for_caller"]
    full = case.steps[-1].output.split("</header>\n", 1)[1]
    call = RawToolCall("exit_agent", json.dumps({"text": full, "items": []}))
    assert grade_workflow(case, (call,), "").score == 100
    summary = RawToolCall(
        "exit_agent", '{"text":"36 calibration rows read","items":[]}'
    )
    assert grade_workflow(case, (summary,), "").score < 100


def test_saved_responses_can_be_regraded_without_inference_or_changing_trace(tmp_path):
    case = CASES["summary_handoff"]
    call = gold_call(case)
    trial = Trial(
        case.name,
        case.category,
        "baseline",
        1,
        WorkflowGrade(0, {"old": False}),
        1,
        100,
    )
    record = {
        "case": asdict(case),
        "trial": asdict(trial),
        "request": {"model": "frozen-model", "max_output_tokens": 1024},
        "response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": call.name,
                    "arguments": call.arguments,
                }
            ],
        },
    }
    trace = tmp_path / "original.jsonl"
    source = json.dumps(record) + "\n"
    trace.write_text(source)
    settings = WorkflowSettings("unused-model", tmp_path / "reviewed.xml", tuple(CASES))
    reviewed = regrade(settings, trace)
    assert reviewed[0].grade.score == 100
    assert trace.read_text() == source
    root = ET.fromstring(settings.output.read_text())
    assert root.attrib["grading-version"] == "2"
    assert root.findtext("source_trace") == str(trace)
    assert root.findtext("settings/model") == "frozen-model"
    assert root.findtext("complete") == "true"


@pytest.mark.parametrize("text", ["Result {{missing}}", "Result {{broken", "Result }}"])
def test_invalid_exit_placeholders_do_not_receive_full_credit(text):
    case = CASES["repair_exit_only"]
    call = RawToolCall("exit_agent", json.dumps({"text": f"42 {text}", "items": []}))
    assert grade_workflow(case, (call,), "").score < 100


def test_deferred_evidence_should_not_be_promoted_into_caller_context():
    case = CASES["reference_for_later_skill"]
    call = RawToolCall(
        "exit_agent", '{"text":"Saved evidence-17","items":["evidence-17"]}'
    )
    assert grade_workflow(case, (call,), "").checks["selected_outputs"] is False


def test_creation_grade_uses_protocol_semantics_and_rejects_repeating_success():
    case = CASES["retry_only_failed_creation"]
    document = '<new-exmekh-dir path="missing" />\n<new-exmekh-file path="missing/next.txt" />NEXT'
    call = RawToolCall("new-file", json.dumps({"document": document}))
    assert grade_workflow(case, (call,), "").score == 100
    repeated = RawToolCall(
        "new-file",
        json.dumps({"document": '<new-exmekh-file path="done.txt"/>OK' + document}),
    )
    assert grade_workflow(case, (repeated,), "").score < 100


def test_prompt_capture_and_snapshots_do_not_require_endpoint_or_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    current = capture_prompts(tmp_path)
    destination = tmp_path / "snapshot.xml"
    save_bundle(current, destination)
    assert load_bundle(destination) == current
    with pytest.raises(FileExistsError):
        save_bundle(current, destination)


@pytest.mark.parametrize(
    "suffix", [".xml", ".jsonl", ".prompts.xml", ".schema.prompts.xml"]
)
def test_existing_evidence_is_rejected_before_contacting_provider(tmp_path, suffix):
    report = tmp_path / "report.xml"
    existing = report.with_suffix(suffix)
    existing.write_text("Preserve previous evidence")
    settings = WorkflowSettings("model", report, tuple(CASES))
    with pytest.raises(FileExistsError):
        run(settings)
    assert existing.read_text() == "Preserve previous evidence"


def test_variants_preserve_cases_schema_contracts_and_use_real_skill_instructions(
    tmp_path,
):
    current = capture_prompts(tmp_path)
    bundles = variant_bundles(current)
    baseline = load_bundle(BASELINE)
    assert bundles["workflow"].tools == baseline.tools
    assert bundles["workflow"].skills == baseline.skills
    assert bundles["workflow"].system == current.system
    old = {
        tool["name"]: without_schema_descriptions(tool["parameters"])
        for tool in baseline.tools
    }
    new = {
        tool["name"]: without_schema_descriptions(tool["parameters"])
        for tool in current.tools
    }
    # The frozen prompt snapshot predates explicit text read-plan pagination.
    inspection = new["inspect-file"]
    assert inspection["properties"].pop("read_from_line") == {
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }
    inspection["required"].remove("read_from_line")
    assert old == new
    settings = WorkflowSettings("model", tmp_path / "report.xml", tuple(CASES))
    for case in build_cases():
        payloads = [
            build_payload(case, bundle, settings) for bundle in bundles.values()
        ]
        assert all(
            payload["input"][1] == payloads[0]["input"][1] for payload in payloads
        )
        for payload in payloads:
            assert "extra_body" not in payload
            assert payload["tool_choice"] == "auto"
            calls = [
                item["call_id"]
                for item in payload["input"]
                if item.get("type") == "function_call"
            ]
            results = [
                item["call_id"]
                for item in payload["input"]
                if item.get("type") == "function_call_output"
            ]
            assert calls == results
            assert len(calls) == len(set(calls))


def test_parameter_location_comparison_changes_only_documentation_placement(tmp_path):
    scoped = capture_prompts(tmp_path, parameter_descriptions_in_schema=False)
    schema = capture_prompts(tmp_path, parameter_descriptions_in_schema=True)
    assert scoped.system == schema.system
    assert scoped.tools != schema.tools
    assert scoped.skills != schema.skills
    for left, right in zip(scoped.tools, schema.tools, strict=True):
        assert left["name"] == right["name"]
        assert left.get("description") == right.get("description")
        assert without_schema_descriptions(
            left["parameters"]
        ) == without_schema_descriptions(right["parameters"])
    assert "parameter-descriptions" in scoped.skills["file-processing"]
    assert "parameter-descriptions" not in schema.skills["file-processing"]
    for name, instructions in scoped.skills.items():
        without_catalog_parameters = re.sub(
            r"\n[ \t]*<parameter-descriptions\b[^>]*>.*?</parameter-descriptions>",
            "",
            instructions,
            flags=re.DOTALL,
        )
        assert without_catalog_parameters == schema.skills[name]
    bundles = variant_bundles(scoped, schema)
    assert bundles["compact"] == scoped
    assert bundles["schema"] == schema
    first = Trial(
        "one", "entry", "compact", 1, WorkflowGrade(0, {"action": False}), 1, 100
    )
    second = replace(
        first, variant="schema", grade=WorkflowGrade(100, {"action": True})
    )
    assert summarize([first, second])["parameter_location_comparison"] == {
        "pairs": 1,
        "mean_schema_minus_skill": 100,
    }


def test_report_does_not_score_infrastructure_failures_or_compare_unpaired_cases(
    tmp_path,
):
    settings = WorkflowSettings("model", tmp_path / "report.xml", tuple(CASES))
    good = Trial(
        "one", "entry", "baseline", 1, WorkflowGrade(100, {"action": True}), 1, 100
    )
    bad = replace(good, variant="compact", grade=None, infrastructure_error="offline")
    other = replace(
        good, case="two", variant="compact", grade=WorkflowGrade(0, {"action": False})
    )
    summary = summarize([good, bad, other])
    assert summary["compact"]["scored"] == 1
    assert summary["compact"]["infrastructure_errors"] == 1
    assert summary["paired_comparison"]["compact"]["pairs"] == 0
    write_report(settings, [good, bad, other], complete=False)
    root = ET.fromstring(settings.output.read_text())
    assert root.findtext("complete") == "false"
    assert root.findtext("planned_samples") == "60"
    assert len(root.findall("cases/item")) == len(CASES)


def test_dataset_separates_varied_domains_and_unseen_decision_categories():
    cases = build_cases()
    assert len(cases) == len({case.name for case in cases}) == 20
    assert len({case.domain for case in cases}) == 20
    assert sum(case.split == "holdout" for case in cases) == 4
    development = {case.category for case in cases if case.split == "development"}
    holdout = {case.category for case in cases if case.split == "holdout"}
    assert development.isdisjoint(holdout)


def test_semantic_checks_accept_equivalent_expressions_and_timezone_offsets():
    math_case = CASES["symbolic_derivative"]
    valid = RawToolCall(
        "calculate", '{"operation":"differentiate","expression":"x*(x**2+2)"}'
    )
    assert grade_workflow(math_case, (valid,), "").score == 100
    wrong = RawToolCall(
        "calculate", '{"operation":"differentiate","expression":"x**2"}'
    )
    assert grade_workflow(math_case, (wrong,), "").score < 100
    time_case = CASES["elapsed_time_across_offsets"]
    valid = RawToolCall(
        "date-time",
        '{"operation":"difference","value":"2027-02-15T02:15:00Z","other":"2027-02-15T17:45:00Z","unit":"hours"}',
    )
    assert grade_workflow(time_case, (valid,), "").score == 100
    wrong = RawToolCall(
        "date-time",
        '{"operation":"difference","value":"2027-02-15","other":"2027-02-16","unit":"hours"}',
    )
    assert grade_workflow(time_case, (wrong,), "").score < 100


def test_incomplete_model_response_scores_zero_instead_of_using_partial_calls(tmp_path):
    settings = WorkflowSettings("model", tmp_path / "report.xml", tuple(CASES))
    response = SimpleNamespace(
        status="incomplete",
        output=[],
        output_text="AZ-041",
        model_dump=lambda **kwargs: {"status": "incomplete"},
    )
    provider = SimpleNamespace(
        client=SimpleNamespace(
            with_options=lambda **kwargs: SimpleNamespace(
                responses=SimpleNamespace(create=lambda **kwargs: response)
            )
        ),
        get_tokenizer=lambda: StubTokenizer(),
    )
    trial, trace = execute_trial(
        provider,
        settings,
        CASES["finish_after_skill_return"],
        load_bundle(BASELINE),
        "baseline",
        1,
    )
    assert trial.grade.score == 0
    assert trial.infrastructure_error is None
    assert trace["response"]["status"] == "incomplete"
