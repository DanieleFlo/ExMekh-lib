from pathlib import Path
from typing import Any

import pytest

from exmekh_lib.core.agents.exit_tool.data import ExitToolData
from exmekh_lib.core.agents.exit_tool.decorator import (
    exit_tool,
    get_exit_tool_definition,
)
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.tools.decorator import deep_tool, get_tool_definition
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool import CalledTool, Tool, ToolExecutionOperation
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.operation_progress import operation_progress_observer
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
)


class ChangeArguments(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        assert context.arguments == {"value": "original"}
        assert len(context.history) == 1
        message = context.history[0]
        assert isinstance(message, UserMessage)
        assert message.get_message() == "call it"
        return ToolArguments({"value": "changed"})


def test_progress_callback_is_runtime_context_and_never_a_tool_argument(tmp_path):
    seen = []

    class Observe(BeforeTool):
        def handle(self, context):
            return ToolArguments(context.arguments, on_progress=seen.append)

    @deep_tool(description="Test", before=Observe(), background_enabled=False)
    def entrypoint(work_root: Path, value: str) -> str:
        callback = operation_progress_observer()
        assert callback is not None
        callback(value)
        return value

    tool = DeepTool(entrypoint, tmp_path)
    assert "on_progress" not in tool.get_signature().parameters
    assert (
        tool.execute_with_hooks({"value": "progress"}, work_root=tmp_path, history=())
        == "progress"
    )
    assert seen == ["progress"]
    assert operation_progress_observer() is None


class ChangeResult(AfterTool):
    def handle(self, context: ToolContext, result: Any) -> str:
        assert context.arguments == {"value": "changed"}
        return f"{result}!"


def test_hooks_change_arguments_and_result(tmp_path: Path) -> None:
    calls: list[str] = []
    before = ChangeArguments()
    after = ChangeResult()

    @deep_tool(description="Test", before=before, after=after)
    def entrypoint(work_root: Path, value: str) -> str:
        assert work_root == tmp_path
        calls.append(value)
        return value

    definition = get_tool_definition(entrypoint)
    assert definition is not None
    assert definition.before is before
    assert definition.after is after
    tool = DeepTool(entrypoint, tmp_path)
    result = tool.execute_with_hooks(
        {"value": "original"},
        work_root=tmp_path,
        history=(UserMessage("call it"),),
    )
    assert result == "changed!"
    assert calls == ["changed"]
    assert tuple(tool.get_signature().parameters) == (
        "value",
        "execution_mode",
        "wait_seconds",
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_nested_hook_arguments_do_not_mutate_the_call_or_after_snapshot(
    tmp_path, replacement
):
    original = {"payload": {"steps": ["original"]}}
    replacement_arguments = {"payload": {"steps": ["replacement"]}}
    selected = ["replacement"] if replacement else ["original", "before"]

    class Before(BeforeTool):
        def handle(self, context):
            if replacement:
                return ToolArguments(replacement_arguments)
            context.arguments["payload"]["steps"].append("before")

    class After(AfterTool):
        def handle(self, context, result):
            assert context.arguments["payload"]["steps"] == selected
            return result

    def work(payload):
        payload["steps"].append("work")
        return payload["steps"]

    tool = Tool(work, before=Before(), after=After())
    called = CalledTool(tool, ToolCall("call", FunctionCall("work", original)))
    result = called.execute(work_root=tmp_path)
    assert result == [*selected, "work"]
    assert original == {"payload": {"steps": ["original"]}}
    assert replacement_arguments == {"payload": {"steps": ["replacement"]}}
    assert called.executed_arguments == {"payload": {"steps": selected}}


def test_deferred_execution_uses_arguments_from_handle_creation(tmp_path):
    arguments = {"payload": {"steps": ["original"]}}
    tool = Tool(lambda payload: payload["steps"], name="work")
    operation = tool.create_execution_with_hooks(
        arguments, call_id="call", work_root=tmp_path, history=()
    )
    arguments["payload"]["steps"].append("changed while pending")
    assert operation.run() == ["original"]
    assert operation.get_executed_arguments() == {"payload": {"steps": ["original"]}}


def test_before_can_answer_without_calling_tool(tmp_path: Path) -> None:
    class AnswerEarly(BeforeTool):
        def handle(self, context: ToolContext) -> ToolResponse:
            assert context.work_root == tmp_path
            return ToolResponse(None)

    class FormatAnswer(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            assert result is None
            return "handled"

    def unreachable(value: str) -> str:
        pytest.fail("tool should not run")

    tool = Tool(unreachable, before=AnswerEarly(), after=FormatAnswer())
    assert (
        tool.execute_with_hooks({"value": "x"}, work_root=tmp_path, history=())
        == "handled"
    )


def test_called_tool_applies_hooks_without_explicit_context() -> None:
    class PrefixResult(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            assert context.work_root == Path.cwd()
            return f"hooked: {result}"

    tool = Tool(lambda value: value, name="original", after=PrefixResult())
    renamed = tool.with_name("renamed")
    called = CalledTool(
        renamed, ToolCall("call-1", FunctionCall("renamed", {"value": "ok"}))
    )

    assert renamed.after is tool.after
    assert called.execute() == "hooked: ok"


def test_invalid_final_arguments_are_rejected_before_tool_runs(tmp_path: Path) -> None:
    calls: list[str] = []

    def entrypoint(value: str) -> str:
        calls.append(value)
        return value

    tool = Tool(entrypoint)
    with pytest.raises(TypeError, match="missing a required argument"):
        tool.execute_with_hooks({}, work_root=tmp_path, history=())
    assert calls == []


def test_rejects_invalid_hooks_at_registration() -> None:
    with pytest.raises(TypeError, match="before must be a BeforeTool"):
        deep_tool(before=object())

    with pytest.raises(TypeError, match="after must be an AfterTool"):
        deep_tool(after=object())

    class WrongSignature(BeforeTool):
        def handle(self) -> None:
            pass

    with pytest.raises(TypeError, match="must accept 1 argument"):
        deep_tool(before=WrongSignature())

    class AsyncAfter(AfterTool):
        async def handle(self, context: ToolContext, result: Any) -> Any:
            return result

    with pytest.raises(TypeError, match="must be synchronous"):
        deep_tool(after=AsyncAfter())


@pytest.mark.parametrize("managed", [False, True])
def test_invalid_before_hook_response_prevents_execution(tmp_path, managed) -> None:
    class InvalidResponse(BeforeTool):
        def handle(self, context: ToolContext):
            return "unsupported response"

    def work() -> None:
        pytest.fail("An invalid hook response must stop the tool")

    tool = Tool(work, before=InvalidResponse())
    with pytest.raises(TypeError, match="BeforeTool.handle must return"):
        if managed:
            operation = tool.create_execution_with_hooks(
                {}, call_id="call", work_root=tmp_path, history=()
            )
            operation.run()
        else:
            tool.execute_with_hooks({}, work_root=tmp_path, history=())


def test_managed_before_hook_can_answer_without_creating_work(tmp_path) -> None:
    after_results = []

    class AnswerEarly(BeforeTool):
        def handle(self, context: ToolContext) -> ToolResponse:
            return ToolResponse("shortcut")

    class FormatAnswer(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            after_results.append(result)
            return f"public: {result}"

    def work() -> None:
        pytest.fail("The before hook already answered")

    operation = Tool(
        work, before=AnswerEarly(), after=FormatAnswer()
    ).create_execution_with_hooks({}, call_id="call", work_root=tmp_path, history=())

    assert operation.run() == "public: shortcut"
    assert after_results == ["shortcut"]
    assert operation.cancel is None and operation.read_partial is None


def test_exit_hooks_can_fix_arguments_and_change_output(tmp_path: Path) -> None:
    class FixArguments(BeforeTool):
        def handle(self, context: ToolContext) -> ToolArguments:
            return ToolArguments({"text": "finished", "items": []})

    class ChangeOutput(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> dict[str, Any]:
            return {**result, "text": "changed"}

    @exit_tool(
        instructions_path="instructions.md", before=FixArguments(), after=ChangeOutput()
    )
    def finish(text: str, items: list[str]) -> ExitToolData:
        return ExitToolData(text, items)

    definition = get_exit_tool_definition(finish)
    assert definition is not None
    tool = ExitTool(
        finish,
        name="finish",
        description="Finish",
        instructions_path=tmp_path / "instructions.md",
        instructions="Finish the task.",
        folder_path=tmp_path,
        before=definition.before,
        after=definition.after,
    )
    assert tool.execute_with_hooks(
        {"text": "", "items": []}, work_root=tmp_path, history=()
    ) == {"text": "changed", "items": []}


@pytest.mark.parametrize("managed", [False, True])
def test_custom_execution_uses_the_same_hooks_and_context_in_both_modes(
    tmp_path, managed
):
    steps = []

    class ChangeInput(BeforeTool):
        def handle(self, context: ToolContext) -> ToolArguments:
            steps.append("before")
            return ToolArguments({"value": "changed"})

    class ChangeOutput(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            steps.append("after")
            assert context.arguments == {"value": "changed"}
            return result + "!"

    class CustomTool(Tool):
        def create_execution(self, arguments, *, call_id, work_root, history):
            assert call_id == "call"
            assert work_root == tmp_path
            assert len(history) == 1
            assert isinstance(history[0], UserMessage)
            assert history[0].get_message() == "start"
            steps.append("create")

            def run():
                steps.append("run")
                return arguments["value"]

            return ToolExecutionOperation(run)

    def unreachable(value):
        pytest.fail("The custom execution operation must run")

    called = CalledTool(
        CustomTool(
            unreachable, name="custom", before=ChangeInput(), after=ChangeOutput()
        ),
        ToolCall("call", FunctionCall("custom", {"value": "original"})),
    )
    context = {
        "work_root": tmp_path,
        "history": (UserMessage("start"),),
    }
    result = (
        called.create_execution(**context).run()
        if managed
        else called.execute(**context)
    )

    assert result == "changed!"
    assert steps == ["before", "create", "run", "after"]


def test_deferred_execution_keeps_a_private_typed_history_snapshot(tmp_path):
    user = UserMessage(["start"])
    call = ToolCall("inspect", FunctionCall("inspect-file", {"path": "notes.txt"}))
    assistant = AssistantMessage(None, tool_calls=[call])
    attachments = UserMessage(["attachment"])
    reply = ToolMessage("report", "inspect", "inspect-file", attachments=attachments)
    reply.set_metadata("inspection", {"digest": b"original"})
    seen = []

    class InspectHistory(BeforeTool):
        def handle(self, context: ToolContext) -> None:
            saved_user, saved_assistant, saved_reply = context.history
            assert isinstance(saved_user, UserMessage)
            assert saved_user.get_message() == ["start"]
            assert isinstance(saved_assistant, AssistantMessage)
            saved_call = saved_assistant.get_tool_calls()[0]
            assert saved_call.get_id() == "inspect"
            assert saved_call.get_arguments() == {"path": "notes.txt"}
            assert isinstance(saved_reply, ToolMessage)
            assert saved_reply.get_tool_call_id() == "inspect"
            assert saved_reply.get_metadata() == {"inspection": {"digest": b"original"}}
            saved_attachments = saved_reply.get_attachments()
            assert saved_attachments.get_message() == ["attachment"]
            saved_attachments.get_message().append("hook change")
            saved_call.get_arguments()["path"] = "hook.txt"
            saved_reply.set_metadata("inspection", {"digest": b"hook"})
            seen.append(context.history)

    class CheckHistory(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> Any:
            assert context.history is seen[0]
            assert context.history[-1].get_metadata()["inspection"]["digest"] == b"hook"
            return result

    operation = Tool(
        lambda: "done", name="read", before=InspectHistory(), after=CheckHistory()
    ).create_execution_with_hooks(
        {}, call_id="read", work_root=tmp_path, history=(user, assistant, reply)
    )
    # A delayed worker must see the conversation captured before it starts.
    user.get_message().append("later")
    call.get_arguments()["path"] = "later.txt"
    attachments.get_message().append("later")
    reply.set_metadata("inspection", {"digest": b"later"})

    assert operation.run() == "done"
    assert user.get_message() == ["start", "later"]
    assert call.get_arguments() == {"path": "later.txt"}
    assert attachments.get_message() == ["attachment", "later"]
    assert reply.get_metadata() == {"inspection": {"digest": b"later"}}
