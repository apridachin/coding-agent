from datetime import datetime, timezone

from unified_llm.types import ToolCallData, Usage

from coding_agent.execution import LocalExecutionEnvironment
from coding_agent.providers import OpenAIProfile
from coding_agent.session import Session
from coding_agent.turns import AssistantTurn


def _tool_call(call_id: str, name: str) -> ToolCallData:
    return ToolCallData(id=call_id, name=name, arguments={})


def test_detect_loop_repeating_pattern(tmp_path):
    env = LocalExecutionEnvironment(str(tmp_path))
    session = Session(OpenAIProfile(), env, llm_client=None)

    calls = []
    for idx in range(10):
        name = "tool_a" if idx % 2 == 0 else "tool_b"
        calls.append(_tool_call(str(idx), name))

    session.history.append(
        AssistantTurn(
            content="",
            tool_calls=calls,
            reasoning=None,
            usage=Usage.zero(),
            response_id=None,
            timestamp=datetime.now(timezone.utc),
        )
    )

    assert session._detect_loop(10) is True


def test_detect_loop_non_repeating(tmp_path):
    env = LocalExecutionEnvironment(str(tmp_path))
    session = Session(OpenAIProfile(), env, llm_client=None)

    calls = []
    for idx in range(10):
        calls.append(_tool_call(str(idx), f"tool_{idx}"))

    session.history.append(
        AssistantTurn(
            content="",
            tool_calls=calls,
            reasoning=None,
            usage=Usage.zero(),
            response_id=None,
            timestamp=datetime.now(timezone.utc),
        )
    )

    assert session._detect_loop(10) is False
