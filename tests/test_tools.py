from datetime import datetime, timezone

from unified_llm.types import ToolResultData

from coding_agent.execution import LocalExecutionEnvironment
from coding_agent.providers import OpenAIProfile
from coding_agent.session import Session
from coding_agent.tools import ToolOutput, tool_read_file
from coding_agent.turns import ToolExecutionResult, ToolResultsTurn


def test_tool_read_file_image(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-an-image")

    env = LocalExecutionEnvironment(str(tmp_path))
    result = tool_read_file({"file_path": "image.png"}, env)
    assert isinstance(result, ToolOutput)
    assert result.image_media_type == "image/png"
    assert result.image_data == b"not-an-image"


def test_tool_result_image_in_message(tmp_path):
    env = LocalExecutionEnvironment(str(tmp_path))
    session = Session(OpenAIProfile(), env, llm_client=None)

    tool_result = ToolExecutionResult(
        tool_call_id="1",
        content="image-content",
        is_error=False,
        image_data=b"123",
        image_media_type="image/png",
    )
    session.history.append(
        ToolResultsTurn(results=[tool_result], timestamp=datetime.now(timezone.utc))
    )

    messages = session._convert_history_to_messages()
    assert len(messages) == 1
    part = messages[0].content[0].tool_result
    assert isinstance(part, ToolResultData)
    assert part.image_media_type == "image/png"
    assert part.image_data == b"123"
