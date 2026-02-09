from coding_agent.truncation import truncate_lines, truncate_output, truncate_tool_output


def test_truncate_output_head_tail():
    output = "abcdefghijklmnopqrstuvwxyz"
    truncated = truncate_output(output, 10, "head_tail")
    assert "Tool output was truncated" in truncated
    assert truncated.startswith("abcde")
    assert truncated.endswith("vwxyz")


def test_truncate_output_tail():
    output = "abcdefghijklmnopqrstuvwxyz"
    truncated = truncate_output(output, 8, "tail")
    assert "Tool output was truncated" in truncated
    assert truncated.endswith("stuvwxyz"[-8:])


def test_truncate_lines():
    output = "1\n2\n3\n4\n5"
    truncated = truncate_lines(output, 4)
    assert "lines omitted" in truncated
    assert truncated.splitlines()[0] == "1"
    assert truncated.splitlines()[1] == "2"
    assert truncated.splitlines()[-2:] == ["4", "5"]


def test_truncate_tool_output():
    output = "x" * 200
    truncated = truncate_tool_output(
        output,
        "shell",
        {"shell": 50},
        {},
    )
    assert "Tool output was truncated" in truncated
    assert truncated.startswith("x" * 25)
    assert truncated.endswith("x" * 25)
