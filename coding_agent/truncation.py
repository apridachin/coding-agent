from __future__ import annotations

from typing import Dict, Optional

DEFAULT_TOOL_LIMITS: Dict[str, int] = {
    "read_file": 50_000,
    "shell": 30_000,
    "grep": 20_000,
    "glob": 20_000,
    "edit_file": 10_000,
    "apply_patch": 10_000,
    "write_file": 1_000,
    "spawn_agent": 20_000,
}

DEFAULT_TRUNCATION_MODES: Dict[str, str] = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "apply_patch": "tail",
    "write_file": "tail",
    "spawn_agent": "head_tail",
}

DEFAULT_LINE_LIMITS: Dict[str, Optional[int]] = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
    "read_file": None,
    "edit_file": None,
}


def truncate_output(output: str, max_chars: int, mode: str) -> str:
    if len(output) <= max_chars:
        return output

    if mode == "head_tail":
        half = max_chars // 2
        removed = len(output) - max_chars
        return (
            output[:half]
            + "\n\n[WARNING: Tool output was truncated. "
            + str(removed)
            + " characters were removed from the middle. "
            + "The full output is available in the event stream. "
            + "If you need to see specific parts, re-run the tool with more targeted parameters.]\n\n"
            + output[-half:]
        )

    if mode == "tail":
        removed = len(output) - max_chars
        return (
            "[WARNING: Tool output was truncated. First "
            + str(removed)
            + " characters were removed. "
            + "The full output is available in the event stream.]\n\n"
            + output[-max_chars:]
        )

    return output[-max_chars:]


def truncate_lines(output: str, max_lines: int) -> str:
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count
    return (
        "\n".join(lines[:head_count])
        + "\n[... "
        + str(omitted)
        + " lines omitted ...]\n"
        + "\n".join(lines[-tail_count:])
    )


def truncate_tool_output(
    output: str,
    tool_name: str,
    tool_output_limits: Dict[str, int],
    tool_line_limits: Dict[str, Optional[int]],
) -> str:
    max_chars = tool_output_limits.get(tool_name, DEFAULT_TOOL_LIMITS.get(tool_name, 20_000))
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, "tail")

    result = truncate_output(output, max_chars, mode)
    max_lines = tool_line_limits.get(tool_name, DEFAULT_LINE_LIMITS.get(tool_name))
    if max_lines is not None:
        result = truncate_lines(result, max_lines)
    return result
