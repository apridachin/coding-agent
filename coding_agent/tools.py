from __future__ import annotations

import html
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from unified_llm.types import ToolDefinition

from .execution import ExecutionEnvironment
from .utils import normalize_line


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    executor: Callable[[Dict[str, Any], ExecutionEnvironment], Any]


@dataclass
class ToolOutput:
    content: str
    image_data: Optional[bytes] = None
    image_media_type: Optional[str] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def definitions(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def clone(self) -> "ToolRegistry":
        clone = ToolRegistry()
        for tool in self._tools.values():
            clone.register(tool)
        return clone


# Tool definitions


def tool_definition_read_file() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file from the filesystem. Returns line-numbered content.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    )


def tool_definition_write_file() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Write content to a file. Creates the file and parent directories if needed.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    )


def tool_definition_edit_file() -> ToolDefinition:
    return ToolDefinition(
        name="edit_file",
        description="Replace an exact string occurrence in a file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    )


def tool_definition_apply_patch() -> ToolDefinition:
    return ToolDefinition(
        name="apply_patch",
        description=(
            "Apply code changes using the patch format. Supports creating, deleting, and modifying files."
        ),
        parameters={
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
    )


def tool_definition_shell() -> ToolDefinition:
    return ToolDefinition(
        name="shell",
        description="Execute a shell command. Returns stdout, stderr, and exit code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    )


def tool_definition_grep() -> ToolDefinition:
    return ToolDefinition(
        name="grep",
        description="Search file contents using regex patterns.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob_filter": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    )


def tool_definition_glob() -> ToolDefinition:
    return ToolDefinition(
        name="glob",
        description="Find files matching a glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    )


def tool_definition_list_dir() -> ToolDefinition:
    return ToolDefinition(
        name="list_dir",
        description="List directory contents up to a given depth.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "depth": {"type": "integer"},
            },
        },
    )


def tool_definition_read_many_files() -> ToolDefinition:
    return ToolDefinition(
        name="read_many_files",
        description="Read multiple files in one call.",
        parameters={
            "type": "object",
            "properties": {
                "file_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["file_paths"],
        },
    )


def tool_definition_web_search() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web for relevant pages.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer"},
                "timeout_ms": {"type": "integer"},
            },
            "required": ["query"],
        },
    )


def tool_definition_web_fetch() -> ToolDefinition:
    return ToolDefinition(
        name="web_fetch",
        description="Fetch content from a URL.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_ms": {"type": "integer"},
            },
            "required": ["url"],
        },
    )


def tool_definition_spawn_agent() -> ToolDefinition:
    return ToolDefinition(
        name="spawn_agent",
        description="Spawn a subagent to handle a scoped task autonomously.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "working_dir": {"type": "string"},
                "model": {"type": "string"},
                "max_turns": {"type": "integer"},
            },
            "required": ["task"],
        },
    )


def tool_definition_send_input() -> ToolDefinition:
    return ToolDefinition(
        name="send_input",
        description="Send a message to a running subagent.",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["agent_id", "message"],
        },
    )


def tool_definition_wait() -> ToolDefinition:
    return ToolDefinition(
        name="wait",
        description="Wait for a subagent to complete and return its result.",
        parameters={
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
            "required": ["agent_id"],
        },
    )


def tool_definition_close_agent() -> ToolDefinition:
    return ToolDefinition(
        name="close_agent",
        description="Terminate a subagent.",
        parameters={
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
            "required": ["agent_id"],
        },
    )


# Tool executors


def tool_read_file(args: Dict[str, Any], env: ExecutionEnvironment) -> Any:
    file_path = args["file_path"]
    offset = args.get("offset")
    limit = args.get("limit")

    media_type, _ = mimetypes.guess_type(file_path)
    if media_type and media_type.startswith("image/"):
        abs_path = os.path.abspath(os.path.join(env.working_directory(), file_path))
        with open(abs_path, "rb") as handle:
            data = handle.read()
        return ToolOutput(
            content=f"[image data for {file_path}]",
            image_data=data,
            image_media_type=media_type,
        )

    return env.read_file(file_path, offset, limit)


def tool_write_file(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    file_path = args["file_path"]
    content = args["content"]
    env.write_file(file_path, content)
    return f"Wrote {len(content.encode('utf-8'))} bytes to {file_path}"


def tool_edit_file(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    file_path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = bool(args.get("replace_all", False))

    abs_path = os.path.abspath(os.path.join(env.working_directory(), file_path))
    with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()

    count = content.count(old_string)
    if count == 0:
        raise ValueError("old_string not found")
    if not replace_all and count > 1:
        raise ValueError("old_string is not unique")

    if replace_all:
        updated = content.replace(old_string, new_string)
        replacements = count
    else:
        updated = content.replace(old_string, new_string, 1)
        replacements = 1

    with open(abs_path, "w", encoding="utf-8") as handle:
        handle.write(updated)

    return f"Replaced {replacements} occurrence(s) in {file_path}"


def _parse_patch_sections(patch: str) -> List[Tuple[str, List[str]]]:
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise ValueError("Patch must start with *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise ValueError("Patch must end with *** End Patch")
    sections: List[Tuple[str, List[str]]] = []
    idx = 1
    current_header: Optional[str] = None
    current_lines: List[str] = []
    header_prefixes = ("*** Add File: ", "*** Delete File: ", "*** Update File: ")
    while idx < len(lines) - 1:
        line = lines[idx]
        if line.startswith(header_prefixes):
            if current_header is not None:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)
        idx += 1
    if current_header is not None:
        sections.append((current_header, current_lines))
    return sections


def _apply_update_hunks(file_lines: List[str], hunks: List[str]) -> List[str]:
    idx = 0
    search_start = 0
    while idx < len(hunks):
        line = hunks[idx]
        if not line.startswith("@@"):
            raise ValueError("Missing hunk header")
        idx += 1
        before: List[str] = []
        after: List[str] = []
        while idx < len(hunks) and not hunks[idx].startswith("@@"):
            hunk_line = hunks[idx]
            if hunk_line.startswith(" "):
                before.append(hunk_line[1:])
                after.append(hunk_line[1:])
            elif hunk_line.startswith("-"):
                before.append(hunk_line[1:])
            elif hunk_line.startswith("+"):
                after.append(hunk_line[1:])
            elif hunk_line == "*** End of File":
                idx += 1
                break
            else:
                raise ValueError("Invalid hunk line")
            idx += 1

        match_index = _find_subsequence(file_lines, before, start=search_start)
        if match_index is None:
            match_index = _find_subsequence(file_lines, before, start=search_start, fuzzy=True)
        if match_index is None:
            raise ValueError("Hunk context not found")

        file_lines = (
            file_lines[:match_index]
            + after
            + file_lines[match_index + len(before) :]
        )
        search_start = match_index + len(after)
    return file_lines


def _find_subsequence(
    lines: List[str],
    subseq: List[str],
    start: int = 0,
    fuzzy: bool = False,
) -> Optional[int]:
    if not subseq:
        return start
    for i in range(start, len(lines) - len(subseq) + 1):
        window = lines[i : i + len(subseq)]
        if not fuzzy:
            if window == subseq:
                return i
        else:
            normalized_window = [normalize_line(line) for line in window]
            normalized_subseq = [normalize_line(line) for line in subseq]
            if normalized_window == normalized_subseq:
                return i
    return None


def tool_apply_patch(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    patch = args["patch"]
    sections = _parse_patch_sections(patch)
    results: List[str] = []

    for header, lines in sections:
        if header.startswith("*** Add File: "):
            path = header.replace("*** Add File: ", "").strip()
            content = "\n".join(line[1:] for line in lines if line.startswith("+"))
            env.write_file(path, content + ("\n" if content and not content.endswith("\n") else ""))
            results.append(f"added {path}")
        elif header.startswith("*** Delete File: "):
            path = header.replace("*** Delete File: ", "").strip()
            abs_path = os.path.abspath(os.path.join(env.working_directory(), path))
            if os.path.exists(abs_path):
                os.remove(abs_path)
            results.append(f"deleted {path}")
        elif header.startswith("*** Update File: "):
            path = header.replace("*** Update File: ", "").strip()
            move_to = None
            if lines and lines[0].startswith("*** Move to: "):
                move_to = lines[0].replace("*** Move to: ", "").strip()
                lines = lines[1:]

            abs_path = os.path.abspath(os.path.join(env.working_directory(), path))
            if not os.path.exists(abs_path):
                raise FileNotFoundError(path)
            with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
                file_lines = handle.read().splitlines()

            file_lines = _apply_update_hunks(file_lines, lines)
            updated = "\n".join(file_lines)
            with open(abs_path, "w", encoding="utf-8") as handle:
                handle.write(updated + ("\n" if updated and not updated.endswith("\n") else ""))

            if move_to:
                new_abs = os.path.abspath(os.path.join(env.working_directory(), move_to))
                os.makedirs(os.path.dirname(new_abs), exist_ok=True)
                os.replace(abs_path, new_abs)
                results.append(f"updated {path} -> {move_to}")
            else:
                results.append(f"updated {path}")
        else:
            raise ValueError("Unknown patch operation")

    return "\n".join(results)


def tool_shell(args: Dict[str, Any], env: ExecutionEnvironment, timeout_ms: int) -> str:
    command = args["command"]
    result = env.exec_command(command, timeout_ms, None, None)
    output = [
        f"exit_code: {result.exit_code}",
        f"timed_out: {result.timed_out}",
        f"duration_ms: {result.duration_ms}",
        "stdout:",
        result.stdout or "",
        "stderr:",
        result.stderr or "",
    ]
    if result.timed_out:
        output.append(
            f"[ERROR: Command timed out after {timeout_ms}ms. Partial output is shown above. "
            "You can retry with a longer timeout by setting the timeout_ms parameter.]"
        )
    return "\n".join(output)


def tool_grep(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path") or env.working_directory()
    options = {
        "glob_filter": args.get("glob_filter"),
        "case_insensitive": args.get("case_insensitive", False),
        "max_results": args.get("max_results", 100),
    }
    return env.grep(pattern, path, options)


def tool_glob(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path") or env.working_directory()
    matches = env.glob(pattern, path)
    return "\n".join(matches)


def tool_list_dir(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    path = args.get("path") or env.working_directory()
    depth = int(args.get("depth", 1))
    entries = env.list_directory(path, depth)
    lines = []
    for entry in entries:
        size = "" if entry.size is None else str(entry.size)
        lines.append(f"{entry.name}\t{entry.is_dir}\t{size}")
    return "\n".join(lines)


def tool_read_many_files(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    file_paths = args.get("file_paths") or []
    results = []
    for path in file_paths:
        content = env.read_file(path, None, None)
        results.append(f"# {path}\n{content}")
    return "\n\n".join(results)


def tool_web_search(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    import httpx

    query = args["query"]
    num_results = int(args.get("num_results", 5))
    timeout_ms = int(args.get("timeout_ms", 10_000))

    response = httpx.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        timeout=timeout_ms / 1000,
    )
    response.raise_for_status()
    html_text = response.text

    results: List[str] = []
    pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>')
    for match in pattern.finditer(html_text):
        url = match.group(1)
        title = re.sub(r"<.*?>", "", match.group(2))
        title = html.unescape(title).strip()
        results.append(f"{title} - {url}")
        if len(results) >= num_results:
            break

    return "\n".join(results)


def tool_web_fetch(args: Dict[str, Any], env: ExecutionEnvironment) -> str:
    import httpx

    url = args["url"]
    timeout_ms = int(args.get("timeout_ms", 10_000))
    response = httpx.get(url, timeout=timeout_ms / 1000)
    response.raise_for_status()
    return response.text


def describe_tool(definition: ToolDefinition) -> str:
    return (
        f"Tool: {definition.name}\n"
        f"Description: {definition.description}\n"
        f"Parameters: {json.dumps(definition.parameters, indent=2)}\n"
    )
