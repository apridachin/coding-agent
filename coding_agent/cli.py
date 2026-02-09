from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from typing import Any, Optional

from unified_llm.client import Client

from . import LocalExecutionEnvironment, Session, SessionConfig
from .events import EventKind
from .providers import AnthropicProfile, GeminiProfile, OpenAIProfile


def _build_profile(provider: str, model: Optional[str]):
    provider = provider.lower()
    if provider == "openai":
        return OpenAIProfile(model=model or "gpt-5.2-codex")
    if provider == "anthropic":
        return AnthropicProfile(model=model or "claude-opus-4.5")
    if provider == "gemini":
        return GeminiProfile(model=model or "gemini-2.5-pro")
    raise ValueError(f"Unsupported provider: {provider}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive coding agent session")
    parser.add_argument("--provider", default="openai", help="Provider: openai|anthropic|gemini")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--workdir", default=None, help="Working directory for tools")
    parser.add_argument(
        "--reasoning",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=200,
        help="Max tool rounds per input",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Default shell timeout in milliseconds",
    )
    parser.add_argument(
        "--approve-tools",
        action="store_true",
        help="Prompt before executing each tool call",
    )
    # --show-tools controls tool rendering (inputs + outputs + start/end)
    tools_group = parser.add_mutually_exclusive_group()
    tools_group.add_argument(
        "--show-tools",
        dest="show_tools",
        action="store_true",
        help="Show tool calls (default)",
    )
    tools_group.add_argument(
        "--no-show-tools",
        dest="show_tools",
        action="store_false",
        help="Hide tool calls",
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        help="Stream assistant output",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming output (default)",
    )
    parser.set_defaults(stream=False, show_tools=True)
    return parser.parse_args()


async def _prompt_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _short_text(value: Any, limit: int = 140) -> str:
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _normalize_tool_args(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _summarize_tool_args(tool_name: str, args: Any) -> str:
    if not isinstance(args, dict):
        if isinstance(args, str) and args:
            return f"args={_short_text(args)}"
        return ""

    if tool_name == "shell":
        command = args.get("command")
        if command:
            return f"command={_short_text(command)}"
    if tool_name in {"read_file", "write_file", "edit_file"}:
        path = args.get("file_path")
        if path:
            extras = []
            if "offset" in args:
                extras.append(f"offset={args['offset']}")
            if "limit" in args:
                extras.append(f"limit={args['limit']}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            return f"file={_short_text(path)}{suffix}"
    if tool_name == "apply_patch":
        patch = args.get("patch")
        if patch is not None:
            return f"patch_len={len(str(patch))}"
    if tool_name == "grep":
        pattern = args.get("pattern")
        path = args.get("path")
        parts = []
        if pattern:
            parts.append(f"pattern={_short_text(pattern)}")
        if path:
            parts.append(f"path={_short_text(path)}")
        if parts:
            return " ".join(parts)
    if tool_name == "glob":
        pattern = args.get("pattern")
        path = args.get("path")
        parts = []
        if pattern:
            parts.append(f"pattern={_short_text(pattern)}")
        if path:
            parts.append(f"path={_short_text(path)}")
        if parts:
            return " ".join(parts)
    if tool_name == "spawn_agent":
        task = args.get("task")
        if task:
            return f"task={_short_text(task)}"
    if tool_name in {"send_input", "wait", "close_agent"}:
        agent_id = args.get("agent_id")
        if agent_id:
            return f"agent_id={_short_text(agent_id)}"

    return ""


async def _consume_events(session: Session, stream: bool, show_tools: bool) -> None:
    in_text = False
    pending_prefix = False
    tool_names: dict[str, str] = {}
    async for event in session.events():
        if event.kind == EventKind.ASSISTANT_TEXT_START:
            if stream:
                pending_prefix = True
        elif event.kind == EventKind.ASSISTANT_TEXT_DELTA and stream:
            if pending_prefix:
                print("assistant> ", end="", flush=True)
                pending_prefix = False
                in_text = True
            delta = event.data.get("delta", "")
            print(delta, end="", flush=True)
        elif event.kind == EventKind.ASSISTANT_TEXT_END:
            if stream:
                if in_text:
                    print()
                    in_text = False
                pending_prefix = False
            else:
                text = event.data.get("text", "")
                if text:
                    print(f"assistant> {text}")
        elif event.kind == EventKind.TOOL_CALL_START and show_tools:
            tool_name = event.data.get("tool_name", "unknown")
            call_id = event.data.get("call_id", "")
            if call_id:
                tool_names[call_id] = tool_name
            args = event.data.get("args")
            args_raw = event.data.get("args_raw")
            args = _normalize_tool_args(args if args is not None else args_raw)
            summary = _summarize_tool_args(tool_name, args)
            if summary:
                print(f"[tool:start] {tool_name} {summary}")
            else:
                print(f"[tool:start] {tool_name}")
            if args is not None:
                try:
                    args_text = json.dumps(args, indent=2)
                except TypeError:
                    args_text = str(args)
                print(f"args:\\n{args_text}")
        elif event.kind == EventKind.TOOL_CALL_END and show_tools:
            call_id = event.data.get("call_id", "")
            tool_name = tool_names.pop(call_id, "unknown") if call_id else "unknown"
            if "error" in event.data:
                error_msg = event.data.get("error", "unknown error")
                print(f"[tool:end] {tool_name} (error) {error_msg}")
            else:
                print(f"[tool:end] {tool_name}")
            if "output" in event.data:
                output = str(event.data.get("output", ""))
                if output:
                    preview_limit = 800
                    if len(output) > preview_limit:
                        half = preview_limit // 2
                        output = output[:half] + "\n... (truncated) ...\n" + output[-half:]
                    print(f"output:\\n{output}")
        elif event.kind == EventKind.ERROR:
            message = event.data.get("message", "unknown error")
            print(f"[error] {message}", file=sys.stderr)
        elif event.kind == EventKind.WARNING:
            message = event.data.get("message", "warning")
            print(f"[warn] {message}", file=sys.stderr)


async def _run() -> int:
    args = _parse_args()
    client = Client.from_env()
    if not client.providers:
        print("No providers configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY.")
        return 2

    if args.provider not in client.providers:
        available = ", ".join(sorted(client.providers.keys()))
        print(f"Provider '{args.provider}' not configured. Available: {available}")
        return 2

    profile = _build_profile(args.provider, args.model)
    workdir = args.workdir or os.getcwd()
    env = LocalExecutionEnvironment(workdir)

    config = SessionConfig(
        max_tool_rounds_per_input=args.max_rounds,
        reasoning_effort=args.reasoning,
        use_streaming=args.stream,
    )
    config.require_tool_approval = args.approve_tools
    if args.timeout is not None:
        config.default_command_timeout_ms = args.timeout

    approve_all = False

    async def _approve_tool(tool_call, args) -> bool:
        nonlocal approve_all
        if approve_all:
            return True
        try:
            args_text = json.dumps(args, indent=2)
        except TypeError:
            args_text = str(args)
        print(f"[approve] {tool_call.name}\n{args_text}")
        response = (await _prompt_input("approve? [y/N/a] ")).strip().lower()
        if response in {"a", "all"}:
            approve_all = True
            return True
        return response in {"y", "yes"}

    session_kwargs = {
        "provider_profile": profile,
        "execution_env": env,
        "llm_client": client,
        "config": config,
    }
    try:
        accepts_tool_approval = "tool_approval" in inspect.signature(Session.__init__).parameters
    except (TypeError, ValueError):
        accepts_tool_approval = False
    if accepts_tool_approval:
        session_kwargs["tool_approval"] = _approve_tool if args.approve_tools else None
    session = Session(**session_kwargs)

    print("Interactive coding agent. Type 'exit' to quit.")
    event_task = asyncio.create_task(
        _consume_events(
            session,
            stream=args.stream,
            show_tools=args.show_tools,
        )
    )
    try:
        while True:
            try:
                user_input = await _prompt_input("user> ")
            except EOFError:
                print()
                break

            if not user_input.strip():
                continue
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            await session.submit(user_input)
    finally:
        session.close()
        await event_task

    return 0


def main() -> None:
    try:
        code = asyncio.run(_run())
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
