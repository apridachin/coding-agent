from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import jsonschema
from unified_llm.stream import StreamResult
from unified_llm.types import (
    ContentKind,
    ContentPart,
    Message,
    Request,
    Role,
    StreamEventType,
    ToolCallData,
    ToolChoice,
    ToolResultData,
)

from .config import SessionConfig
from .events import EventEmitter, EventKind
from .execution import ExecutionEnvironment, ScopedExecutionEnvironment
from .project_docs import discover_project_docs
from .providers import ProviderProfile
from .tools import (
    RegisteredTool,
    ToolOutput,
    describe_tool,
    tool_apply_patch,
    tool_edit_file,
    tool_glob,
    tool_grep,
    tool_list_dir,
    tool_read_file,
    tool_read_many_files,
    tool_shell,
    tool_web_fetch,
    tool_web_search,
    tool_write_file,
)
from .truncation import truncate_tool_output
from .turns import (
    AssistantTurn,
    SteeringTurn,
    SystemTurn,
    ToolExecutionResult,
    ToolResultsTurn,
    UserTurn,
)
from .utils import hash_tool_call, now_utc, total_chars_in_history


class SessionState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass
class SubAgentHandle:
    id: str
    session: "Session"
    task: asyncio.Task[str]
    status: str = "running"
    result: Optional[Dict[str, Any]] = None


class Session:
    def __init__(
        self,
        provider_profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        llm_client,
        config: Optional[SessionConfig] = None,
        depth: int = 0,
        tool_approval: Optional[
            Callable[[ToolCallData, Dict[str, Any]], Awaitable[bool]]
        ] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider_profile = provider_profile
        self.execution_env = execution_env
        self.llm_client = llm_client
        if config is None:
            self.config = SessionConfig()
            self.config.default_command_timeout_ms = provider_profile.default_command_timeout_ms
        else:
            self.config = config
        self.state = SessionState.IDLE
        self.history: List[Any] = []
        self.steering_queue: asyncio.Queue[str] = asyncio.Queue()
        self.followup_queue: asyncio.Queue[str] = asyncio.Queue()
        self.subagents: Dict[str, SubAgentHandle] = {}
        self._abort = False
        self._llm_task: Optional[asyncio.Task[Any]] = None
        self._depth = depth
        self.event_emitter = EventEmitter(self.id)
        self.tool_registry = provider_profile.tool_registry.clone()
        self._bind_tool_executors()
        self._tool_approval = tool_approval
        self.event_emitter.emit(EventKind.SESSION_START)

    def _bind_tool_executors(self) -> None:
        bindings = {
            "read_file": tool_read_file,
            "write_file": tool_write_file,
            "edit_file": tool_edit_file,
            "apply_patch": tool_apply_patch,
            "shell": self._tool_shell,
            "grep": tool_grep,
            "glob": tool_glob,
            "list_dir": tool_list_dir,
            "read_many_files": tool_read_many_files,
            "web_search": tool_web_search,
            "web_fetch": tool_web_fetch,
            "spawn_agent": self._tool_spawn_agent,
            "send_input": self._tool_send_input,
            "wait": self._tool_wait_agent,
            "close_agent": self._tool_close_agent,
        }

        for name, executor in bindings.items():
            definition = self.tool_registry.get(name)
            if definition is None:
                continue
            self.tool_registry.register(RegisteredTool(definition.definition, executor))

    async def submit(self, user_input: str) -> None:
        await self.process_input(user_input)

    async def process_input(self, user_input: str) -> None:
        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input, timestamp=now_utc()))
        self.event_emitter.emit(EventKind.USER_INPUT, content=user_input)

        await self._drain_steering()

        round_count = 0

        while True:
            if self.config.max_tool_rounds_per_input and round_count >= self.config.max_tool_rounds_per_input:
                self.event_emitter.emit(EventKind.TURN_LIMIT, round=round_count)
                break

            if self.config.max_turns and len(self.history) >= self.config.max_turns:
                self.event_emitter.emit(EventKind.TURN_LIMIT, total_turns=len(self.history))
                break

            if self._abort:
                break

            system_prompt = self._build_system_prompt()
            messages = [Message.system(system_prompt)] + self._convert_history_to_messages()

            request = Request(
                model=self.provider_profile.model,
                messages=messages,
                tools=self.tool_registry.definitions(),
                tool_choice=ToolChoice(mode="auto"),
                reasoning_effort=self.config.reasoning_effort,
                provider=self.provider_profile.id,
                provider_options=self.provider_profile.provider_options(),
            )

            self.event_emitter.emit(EventKind.ASSISTANT_TEXT_START)
            try:
                response = await self._call_llm(request)
            except Exception as exc:  # noqa: BLE001
                self.event_emitter.emit(EventKind.ERROR, message=f"LLM error: {exc}")
                self.state = SessionState.CLOSED
                self.event_emitter.emit(EventKind.SESSION_END, state=self.state)
                return

            assistant_turn = AssistantTurn(
                content=response.text,
                tool_calls=response.tool_calls,
                reasoning=response.reasoning,
                usage=response.usage,
                response_id=response.id,
                timestamp=now_utc(),
                message=response.message,
            )
            self.history.append(assistant_turn)
            self.event_emitter.emit(
                EventKind.ASSISTANT_TEXT_END,
                text=response.text,
                reasoning=response.reasoning,
            )

            if not response.tool_calls:
                break

            round_count += 1
            results = await self._execute_tool_calls(response.tool_calls)
            self.history.append(ToolResultsTurn(results=results, timestamp=now_utc()))

            await self._drain_steering()

            if self.config.enable_loop_detection and self._detect_loop(self.config.loop_detection_window):
                warning = (
                    "Loop detected: the last "
                    + str(self.config.loop_detection_window)
                    + " tool calls follow a repeating pattern. Try a different approach."
                )
                self.history.append(SteeringTurn(content=warning, timestamp=now_utc()))
                self.event_emitter.emit(EventKind.LOOP_DETECTION, message=warning)

            self._check_context_usage()

        if not self.followup_queue.empty():
            next_input = await self.followup_queue.get()
            await self.process_input(next_input)
            return

        if self._should_await_input():
            self.state = SessionState.AWAITING_INPUT
        else:
            self.state = SessionState.IDLE
        self.event_emitter.emit(EventKind.SESSION_END, state=self.state)

    async def _call_llm(self, request: Request):
        self._llm_task = asyncio.current_task()
        try:
            if self.config.use_streaming and self.provider_profile.supports_streaming:
                stream = await self.llm_client.stream(request)
                stream_result = StreamResult(stream)
                async for event in stream_result:
                    if event.type == StreamEventType.TEXT_DELTA and event.delta is not None:
                        self.event_emitter.emit(EventKind.ASSISTANT_TEXT_DELTA, delta=event.delta)
                return await stream_result.response()
            return await self.llm_client.complete(request)
        finally:
            self._llm_task = None

    async def _execute_tool_calls(
        self, tool_calls: List[ToolCallData]
    ) -> List[ToolExecutionResult]:
        if self.provider_profile.supports_parallel_tool_calls and len(tool_calls) > 1:
            results = await asyncio.gather(
                *[self._execute_single_tool(tool_call) for tool_call in tool_calls]
            )
            return list(results)

        results: List[ToolExecutionResult] = []
        for tool_call in tool_calls:
            results.append(await self._execute_single_tool(tool_call))
        return results

    async def _execute_single_tool(self, tool_call: ToolCallData) -> ToolExecutionResult:
        args: Dict[str, Any] = {}
        try:
            args = self._parse_tool_arguments(tool_call.arguments)
        except Exception as exc:  # noqa: BLE001
            self.event_emitter.emit(
                EventKind.TOOL_CALL_START,
                tool_name=tool_call.name,
                call_id=tool_call.id,
                args=args,
                args_raw=tool_call.arguments,
            )
            error_msg = f"Tool error ({tool_call.name}): {exc}"
            self.event_emitter.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            self.event_emitter.emit(EventKind.ERROR, message=error_msg)
            return ToolExecutionResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

        self.event_emitter.emit(
            EventKind.TOOL_CALL_START,
            tool_name=tool_call.name,
            call_id=tool_call.id,
            args=args,
            args_raw=tool_call.arguments,
        )
        registered = self.tool_registry.get(tool_call.name)
        if registered is None:
            error_msg = f"Unknown tool: {tool_call.name}"
            self.event_emitter.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            self.event_emitter.emit(EventKind.ERROR, message=error_msg)
            return ToolExecutionResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

        try:
            jsonschema.validate(instance=args, schema=registered.definition.parameters)
            if self.config.require_tool_approval and not await self._approve_tool_call(
                tool_call, args
            ):
                error_msg = f"Tool call declined: {tool_call.name}"
                self.event_emitter.emit(
                    EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg
                )
                self.event_emitter.emit(EventKind.ERROR, message=error_msg)
                return ToolExecutionResult(
                    tool_call_id=tool_call.id, content=error_msg, is_error=True
                )
            raw_output = await self._run_tool_executor(registered.executor, args)
            image_data = None
            image_media_type = None
            if isinstance(raw_output, ToolOutput):
                image_data = raw_output.image_data
                image_media_type = raw_output.image_media_type
                raw_output = raw_output.content
            if not isinstance(raw_output, str):
                raw_output = json.dumps(raw_output)
            truncated_output = truncate_tool_output(
                raw_output,
                tool_call.name,
                self.config.tool_output_limits,
                self.config.tool_line_limits,
            )
            if raw_output:
                chunk_size = 4096
                for idx in range(0, len(raw_output), chunk_size):
                    self.event_emitter.emit(
                        EventKind.TOOL_CALL_OUTPUT_DELTA,
                        call_id=tool_call.id,
                        delta=raw_output[idx : idx + chunk_size],
                    )
            self.event_emitter.emit(
                EventKind.TOOL_CALL_END,
                call_id=tool_call.id,
                output=raw_output,
            )
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                content=truncated_output,
                is_error=False,
                image_data=image_data,
                image_media_type=image_media_type,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Tool error ({tool_call.name}): {exc}"
            self.event_emitter.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            self.event_emitter.emit(EventKind.ERROR, message=error_msg)
            return ToolExecutionResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

    async def _approve_tool_call(self, tool_call: ToolCallData, args: Dict[str, Any]) -> bool:
        if not getattr(self.config, "require_tool_approval", False):
            return True
        if self._tool_approval is None:
            return False
        return await self._tool_approval(tool_call, args)

    async def _run_tool_executor(self, executor, args: Dict[str, Any]) -> str:
        if asyncio.iscoroutinefunction(executor):
            return await executor(args, self.execution_env)
        return await asyncio.to_thread(executor, args, self.execution_env)

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            return json.loads(arguments or "{}")
        raise ValueError("Invalid tool arguments")

    async def _drain_steering(self) -> None:
        while not self.steering_queue.empty():
            msg = await self.steering_queue.get()
            self.history.append(SteeringTurn(content=msg, timestamp=now_utc()))
            self.event_emitter.emit(EventKind.STEERING_INJECTED, content=msg)

    def steer(self, message: str) -> None:
        self.steering_queue.put_nowait(message)

    def follow_up(self, message: str) -> None:
        self.followup_queue.put_nowait(message)

    def abort(self) -> None:
        self._abort = True
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
        for handle in self.subagents.values():
            if not handle.task.done():
                handle.task.cancel()
            handle.status = "failed"
        self.execution_env.cleanup()
        self.state = SessionState.CLOSED
        self.event_emitter.emit(EventKind.SESSION_END, state=self.state)

    def close(self) -> None:
        self.state = SessionState.CLOSED
        self.event_emitter.emit(EventKind.SESSION_END, state=self.state)
        self.event_emitter.close()

    def events(self):
        return self.event_emitter.events()

    def _build_system_prompt(self) -> str:
        base_prompt = self.provider_profile.build_system_prompt()
        env_block = self._environment_context()
        tool_descriptions = self._tool_descriptions()
        project_docs = discover_project_docs(
            self.execution_env.working_directory(), self.provider_profile.id
        )
        user_override = self.config.user_instructions or ""

        return "\n\n".join(
            chunk
            for chunk in [base_prompt, env_block, tool_descriptions, project_docs, user_override]
            if chunk
        )

    def _environment_context(self) -> str:
        return (
            "<environment>\n"
            f"Working directory: {self.execution_env.working_directory()}\n"
            f"Is git repository: {self._is_git_repo()}\n"
            f"Git branch: {self._git_branch()}\n"
            f"Platform: {self.execution_env.platform()}\n"
            f"OS version: {self.execution_env.os_version()}\n"
            f"Today's date: {datetime.now().date().isoformat()}\n"
            f"Model: {self.provider_profile.model}\n"
            f"Knowledge cutoff: {self.provider_profile.knowledge_cutoff}\n"
            "</environment>\n"
            f"{self._git_context()}"
        )

    def _tool_descriptions(self) -> str:
        descriptions = [describe_tool(defn) for defn in self.tool_registry.definitions()]
        return "<tools>\n" + "\n".join(descriptions) + "</tools>"

    def _git_context(self) -> str:
        if not self._is_git_repo():
            return ""
        recent_commits = self._git_recent_commits()
        status_summary = self._git_status_summary()
        return (
            "<git_context>\n"
            f"Branch: {self._git_branch()}\n"
            f"Status: {status_summary}\n"
            f"Recent commits:\n{recent_commits}\n"
            "</git_context>"
        )

    def _git_recent_commits(self) -> str:
        result = self.execution_env.exec_command(
            "git log -n 5 --pretty=format:%h %s",
            timeout_ms=5_000,
            working_dir=self.execution_env.working_directory(),
            env_vars=None,
        )
        return result.stdout.strip()

    def _git_status_summary(self) -> str:
        result = self.execution_env.exec_command(
            "git status --porcelain",
            timeout_ms=5_000,
            working_dir=self.execution_env.working_directory(),
            env_vars=None,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        modified = sum(1 for line in lines if line and line[0] != "?" and line[1] != "?")
        untracked = sum(1 for line in lines if line.startswith("??"))
        return f"modified={modified}, untracked={untracked}"

    def _is_git_repo(self) -> bool:
        result = self.execution_env.exec_command(
            "git rev-parse --is-inside-work-tree",
            timeout_ms=3_000,
            working_dir=self.execution_env.working_directory(),
            env_vars=None,
        )
        return result.exit_code == 0

    def _git_branch(self) -> str:
        result = self.execution_env.exec_command(
            "git branch --show-current",
            timeout_ms=3_000,
            working_dir=self.execution_env.working_directory(),
            env_vars=None,
        )
        branch = result.stdout.strip()
        if branch:
            return branch
        fallback = self.execution_env.exec_command(
            "git rev-parse --short HEAD",
            timeout_ms=3_000,
            working_dir=self.execution_env.working_directory(),
            env_vars=None,
        )
        return fallback.stdout.strip() or "unknown"

    def _convert_history_to_messages(self) -> List[Message]:
        messages: List[Message] = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, SteeringTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, SystemTurn):
                messages.append(Message.system(turn.content))
            elif isinstance(turn, AssistantTurn):
                if turn.message is not None:
                    messages.append(turn.message)
                else:
                    content_parts = [ContentPart(kind=ContentKind.TEXT, text=turn.content)]
                    for tool_call in turn.tool_calls:
                        content_parts.append(
                            ContentPart(kind=ContentKind.TOOL_CALL, tool_call=tool_call)
                        )
                    messages.append(Message(role=Message.assistant("").role, content=content_parts))
            elif isinstance(turn, ToolResultsTurn):
                for result in turn.results:
                    if result.image_data or result.image_media_type:
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                content=[
                                    ContentPart(
                                        kind=ContentKind.TOOL_RESULT,
                                        tool_result=ToolResultData(
                                            tool_call_id=result.tool_call_id,
                                            content=result.content,
                                            is_error=result.is_error,
                                            image_data=result.image_data,
                                            image_media_type=result.image_media_type,
                                        ),
                                    )
                                ],
                                tool_call_id=result.tool_call_id,
                            )
                        )
                    else:
                        messages.append(
                            Message.tool_result(
                                tool_call_id=result.tool_call_id,
                                content=result.content,
                                is_error=result.is_error,
                            )
                        )
        return messages

    def _detect_loop(self, window_size: int) -> bool:
        signatures: List[str] = []
        for turn in self.history:
            if isinstance(turn, AssistantTurn):
                for tool_call in turn.tool_calls:
                    signatures.append(hash_tool_call(tool_call.name, tool_call.arguments))
        if len(signatures) < window_size:
            return False
        recent = signatures[-window_size:]
        for pattern_len in (1, 2, 3):
            if window_size % pattern_len != 0:
                continue
            pattern = recent[:pattern_len]
            matches = True
            for idx in range(pattern_len, window_size, pattern_len):
                if recent[idx : idx + pattern_len] != pattern:
                    matches = False
                    break
            if matches:
                return True
        return False

    def _check_context_usage(self) -> None:
        texts: List[str] = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                texts.append(turn.content)
            elif isinstance(turn, AssistantTurn):
                texts.append(turn.content)
            elif isinstance(turn, SteeringTurn):
                texts.append(turn.content)
            elif isinstance(turn, ToolResultsTurn):
                for result in turn.results:
                    if isinstance(result.content, str):
                        texts.append(result.content)
                    else:
                        texts.append(json.dumps(result.content))
        approx_tokens = total_chars_in_history(texts) / 4
        threshold = self.provider_profile.context_window_size * 0.8
        if approx_tokens > threshold:
            percent = round(approx_tokens / self.provider_profile.context_window_size * 100)
            self.event_emitter.emit(
                EventKind.WARNING,
                message=f"Context usage at ~{percent}% of context window",
            )

    def _should_await_input(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        if isinstance(last, AssistantTurn):
            text = last.content.strip()
            if not text:
                return False
            last_line = text.splitlines()[-1].strip()
            if last_line.endswith("?"):
                return True
            lowered = text.lower()
            question_starters = (
                "please provide",
                "could you",
                "can you",
                "what is",
                "which",
                "who",
                "when",
                "where",
                "why",
                "how",
            )
            return any(starter in lowered for starter in question_starters)
        return False

    async def _tool_shell(self, args: Dict[str, Any], env: ExecutionEnvironment) -> str:
        timeout_ms = int(args.get("timeout_ms") or self.config.default_command_timeout_ms)
        timeout_ms = min(timeout_ms, self.config.max_command_timeout_ms)
        if timeout_ms <= 0:
            timeout_ms = self.config.default_command_timeout_ms
        return await asyncio.to_thread(tool_shell, args, env, timeout_ms)

    async def _tool_spawn_agent(self, args: Dict[str, Any], env: ExecutionEnvironment) -> str:
        if self._depth >= self.config.max_subagent_depth:
            raise RuntimeError("Max subagent depth reached")
        task = args["task"]
        working_dir = args.get("working_dir") or env.working_directory()
        if not str(working_dir).startswith("/"):
            import os

            working_dir = os.path.abspath(os.path.join(env.working_directory(), str(working_dir)))
        max_turns = args.get("max_turns")
        model = args.get("model")

        profile = self.provider_profile
        if model and model != profile.model:
            profile = type(profile)(
                model=model,
                context_window_size=profile.context_window_size,
                knowledge_cutoff=profile.knowledge_cutoff,
                default_command_timeout_ms=profile.default_command_timeout_ms,
                provider_options=profile.provider_options_data,
            )

        config = copy.deepcopy(self.config)
        if max_turns is not None:
            config.max_turns = int(max_turns)

        sub_env = env
        if working_dir != env.working_directory():
            sub_env = ScopedExecutionEnvironment(env, working_dir)

        sub_session = Session(
            provider_profile=profile,
            execution_env=sub_env,
            llm_client=self.llm_client,
            config=config,
            depth=self._depth + 1,
        )

        async def _run() -> str:
            await sub_session.submit(task)
            last_text = ""
            for turn in reversed(sub_session.history):
                if isinstance(turn, AssistantTurn):
                    last_text = turn.content
                    break
            return last_text

        agent_id = str(uuid.uuid4())
        task_handle = asyncio.create_task(_run())
        self.subagents[agent_id] = SubAgentHandle(id=agent_id, session=sub_session, task=task_handle)
        return json.dumps({"agent_id": agent_id, "status": "running"})

    async def _tool_send_input(self, args: Dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        message = args["message"]
        handle = self.subagents.get(agent_id)
        if not handle:
            raise ValueError("Unknown agent_id")
        if handle.session.state == SessionState.PROCESSING:
            handle.session.steer(message)
        else:
            await handle.session.submit(message)
        return json.dumps({"agent_id": agent_id, "status": handle.session.state.value})

    async def _tool_wait_agent(self, args: Dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        handle = self.subagents.get(agent_id)
        if not handle:
            raise ValueError("Unknown agent_id")
        try:
            output = await handle.task
            handle.status = "completed"
            result = {
                "output": output,
                "success": True,
                "turns_used": len(handle.session.history),
            }
        except asyncio.CancelledError:
            handle.status = "failed"
            result = {"output": "", "success": False, "turns_used": len(handle.session.history)}
        except Exception as exc:  # noqa: BLE001
            handle.status = "failed"
            result = {
                "output": f"Subagent error: {exc}",
                "success": False,
                "turns_used": len(handle.session.history),
            }
        handle.result = result
        return json.dumps(result)

    async def _tool_close_agent(self, args: Dict[str, Any], env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        handle = self.subagents.get(agent_id)
        if not handle:
            raise ValueError("Unknown agent_id")
        if not handle.task.done():
            handle.task.cancel()
        handle.status = "failed"
        return json.dumps({"agent_id": agent_id, "status": handle.status})
