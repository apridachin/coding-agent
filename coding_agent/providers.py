from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .tools import (
    RegisteredTool,
    ToolRegistry,
    tool_definition_apply_patch,
    tool_definition_close_agent,
    tool_definition_edit_file,
    tool_definition_glob,
    tool_definition_grep,
    tool_definition_list_dir,
    tool_definition_read_file,
    tool_definition_read_many_files,
    tool_definition_send_input,
    tool_definition_shell,
    tool_definition_spawn_agent,
    tool_definition_wait,
    tool_definition_web_fetch,
    tool_definition_web_search,
    tool_definition_write_file,
)


@dataclass
class ProviderProfile:
    id: str
    model: str
    tool_registry: ToolRegistry
    supports_reasoning: bool
    supports_streaming: bool
    supports_parallel_tool_calls: bool
    context_window_size: int
    knowledge_cutoff: str
    default_command_timeout_ms: int
    provider_options_data: Optional[Dict[str, Any]] = None

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def tools(self) -> list:
        return self.tool_registry.definitions()

    def provider_options(self) -> Optional[Dict[str, Any]]:
        return self.provider_options_data


OPENAI_BASE_PROMPT = """
# Instructions
- The user will provide a task.
- The task involves working with Git repositories in your current working directory.
- Wait for all terminal commands to be completed (or terminate them) before finishing.

# Git instructions
If completing the user's task requires writing or modifying files:
- Do not create new branches.
- Use git to commit your changes.
- If pre-commit fails, fix issues and retry.
- Check git status to confirm your commit. You must leave your worktree in a clean state.
- Only committed code will be evaluated.
- Do not modify or amend existing commits.

# AGENTS.md spec
- Containers often contain AGENTS.md files. These files can appear anywhere in the container's filesystem. Typical locations include `/`, `~`, and in various places inside of Git repos.
- These files are a way for humans to give you (the agent) instructions or tips for working within the container.
- Some examples might be: coding conventions, info about how code is organized, or instructions for how to run or test code.
- AGENTS.md files may provide instructions about PR messages (messages attached to a GitHub Pull Request produced by the agent, describing the PR). These instructions should be respected.
- Instructions in AGENTS.md files:
  - The scope of an AGENTS.md file is the entire directory tree rooted at the folder that contains it.
  - For every file you touch in the final patch, you must obey instructions in any AGENTS.md file whose scope includes that file.
  - Instructions about code style, structure, naming, etc. apply only to code within the AGENTS.md file's scope, unless the file states otherwise.
  - More-deeply-nested AGENTS.md files take precedence in the case of conflicting instructions.
  - Direct system/developer/user instructions (as part of a prompt) take precedence over AGENTS.md instructions.
- AGENTS.md files need not live only in Git repos. For example, you may find one in your home directory.
- If the AGENTS.md includes programmatic checks to verify your work, you MUST run all of the programmatic checks and make a best effort to validate that the checks pass AFTER all code changes have been made.
  - This applies even for changes that appear simple, i.e. documentation. You still must run all of the programmatic checks.

# Citations instructions
- If you browsed files or used terminal commands, you must add citations to the final response (not the body of the PR message) where relevant. Citations reference file paths and terminal outputs with the following formats:
  1) `【F:<file_path>†L<line_start>(-L<line_end>)?】`
  - File path citations must start with `F:`. `file_path` is the exact file path of the file relative to the root of the repository that contains the relevant text.
  - `line_start` is the 1-indexed start line number of the relevant output within that file.
  2) `【<chunk_id>†L<line_start>(-L<line_end>)?】`
  - Where `chunk_id` is the chunk_id of the terminal output, `line_start` and `line_end` are the 1-indexed start and end line numbers of the relevant output within that chunk.
- Line ends are optional, and if not provided, line end is the same as line start, so only 1 line is cited.
- Ensure that the line numbers are correct, and that the cited file paths or terminal outputs are directly relevant to the word or clause before the citation.
- Do not cite completely empty lines inside the chunk, only cite lines that have content.
- Only cite from file paths and terminal outputs, DO NOT cite from previous pr diffs and comments, nor cite git hashes as chunk ids.
- Use file path citations that reference any code changes, documentation or files, and use terminal citations only for relevant terminal output.
- Prefer file citations over terminal citations unless the terminal output is directly relevant to the clauses before the citation, i.e. clauses on test results.
  - For PR creation tasks, use file citations when referring to code changes in the summary section of your final response, and terminal citations in the testing section.
  - For question-answering tasks, you should only use terminal citations if you need to programmatically verify an answer (i.e. counting lines of code). Otherwise, use file citations.
""".strip()

ANTHROPIC_BASE_PROMPT = """
You are a coding agent. Prefer using edit_file with exact old_string/new_string for modifications.
Read files before editing. Avoid overwriting files when you can edit them.
""".strip()

GEMINI_BASE_PROMPT = """
You are a coding agent. Use the provided tools to read, edit, and run commands. Respect GEMINI.md.
""".strip()


class OpenAIProfile(ProviderProfile):
    def __init__(
        self,
        model: str = "gpt-5.2-codex",
        context_window_size: int = 128_000,
        knowledge_cutoff: str = "2024-06-01",
        default_command_timeout_ms: int = 10_000,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        registry = ToolRegistry()
        registry.register(RegisteredTool(tool_definition_read_file(), _noop))
        registry.register(RegisteredTool(tool_definition_apply_patch(), _noop))
        registry.register(RegisteredTool(tool_definition_write_file(), _noop))
        registry.register(RegisteredTool(tool_definition_shell(), _noop))
        registry.register(RegisteredTool(tool_definition_grep(), _noop))
        registry.register(RegisteredTool(tool_definition_glob(), _noop))
        registry.register(RegisteredTool(tool_definition_spawn_agent(), _noop))
        registry.register(RegisteredTool(tool_definition_send_input(), _noop))
        registry.register(RegisteredTool(tool_definition_wait(), _noop))
        registry.register(RegisteredTool(tool_definition_close_agent(), _noop))
        super().__init__(
            id="openai",
            model=model,
            tool_registry=registry,
            supports_reasoning=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            context_window_size=context_window_size,
            knowledge_cutoff=knowledge_cutoff,
            default_command_timeout_ms=default_command_timeout_ms,
            provider_options_data=provider_options,
        )

    def build_system_prompt(self) -> str:
        return OPENAI_BASE_PROMPT


class AnthropicProfile(ProviderProfile):
    def __init__(
        self,
        model: str = "claude-opus-4.5",
        context_window_size: int = 200_000,
        knowledge_cutoff: str = "2024-06-01",
        default_command_timeout_ms: int = 120_000,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        registry = ToolRegistry()
        registry.register(RegisteredTool(tool_definition_read_file(), _noop))
        registry.register(RegisteredTool(tool_definition_write_file(), _noop))
        registry.register(RegisteredTool(tool_definition_edit_file(), _noop))
        registry.register(RegisteredTool(tool_definition_shell(), _noop))
        registry.register(RegisteredTool(tool_definition_grep(), _noop))
        registry.register(RegisteredTool(tool_definition_glob(), _noop))
        registry.register(RegisteredTool(tool_definition_spawn_agent(), _noop))
        registry.register(RegisteredTool(tool_definition_send_input(), _noop))
        registry.register(RegisteredTool(tool_definition_wait(), _noop))
        registry.register(RegisteredTool(tool_definition_close_agent(), _noop))
        super().__init__(
            id="anthropic",
            model=model,
            tool_registry=registry,
            supports_reasoning=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            context_window_size=context_window_size,
            knowledge_cutoff=knowledge_cutoff,
            default_command_timeout_ms=default_command_timeout_ms,
            provider_options_data=provider_options,
        )

    def build_system_prompt(self) -> str:
        return ANTHROPIC_BASE_PROMPT


class GeminiProfile(ProviderProfile):
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        context_window_size: int = 1_000_000,
        knowledge_cutoff: str = "2024-06-01",
        default_command_timeout_ms: int = 10_000,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        registry = ToolRegistry()
        registry.register(RegisteredTool(tool_definition_read_file(), _noop))
        registry.register(RegisteredTool(tool_definition_read_many_files(), _noop))
        registry.register(RegisteredTool(tool_definition_write_file(), _noop))
        registry.register(RegisteredTool(tool_definition_edit_file(), _noop))
        registry.register(RegisteredTool(tool_definition_shell(), _noop))
        registry.register(RegisteredTool(tool_definition_grep(), _noop))
        registry.register(RegisteredTool(tool_definition_glob(), _noop))
        registry.register(RegisteredTool(tool_definition_list_dir(), _noop))
        registry.register(RegisteredTool(tool_definition_web_search(), _noop))
        registry.register(RegisteredTool(tool_definition_web_fetch(), _noop))
        registry.register(RegisteredTool(tool_definition_spawn_agent(), _noop))
        registry.register(RegisteredTool(tool_definition_send_input(), _noop))
        registry.register(RegisteredTool(tool_definition_wait(), _noop))
        registry.register(RegisteredTool(tool_definition_close_agent(), _noop))
        super().__init__(
            id="gemini",
            model=model,
            tool_registry=registry,
            supports_reasoning=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            context_window_size=context_window_size,
            knowledge_cutoff=knowledge_cutoff,
            default_command_timeout_ms=default_command_timeout_ms,
            provider_options_data=provider_options,
        )

    def build_system_prompt(self) -> str:
        return GEMINI_BASE_PROMPT


def _noop(*_args: Any, **_kwargs: Any) -> str:
    raise RuntimeError("Executor not bound")
