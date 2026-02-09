from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SessionConfig:
    max_turns: int = 0
    max_tool_rounds_per_input: int = 200
    require_tool_approval: bool = False
    default_command_timeout_ms: int = 10_000
    max_command_timeout_ms: int = 600_000
    reasoning_effort: Optional[str] = None
    tool_output_limits: Dict[str, int] = field(default_factory=dict)
    tool_line_limits: Dict[str, Optional[int]] = field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1
    user_instructions: Optional[str] = None
    use_streaming: bool = False
