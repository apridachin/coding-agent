from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from unified_llm.types import Message, ToolCallData, Usage


@dataclass
class UserTurn:
    content: str
    timestamp: datetime


@dataclass
class AssistantTurn:
    content: str
    tool_calls: List[ToolCallData]
    reasoning: Optional[str]
    usage: Usage
    response_id: Optional[str]
    timestamp: datetime
    message: Optional[Message] = None


@dataclass
class ToolResultsTurn:
    results: List["ToolExecutionResult"]
    timestamp: datetime


@dataclass
class ToolExecutionResult:
    tool_call_id: str
    content: Union[str, Dict[str, Any], List[Any]]
    is_error: bool
    image_data: Optional[bytes] = None
    image_media_type: Optional[str] = None


@dataclass
class SystemTurn:
    content: str
    timestamp: datetime


@dataclass
class SteeringTurn:
    content: str
    timestamp: datetime
