from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Optional


class EventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class SessionEvent:
    kind: EventKind
    timestamp: datetime
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)


class EventEmitter:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._queue: asyncio.Queue[Optional[SessionEvent]] = asyncio.Queue()
        self._callbacks: list[Callable[[SessionEvent], None]] = []

    def emit(self, kind: EventKind, **data: Any) -> None:
        event = SessionEvent(
            kind=kind,
            timestamp=datetime.now(timezone.utc),
            session_id=self._session_id,
            data=data,
        )
        self._queue.put_nowait(event)
        if self._callbacks:
            for callback in list(self._callbacks):
                try:
                    callback(event)
                except Exception:
                    continue

    def subscribe(self, callback: Callable[[SessionEvent], None]) -> None:
        self._callbacks.append(callback)

    async def events(self) -> AsyncIterator[SessionEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    def close(self) -> None:
        self._queue.put_nowait(None)
