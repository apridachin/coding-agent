from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, List


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def normalize_line(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line.strip())
    return normalized


def hash_tool_call(name: str, arguments: Any) -> str:
    return f"{name}:{safe_json_dumps(arguments)}"


def total_chars_in_history(texts: Iterable[str]) -> int:
    return sum(len(text) for text in texts)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_path_segments(root: str, target: str) -> List[str]:
    root = os.path.abspath(root)
    target = os.path.abspath(target)
    if not target.startswith(root):
        return [root]
    segments = [root]
    current = root
    while current != target:
        next_part = os.path.relpath(target, current).split(os.sep)[0]
        current = os.path.join(current, next_part)
        segments.append(current)
    return segments
