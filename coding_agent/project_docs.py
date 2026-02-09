from __future__ import annotations

import os
from typing import List, Tuple

from .utils import split_path_segments

PROVIDER_DOCS = {
    "openai": [".codex/instructions.md"],
    "anthropic": ["CLAUDE.md"],
    "gemini": ["GEMINI.md"],
}


def discover_project_docs(working_dir: str, provider_id: str) -> str:
    git_root = _find_git_root(working_dir)
    root = git_root or working_dir
    segments = split_path_segments(root, working_dir)

    filenames = ["AGENTS.md"] + PROVIDER_DOCS.get(provider_id, [])

    docs: List[Tuple[str, str]] = []
    total_bytes = 0
    budget = 32 * 1024

    for directory in segments:
        for name in filenames:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
                header = f"\n\n# {path}\n"
                combined = header + content
                total_bytes += len(combined.encode("utf-8"))
                if total_bytes > budget:
                    remaining = budget - (total_bytes - len(combined.encode("utf-8")))
                    snippet = combined.encode("utf-8")[: max(0, remaining)].decode("utf-8", errors="ignore")
                    docs.append((path, snippet + "\n[Project instructions truncated at 32KB]"))
                    return "".join(doc for _, doc in docs)
                docs.append((path, combined))

    return "".join(doc for _, doc in docs)


def _find_git_root(start: str) -> str | None:
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
