from __future__ import annotations

import fnmatch
import os
import platform as platform_module
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: Optional[int]


class ExecutionEnvironment:
    def read_file(self, path: str, offset: Optional[int], limit: Optional[int]) -> str:
        raise NotImplementedError

    def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError

    def file_exists(self, path: str) -> bool:
        raise NotImplementedError

    def list_directory(self, path: str, depth: int) -> List[DirEntry]:
        raise NotImplementedError

    def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: Optional[str],
        env_vars: Optional[Dict[str, str]],
    ) -> ExecResult:
        raise NotImplementedError

    def grep(self, pattern: str, path: str, options: Dict[str, object]) -> str:
        raise NotImplementedError

    def glob(self, pattern: str, path: str) -> List[str]:
        raise NotImplementedError

    def initialize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None

    def working_directory(self) -> str:
        raise NotImplementedError

    def platform(self) -> str:
        raise NotImplementedError

    def os_version(self) -> str:
        raise NotImplementedError


class ScopedExecutionEnvironment(ExecutionEnvironment):
    def __init__(self, inner: ExecutionEnvironment, working_dir: str) -> None:
        self._inner = inner
        self._working_dir = os.path.abspath(working_dir)

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self._working_dir, path))

    def read_file(self, path: str, offset: Optional[int], limit: Optional[int]) -> str:
        return self._inner.read_file(self._resolve(path), offset, limit)

    def write_file(self, path: str, content: str) -> None:
        self._inner.write_file(self._resolve(path), content)

    def file_exists(self, path: str) -> bool:
        return self._inner.file_exists(self._resolve(path))

    def list_directory(self, path: str, depth: int) -> List[DirEntry]:
        return self._inner.list_directory(self._resolve(path), depth)

    def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: Optional[str],
        env_vars: Optional[Dict[str, str]],
    ) -> ExecResult:
        return self._inner.exec_command(command, timeout_ms, self._working_dir, env_vars)

    def grep(self, pattern: str, path: str, options: Dict[str, object]) -> str:
        return self._inner.grep(pattern, self._resolve(path), options)

    def glob(self, pattern: str, path: str) -> List[str]:
        return self._inner.glob(pattern, self._resolve(path))

    def initialize(self) -> None:
        return self._inner.initialize()

    def cleanup(self) -> None:
        return self._inner.cleanup()

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        return self._inner.platform()

    def os_version(self) -> str:
        return self._inner.os_version()


class LocalExecutionEnvironment(ExecutionEnvironment):
    def __init__(self, working_dir: str, env_policy: str = "core") -> None:
        self._working_dir = os.path.abspath(working_dir)
        self._env_policy = env_policy
        self._running_pgroups: set[int] = set()

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        system = platform_module.system().lower()
        if system.startswith("darwin"):
            return "darwin"
        if system.startswith("windows"):
            return "windows"
        if system.startswith("linux"):
            return "linux"
        return system

    def os_version(self) -> str:
        return platform_module.platform()

    def _filtered_env(self, extra: Optional[Dict[str, str]]) -> Dict[str, str]:
        core_keys = {
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "LANG",
            "TERM",
            "TMPDIR",
            "GOPATH",
            "CARGO_HOME",
            "NVM_DIR",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "UV_CACHE_DIR",
            "UV_PROJECT_ENVIRONMENT",
        }
        blacklist = re.compile(r".*_(API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)$", re.I)

        if self._env_policy == "none":
            env = {}
        elif self._env_policy == "all":
            env = dict(os.environ)
        else:
            env = {k: v for k, v in os.environ.items() if k in core_keys}

        env = {k: v for k, v in env.items() if not blacklist.match(k)}
        if extra:
            env.update(extra)
        return env

    def read_file(self, path: str, offset: Optional[int], limit: Optional[int]) -> str:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        with open(abs_path, "rb") as handle:
            data = handle.read()
        if b"\x00" in data:
            raise ValueError("Binary file detected")
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = max(0, (offset or 1) - 1)
        end = len(lines) if limit is None else min(len(lines), start + limit)
        selected = lines[start:end]
        result_lines = []
        for idx, line in enumerate(selected, start=start + 1):
            result_lines.append(f"{idx:04d} | {line}")
        return "\n".join(result_lines)

    def write_file(self, path: str, content: str) -> None:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def file_exists(self, path: str) -> bool:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        return os.path.exists(abs_path)

    def list_directory(self, path: str, depth: int) -> List[DirEntry]:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        entries: List[DirEntry] = []
        for root, dirs, files in os.walk(abs_path):
            rel_root = os.path.relpath(root, abs_path)
            current_depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            if current_depth > depth:
                dirs[:] = []
                continue
            for name in dirs:
                entry_name = name if rel_root == "." else os.path.join(rel_root, name)
                entries.append(DirEntry(name=entry_name, is_dir=True, size=None))
            for name in files:
                file_path = os.path.join(root, name)
                size = None
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    size = None
                entry_name = name if rel_root == "." else os.path.join(rel_root, name)
                entries.append(DirEntry(name=entry_name, is_dir=False, size=size))
        return entries

    def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: Optional[str],
        env_vars: Optional[Dict[str, str]],
    ) -> ExecResult:
        cwd = self._working_dir if working_dir is None else os.path.abspath(working_dir)
        start = time.time()

        creationflags = 0
        preexec_fn = None
        if os.name == "posix":
            preexec_fn = os.setsid
        else:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._filtered_env(env_vars),
            text=True,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
        )

        pgid = None
        if os.name == "posix":
            try:
                pgid = os.getpgid(process.pid)
                self._running_pgroups.add(pgid)
            except OSError:
                pgid = None

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_ms / 1000)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix" and pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(2)
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.terminate()
                time.sleep(2)
                process.kill()
            stdout, stderr = process.communicate()
        finally:
            if pgid is not None:
                self._running_pgroups.discard(pgid)

        duration_ms = int((time.time() - start) * 1000)
        return ExecResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    def _rg_available(self) -> bool:
        return shutil.which("rg") is not None

    def grep(self, pattern: str, path: str, options: Dict[str, object]) -> str:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        case_insensitive = bool(options.get("case_insensitive", False))
        max_results_value = options.get("max_results", 100)
        if isinstance(max_results_value, (int, float, str)):
            max_results = int(max_results_value)
        else:
            max_results = 100
        glob_filter = options.get("glob_filter")

        if self._rg_available():
            command = ["rg", "--no-heading", "--line-number", "--max-count", str(max_results)]
            if case_insensitive:
                command.append("-i")
            if glob_filter:
                command.extend(["--glob", str(glob_filter)])
            command.append(pattern)
            command.append(abs_path)
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._working_dir,
            )
            return (result.stdout or "") + (result.stderr or "")

        import re as re_module

        matches: List[str] = []
        flags = re_module.IGNORECASE if case_insensitive else 0
        regex = re_module.compile(pattern, flags)
        for root, _, files in os.walk(abs_path):
            for name in files:
                if glob_filter and not fnmatch.fnmatch(name, str(glob_filter)):
                    continue
                file_path = os.path.join(root, name)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                        for idx, line in enumerate(handle, start=1):
                            if regex.search(line):
                                matches.append(f"{file_path}:{idx}:{line.rstrip()}" )
                                if len(matches) >= max_results:
                                    return "\n".join(matches)
                except OSError:
                    continue
        return "\n".join(matches)

    def glob(self, pattern: str, path: str) -> List[str]:
        abs_path = os.path.abspath(os.path.join(self._working_dir, path))
        import glob as glob_module

        matches = glob_module.glob(os.path.join(abs_path, pattern), recursive=True)
        matches = [os.path.abspath(match) for match in matches]
        matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        return matches

    def cleanup(self) -> None:
        if os.name != "posix":
            return
        for pgid in list(self._running_pgroups):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                continue
        time.sleep(1)
        for pgid in list(self._running_pgroups):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                continue
        self._running_pgroups.clear()
