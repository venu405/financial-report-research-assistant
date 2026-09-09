#!/usr/bin/env python3
"""Launch one isolated backend child without PowerShell environment handling.

The helper intentionally does not wait for uvicorn.  It validates its inputs,
starts the child with Python's inherited environment, prints only the PID, and
leaves readiness polling and cleanup to the PowerShell runner.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


class LaunchError(RuntimeError):
    """Invalid launch input or failure to create the child process."""


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc or "").split())
    return text[:300] if text else "no details"


def _resolved_log_path(value: str, *, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LaunchError(f"{description} must be an absolute path")
    path = path.resolve()
    if not path.parent.is_dir():
        raise LaunchError(f"{description} parent directory does not exist")
    if path.exists():
        raise LaunchError(f"{description} already exists; refusing overwrite")
    return path


def _validate_inputs(backend_dir: str, port: int, stdout_path: str, stderr_path: str) -> tuple[Path, Path, Path]:
    if port < 1 or port > 65535 or port == 8000:
        raise LaunchError("port must be between 1 and 65535 and cannot be 8000")
    backend = Path(backend_dir).expanduser()
    if not backend.is_absolute():
        raise LaunchError("backend-dir must be an absolute path")
    backend = backend.resolve()
    if not backend.is_dir() or not (backend / "src").is_dir() or not (backend / "src" / "main.py").is_file():
        raise LaunchError("backend-dir must contain src/main.py")
    stdout = _resolved_log_path(stdout_path, description="stdout log")
    stderr = _resolved_log_path(stderr_path, description="stderr log")
    if stdout == stderr:
        raise LaunchError("stdout and stderr logs must be different paths")
    return backend, stdout, stderr


def launch(backend_dir: str, port: int, stdout_path: str, stderr_path: str) -> int:
    backend, stdout_path_obj, stderr_path_obj = _validate_inputs(
        backend_dir, port, stdout_path, stderr_path
    )
    stdout_stream = None
    stderr_stream = None
    try:
        # xb makes the no-overwrite contract atomic at the file-open boundary.
        stdout_stream = stdout_path_obj.open("xb")
        stderr_stream = stderr_path_obj.open("xb")
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--app-dir",
                "src",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(backend),
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            creationflags=creationflags,
        )
        return int(process.pid)
    finally:
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch an isolated uvicorn backend and print only its PID")
    parser.add_argument("--backend-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        pid = launch(args.backend_dir, args.port, args.stdout, args.stderr)
    except (LaunchError, OSError, ValueError) as exc:
        print(f"LAUNCH_ERROR: {_safe_error(exc)}", file=sys.stderr)
        return 2
    print(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
