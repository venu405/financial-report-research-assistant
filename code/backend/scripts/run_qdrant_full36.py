#!/usr/bin/env python3
"""Run the isolated 36-case RAG evaluation against the completed Qdrant run.

The launcher deliberately owns only the temporary FastAPI process and the new
evaluation output.  It does not repair, verify, hash-audit, or mutate the
Qdrant collection.  Existing LLM/Ollama settings are inherited unchanged;
the five Qdrant selector variables are the only backend settings imposed by
this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_COLLECTION = "kb_full_codex_20260830_24c7407e"
DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_RUN_DIR = Path(
    "migration-record/2026-08-29-qdrant-full-ingest/runs/"
    "full-20260830-codex-v1"
)
DEFAULT_TESTSET = Path("code/backend/testsets/rag_real_quality_v2.yaml")
DEFAULT_PORT = 18080
DEFAULT_STARTUP_TIMEOUT = 60.0
T1_COLLECTION = "qdrant_preflight_20260828_141124_ca47dc59"
EXPECTED_COMPLETED_FILES = 185
EXPECTED_FAILED_FILES = 0
EXPECTED_REMAINING_FILES = 0
TESTSET_ARGUMENT = "testsets/rag_real_quality_v2.yaml"
EVALUATOR_ARGUMENT = "scripts/evaluate_quality.py"

# ---------------------------------------------------------------------------
# 正式检索配置（必须与定向验证脚本 run_qdrant_targeted8_accuracy_20260902.ps1
# 的 FixedConfig 保持一致）。
#
# 背景（2026-09-03 定向6/6 与全量36题 20/36 不一致归因）：
# 本脚本原先只强制 Qdrant 选择器（backend/collection/url），其余配置全部依赖
# 父进程环境继承。而 `KB_STRUCTURED_FIN_ROUTE` 在 retriever.py 中**默认关闭**，
# 一旦父进程没有该变量，全量评测就在"财务结构化路由关闭"的状态下运行：
#     · 文档级召回正常（recall@1=1.0）
#     · 但装着正确数字的那个块进不了 top-K（page_recall=0.0）
#     · 数值核验找不到证据 → 安全拒答
# 定向脚本显式设了 KB_STRUCTURED_FIN_ROUTE=1，所以同样的题定向能过、全量拒答。
#
# 这里把这些"项目已确认的正式检索参数"固化为常量，避免全量链路再次漏传。
# 只覆盖通用检索/路由/重排配置，不含任何公司、题目或答案相关的硬编码。
# ---------------------------------------------------------------------------
FIXED_RETRIEVAL_ENVIRONMENT: dict[str, str] = {
    "KB_EMBEDDING_MODE": "bge_m3",
    "KB_EMBEDDING_MODEL": "bge-m3",
    "KB_OLLAMA_HOST": "http://127.0.0.1:11434",
    "KB_TOP_K": "5",
    "KB_RECALL_K": "20",
    "KB_MAX_HITS_PER_DOC": "12",
    "KB_STRUCTURED_FIN_ROUTE": "1",
    "KB_FIN_ROUTE_TOP_N": "8",
    "KB_RERANK_MODE": "llm",
    "KB_ANSWERABILITY_SCORE": "0.5",
}


class RunnerError(RuntimeError):
    """A launcher or infrastructure failure, distinct from failed test cases."""


class PreflightError(RunnerError):
    """The frozen Qdrant run is not in the exact state required for evaluation."""


@dataclass(frozen=True)
class GitMetadata:
    """The only Git state written to lineage; no diff contents are retained."""

    head: str | None
    dirty: bool | None


@dataclass(frozen=True)
class PreflightMetadata:
    """Validated, non-secret identity copied into the lineage sidecar."""

    checkpoint: dict[str, Any]
    verify_report: dict[str, Any]
    manifest: dict[str, Any]
    run_id: str
    source_manifest_sha256: str
    embedding_model: str


@dataclass(frozen=True)
class RunConfig:
    """Resolved paths and CLI overrides for one isolated evaluation run."""

    project_root: Path
    backend_dir: Path
    checkpoint_path: Path
    verify_report_path: Path
    manifest_path: Path
    testset_path: Path
    output_path: Path
    lineage_path: Path
    port: int = DEFAULT_PORT
    user_id: str | None = None
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    collection: str = DEFAULT_COLLECTION
    qdrant_url: str = DEFAULT_QDRANT_URL
    vector_size: int = DEFAULT_VECTOR_SIZE


@dataclass(frozen=True)
class RunResult:
    """Result of the launcher, including the evaluator's semantic exit code."""

    output_path: Path
    lineage_path: Path
    runner_status: str
    runner_error: str | None
    evaluation_exit_code: int | None


def utc_now() -> str:
    """Return a compact UTC timestamp suitable for lineage and filenames."""

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _safe_error(exc: BaseException) -> str:
    """Reduce an exception to a short, secret-free diagnostic."""

    text = redact_sensitive_text(str(exc or ""))
    text = re.sub(r"\s+", " ", text).strip().replace("\x00", "")
    if len(text) > 320:
        text = text[:317] + "..."
    return f"{type(exc).__name__}: {text or 'no details'}"


def _secret_values(environment: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Collect likely secret values without ever serializing the environment."""

    environment = environment or os.environ
    values: set[str] = set()
    for name, value in environment.items():
        upper_name = name.upper()
        if not value or len(value) < 3:
            continue
        if any(part in upper_name for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def redact_sensitive_text(
    value: str | None,
    *,
    secrets: Sequence[str] = (),
) -> str:
    """Redact known values and common key/token forms from captured text."""

    text = str(value or "")
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)(\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|secret|password)"
        r"\s*[:=]\s*)[^\s,;]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|secret|password)=)"
        r"[^&#\s]+",
        r"\1[redacted]",
        text,
    )
    return text


def summarize_text(
    value: str | None,
    *,
    secrets: Sequence[str] = (),
    limit: int = 4000,
) -> str:
    """Keep a bounded head/tail summary for the lineage sidecar."""

    text = redact_sensitive_text(value, secrets=secrets).replace("\x00", "")
    if len(text) <= limit:
        return text
    head = max(1, limit // 3)
    tail = max(1, limit - head - 45)
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def sanitize_url(value: str) -> str:
    """Keep only scheme, host, and port; never retain URL credentials/query."""

    try:
        parsed = urlsplit(value or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return ""


def _canonical_url(value: str) -> str:
    """Canonicalize a URL for comparing non-secret run metadata."""

    return sanitize_url(value).rstrip("/").lower()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"{description} cannot be read: {_safe_error(exc)}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"{description} must contain a JSON object")
    return payload


def _require_value(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    description: str,
) -> Any:
    actual = mapping.get(key)
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise PreflightError(
            f"{description}.{key} mismatch: expected {expected!r}, got {actual!r}"
        )
    return actual


def _require_nonempty_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"{description}.{key} must be a non-empty string")
    return value.strip()


def _nested_mapping(mapping: Mapping[str, Any], key: str, *, description: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise PreflightError(f"{description}.{key} must be an object")
    return value


def validate_preflight(
    project_root: Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    run_dir: Path = DEFAULT_RUN_DIR,
    testset: Path = DEFAULT_TESTSET,
) -> PreflightMetadata:
    """Validate only the completed run identity needed to start the evaluator.

    This function reads existing JSON records and does not call Qdrant, rerun
    verification, or recalculate the source-manifest hashes.
    """

    project_root = project_root.resolve()
    resolved_run_dir = run_dir if run_dir.is_absolute() else project_root / run_dir
    resolved_testset = testset if testset.is_absolute() else project_root / testset
    checkpoint_path = resolved_run_dir / "checkpoint.json"
    verify_report_path = resolved_run_dir / "verify-report.json"
    manifest_path = resolved_run_dir / "run-manifest.json"

    checkpoint = _read_json(checkpoint_path, "checkpoint")
    verify_report = _read_json(verify_report_path, "verify-report")
    manifest = _read_json(manifest_path, "run-manifest")

    _require_value(checkpoint, "status", "completed", description="checkpoint")
    _require_value(
        checkpoint,
        "completed_files",
        EXPECTED_COMPLETED_FILES,
        description="checkpoint",
    )
    _require_value(
        checkpoint,
        "failed_files",
        EXPECTED_FAILED_FILES,
        description="checkpoint",
    )
    _require_value(
        checkpoint,
        "remaining_files",
        EXPECTED_REMAINING_FILES,
        description="checkpoint",
    )
    _require_value(checkpoint, "collection_name", collection, description="checkpoint")

    _require_value(verify_report, "status", "verified", description="verify-report")
    _require_value(verify_report, "collection", collection, description="verify-report")
    _require_value(verify_report, "vector_size", vector_size, description="verify-report")

    _require_value(manifest, "status", "completed", description="run-manifest")
    _require_value(manifest, "collection_name", collection, description="run-manifest")
    _require_value(manifest, "collection_base", collection, description="run-manifest")
    _require_value(manifest, "embedding_dimension", vector_size, description="run-manifest")
    _require_value(manifest, "qdrant_url", qdrant_url, description="run-manifest")
    config = _nested_mapping(manifest, "config", description="run-manifest")
    _require_value(config, "collection", collection, description="run-manifest.config")
    _require_value(
        config,
        "embedding_dimension",
        vector_size,
        description="run-manifest.config",
    )
    _require_value(config, "qdrant_url", qdrant_url, description="run-manifest.config")

    # These cross-record checks prevent a valid report from being paired with
    # a different run or collection while avoiding any new remote verification.
    run_id = _require_nonempty_string(checkpoint, "run_id", description="checkpoint")
    _require_value(manifest, "run_id", run_id, description="run-manifest")
    _require_value(verify_report, "run_id", run_id, description="verify-report")
    last_verification = checkpoint.get("last_verification")
    if isinstance(last_verification, Mapping):
        _require_value(
            last_verification,
            "status",
            "verified",
            description="checkpoint.last_verification",
        )
        _require_value(
            last_verification,
            "collection",
            collection,
            description="checkpoint.last_verification",
        )
        _require_value(
            last_verification,
            "vector_size",
            vector_size,
            description="checkpoint.last_verification",
        )

    t1_after = checkpoint.get("t1_after_verification")
    if not isinstance(t1_after, Mapping):
        raise PreflightError("checkpoint.t1_after_verification must be an object")
    _require_value(t1_after, "collection", T1_COLLECTION, description="checkpoint.t1_after_verification")
    _require_value(t1_after, "points", 100, description="checkpoint.t1_after_verification")
    _require_value(
        t1_after,
        "vector_size",
        vector_size,
        description="checkpoint.t1_after_verification",
    )

    if not resolved_testset.is_file():
        raise PreflightError(f"testset does not exist: {resolved_testset}")
    source_manifest_sha256 = _require_nonempty_string(
        manifest,
        "source_manifest_sha256",
        description="run-manifest",
    )
    embedding_model = _require_nonempty_string(
        manifest,
        "embedding_model",
        description="run-manifest",
    )
    _require_value(config, "embedding_model", embedding_model, description="run-manifest.config")

    if _canonical_url(str(manifest.get("qdrant_url"))) != _canonical_url(qdrant_url):
        raise PreflightError("run-manifest.qdrant_url is not the configured Qdrant URL")

    return PreflightMetadata(
        checkpoint=checkpoint,
        verify_report=verify_report,
        manifest=manifest,
        run_id=run_id,
        source_manifest_sha256=source_manifest_sha256,
        embedding_model=embedding_model,
    )


def sha256_file(path: Path) -> str:
    """Hash only the selected 36-case testset for lineage."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def get_git_metadata(project_root: Path) -> GitMetadata:
    """Read Git HEAD and dirty state without including the diff in lineage."""

    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return GitMetadata(head=None, dirty=None)
    head = head_result.stdout.strip() if head_result.returncode == 0 else None
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    return GitMetadata(head=head or None, dirty=dirty)


def build_backend_environment(
    backend_dir: Path,
    *,
    base_environment: Mapping[str, str] | None = None,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> dict[str, str]:
    """Copy the current process environment and impose the frozen run selectors.

    Two groups of variables are imposed on top of the inherited environment:

    1. Qdrant selectors (backend / collection / url / vector size).
    2. ``FIXED_RETRIEVAL_ENVIRONMENT`` —— the retrieval, routing and rerank
       settings the project has validated in targeted runs.

    Group 2 used to be left to environment inheritance, which silently disabled
    the structured financial route (``KB_STRUCTURED_FIN_ROUTE`` defaults to off)
    whenever the parent shell did not export it. See the comment on
    ``FIXED_RETRIEVAL_ENVIRONMENT`` for the full attribution.

    Everything else (notably LLM credentials such as ``LLM_API_KEY``) is still
    inherited from the parent process and never overridden here.
    """

    environment = dict(os.environ if base_environment is None else base_environment)
    source_dir = str((backend_dir / "src").resolve())
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        os.pathsep.join([source_dir, existing_pythonpath])
        if existing_pythonpath
        else source_dir
    )
    environment.update(
        {
            "KB_VECTOR_BACKEND": "qdrant",
            "KB_QDRANT_URL": qdrant_url,
            "KB_QDRANT_COLLECTION": collection,
            "KB_QDRANT_VECTOR_SIZE": str(vector_size),
            "KB_QDRANT_CREATE_IF_MISSING": "false",
        }
    )
    environment.update(FIXED_RETRIEVAL_ENVIRONMENT)
    return environment


def build_backend_command(port: int) -> list[str]:
    """Build the fixed isolated uvicorn command."""

    return [
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
    ]


def build_evaluation_command(config: RunConfig) -> list[str]:
    """Build the existing evaluator invocation from the backend working directory."""

    command = [
        sys.executable,
        EVALUATOR_ARGUMENT,
        "--rag",
        TESTSET_ARGUMENT,
        "--base-url",
        f"http://127.0.0.1:{config.port}",
    ]
    if config.user_id:
        command.extend(["--user-id", config.user_id])
    command.extend(
        [
            "--label",
            f"qdrant-full-185-{config.collection}",
            "--output",
            str(config.output_path),
        ]
    )
    return command


def redact_command(command: Sequence[str], *, user_id: str | None = None) -> list[str]:
    """Return an argument list safe to store in lineage."""

    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("[redacted]")
            redact_next = False
            continue
        if argument == "--user-id":
            redacted.append(argument)
            redact_next = True
            continue
        if argument.startswith("--user-id="):
            redacted.append("--user-id=[redacted]")
            continue
        if user_id and argument == user_id:
            redacted.append("[redacted]")
            continue
        redacted.append(redact_sensitive_text(argument, secrets=_secret_values()))
    return redacted


def lineage_path_for(output_path: Path) -> Path:
    """Return ``report.json`` -> ``report.lineage.json``."""

    return output_path.with_suffix(".lineage.json")


def _candidate_is_taken(path: Path) -> bool:
    return path.exists() or lineage_path_for(path).exists()


def default_output_path(
    reports_dir: Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    now: dt.datetime | None = None,
) -> Path:
    """Create a collision-free timestamped report name without touching disk."""

    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = reports_dir / f"qdrant-full-185-{collection}-{stamp}.json"
    candidate = base
    suffix = 1
    while _candidate_is_taken(candidate):
        candidate = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    return candidate


def resolve_output_path(
    raw_output: str | Path | None,
    *,
    project_root: Path,
    backend_dir: Path,
    collection: str = DEFAULT_COLLECTION,
) -> Path:
    """Resolve output and reject explicit collisions rather than overwriting."""

    if raw_output is None:
        return default_output_path(backend_dir / "reports", collection=collection)
    output = Path(raw_output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if _candidate_is_taken(output):
        raise RunnerError(f"output or lineage already exists: {output}")
    return output


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write UTF-8 JSON atomically, retaining no temporary file on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class _StreamCapture:
    """Drain a child pipe in a thread so a chatty uvicorn cannot deadlock."""

    def __init__(self, *, limit: int = 200_000):
        self._limit = limit
        self._parts: list[str] = []
        self._length = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, stream: Any) -> None:
        if stream is None:
            return
        self._thread = threading.Thread(target=self._consume, args=(stream,), daemon=True)
        self._thread.start()

    def _consume(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.readline()
                if chunk in ("", b""):
                    break
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="replace")
                chunk = str(chunk)
                with self._lock:
                    if self._length < self._limit:
                        remaining = self._limit - self._length
                        self._parts.append(chunk[:remaining])
                        self._length += min(len(chunk), remaining)
        except (OSError, ValueError):
            # The pipe can close concurrently with terminate(); the lineage
            # still contains everything that was drained before that point.
            return

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def text(self) -> str:
        with self._lock:
            return "".join(self._parts)


def wait_for_ready(
    process: Any,
    base_url: str,
    timeout: float,
    *,
    opener: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> None:
    """Poll /readyz until it returns a JSON ``status=ok`` or timeout."""

    opener = opener or urlopen
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    deadline = monotonic_fn() + timeout
    endpoint = f"{base_url.rstrip('/')}/readyz"
    last_error = "no response"

    while monotonic_fn() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RunnerError(f"isolated backend exited before readyz (exit {return_code})")
        try:
            request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
            remaining = max(0.1, deadline - monotonic_fn())
            with opener(request, timeout=min(5.0, remaining)) as response:
                status_code = getattr(response, "status", getattr(response, "code", 200))
                body = json.loads(response.read().decode("utf-8"))
            if status_code == 200 and isinstance(body, Mapping) and body.get("status") == "ok":
                return
            last_error = f"readyz returned status={status_code!r} body_status={body.get('status')!r}"
        except Exception as exc:
            last_error = _safe_error(exc)
        sleep_for = min(0.5, max(0.0, deadline - monotonic_fn()))
        if sleep_for:
            sleep_fn(sleep_for)

    raise RunnerError(f"readyz timeout after {timeout:g}s: {last_error}")


def terminate_process(process: Any, *, timeout: float = 10.0) -> str | None:
    """Terminate a child gently, with a kill fallback so no uvicorn is orphaned."""

    try:
        if process.poll() is not None:
            return None
    except Exception as exc:
        return _safe_error(exc)

    try:
        process.terminate()
    except Exception as exc:
        terminate_error = _safe_error(exc)
        try:
            process.kill()
            process.wait(timeout=timeout)
            return f"terminate failed; kill fallback used: {terminate_error}"
        except Exception as fallback_exc:
            return f"terminate and kill failed: {terminate_error}; {_safe_error(fallback_exc)}"

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=timeout)
        except Exception as exc:
            return f"terminate timed out and kill failed: {_safe_error(exc)}"
    except Exception as exc:
        return _safe_error(exc)
    return None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_lineage(
    *,
    config: RunConfig,
    preflight: PreflightMetadata,
    testset_sha256: str,
    git: GitMetadata,
    evaluation_command: Sequence[str],
    started_at: str,
    ended_at: str,
    evaluation_started_at: str | None,
    evaluation_exit_code: int | None,
    runner_status: str,
    runner_error: str | None,
    evaluation_stdout: str,
    evaluation_stderr: str,
    backend_stdout: str,
    backend_stderr: str,
) -> dict[str, Any]:
    """Build a lineage object containing identity and bounded diagnostics."""

    secrets = list(_secret_values())
    if config.user_id:
        secrets.append(config.user_id)
    return {
        "schema_version": 1,
        "collection": config.collection,
        "qdrant_url": sanitize_url(config.qdrant_url),
        "vector_size": config.vector_size,
        "checkpoint": {
            "path": _relative_path(config.checkpoint_path, config.project_root),
            "status": preflight.checkpoint.get("status"),
            "completed_files": preflight.checkpoint.get("completed_files"),
            "failed_files": preflight.checkpoint.get("failed_files"),
            "remaining_files": preflight.checkpoint.get("remaining_files"),
        },
        "run_id": preflight.run_id,
        "source_manifest_sha256": preflight.source_manifest_sha256,
        "embedding_model": preflight.embedding_model,
        "testset": _relative_path(config.testset_path, config.project_root),
        "testset_sha256": testset_sha256,
        "git_head": git.head,
        "git_dirty": git.dirty,
        "evaluation_command": redact_command(evaluation_command, user_id=config.user_id),
        "started_at": started_at,
        "evaluation_started_at": evaluation_started_at,
        "ended_at": ended_at,
        "evaluation_exit_code": evaluation_exit_code,
        "runner_status": runner_status,
        "runner_error": redact_sensitive_text(runner_error, secrets=secrets) if runner_error else None,
        "report_path": _relative_path(config.output_path, config.project_root),
        "report_exists": config.output_path.is_file(),
        "backend": {
            "host": "127.0.0.1",
            "port": config.port,
            "base_url": f"http://127.0.0.1:{config.port}",
            "environment": {
                "KB_VECTOR_BACKEND": "qdrant",
                "KB_QDRANT_URL": sanitize_url(config.qdrant_url),
                "KB_QDRANT_COLLECTION": config.collection,
                "KB_QDRANT_VECTOR_SIZE": str(config.vector_size),
                "KB_QDRANT_CREATE_IF_MISSING": "false",
                # 记录实际生效的检索/路由/重排配置，便于复现与归因。
                # 只落盘非敏感的配置项（无 URL 凭据、无 API Key）。
                **{key: value for key, value in FIXED_RETRIEVAL_ENVIRONMENT.items()},
            },
        },
        "evaluation_stdout_summary": summarize_text(evaluation_stdout, secrets=secrets),
        "evaluation_stderr_summary": summarize_text(evaluation_stderr, secrets=secrets),
        "backend_stdout_summary": summarize_text(backend_stdout, secrets=secrets),
        "backend_stderr_summary": summarize_text(backend_stderr, secrets=secrets),
    }


def run(config: RunConfig, *, preflight: PreflightMetadata | None = None) -> RunResult:
    """Run one isolated evaluation and always clean up the backend child."""

    preflight = preflight or validate_preflight(
        config.project_root,
        collection=config.collection,
        qdrant_url=config.qdrant_url,
        vector_size=config.vector_size,
        run_dir=config.checkpoint_path.parent,
        testset=config.testset_path,
    )
    if _candidate_is_taken(config.output_path):
        raise RunnerError(f"output or lineage already exists: {config.output_path}")

    testset_sha256 = sha256_file(config.testset_path)
    evaluation_command = build_evaluation_command(config)
    started_at = utc_now()
    evaluation_started_at: str | None = None
    evaluation_exit_code: int | None = None
    evaluation_stdout = ""
    evaluation_stderr = ""
    runner_error: str | None = None
    backend_process: Any | None = None
    backend_stdout = _StreamCapture()
    backend_stderr = _StreamCapture()

    try:
        environment = build_backend_environment(
            config.backend_dir,
            collection=config.collection,
            qdrant_url=config.qdrant_url,
            vector_size=config.vector_size,
        )
        popen_kwargs: dict[str, Any] = {
            "cwd": config.backend_dir,
            "env": environment,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        backend_process = subprocess.Popen(build_backend_command(config.port), **popen_kwargs)
        backend_stdout.start(backend_process.stdout)
        backend_stderr.start(backend_process.stderr)

        wait_for_ready(
            backend_process,
            f"http://127.0.0.1:{config.port}",
            config.startup_timeout,
        )
        evaluation_started_at = utc_now()
        try:
            completed = subprocess.run(
                evaluation_command,
                cwd=config.backend_dir,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            evaluation_exit_code = int(completed.returncode)
            evaluation_stdout = str(getattr(completed, "stdout", "") or "")
            evaluation_stderr = str(getattr(completed, "stderr", "") or "")
        except KeyboardInterrupt:
            runner_error = "KeyboardInterrupt during evaluation"
        except Exception as exc:
            runner_error = f"evaluation process error: {_safe_error(exc)}"
    except KeyboardInterrupt:
        runner_error = "KeyboardInterrupt while starting isolated backend"
    except Exception as exc:
        runner_error = _safe_error(exc)
    finally:
        if backend_process is not None:
            cleanup_error = terminate_process(backend_process)
            if cleanup_error and not runner_error:
                runner_error = f"backend cleanup error: {cleanup_error}"
        backend_stdout.join()
        backend_stderr.join()

    ended_at = utc_now()
    backend_stdout_text = backend_stdout.text()
    backend_stderr_text = backend_stderr.text()
    if runner_error:
        runner_status = "runner_error"
    elif evaluation_exit_code == 0:
        runner_status = "completed"
    elif evaluation_exit_code == 1:
        runner_status = "evaluation_completed_with_failed_cases"
    elif evaluation_exit_code is None:
        runner_status = "runner_error"
    else:
        runner_status = f"evaluation_completed_exit_{evaluation_exit_code}"

    git = get_git_metadata(config.project_root)
    lineage = build_lineage(
        config=config,
        preflight=preflight,
        testset_sha256=testset_sha256,
        git=git,
        evaluation_command=evaluation_command,
        started_at=started_at,
        ended_at=ended_at,
        evaluation_started_at=evaluation_started_at,
        evaluation_exit_code=evaluation_exit_code,
        runner_status=runner_status,
        runner_error=runner_error,
        evaluation_stdout=evaluation_stdout,
        evaluation_stderr=evaluation_stderr,
        backend_stdout=backend_stdout_text,
        backend_stderr=backend_stderr_text,
    )
    try:
        _atomic_write_json(config.lineage_path, lineage)
    except Exception as exc:
        lineage_error = f"lineage write error: {_safe_error(exc)}"
        runner_error = f"{runner_error}; {lineage_error}" if runner_error else lineage_error
        runner_status = "runner_error"

    return RunResult(
        output_path=config.output_path,
        lineage_path=config.lineage_path,
        runner_status=runner_status,
        runner_error=runner_error,
        evaluation_exit_code=evaluation_exit_code,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start an isolated Qdrant-backed FastAPI process and run the existing 36-case RAG evaluation."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.port < 1 or args.port > 65535:
        raise RunnerError("port must be between 1 and 65535")
    if args.port == 8000:
        raise RunnerError("port 8000 belongs to the existing service and is not allowed")
    if args.startup_timeout <= 0:
        raise RunnerError("startup-timeout must be positive")
    output_path = resolve_output_path(
        args.output,
        project_root=PROJECT_ROOT,
        backend_dir=BACKEND_DIR,
    )
    return RunConfig(
        project_root=PROJECT_ROOT,
        backend_dir=BACKEND_DIR,
        checkpoint_path=PROJECT_ROOT / DEFAULT_RUN_DIR / "checkpoint.json",
        verify_report_path=PROJECT_ROOT / DEFAULT_RUN_DIR / "verify-report.json",
        manifest_path=PROJECT_ROOT / DEFAULT_RUN_DIR / "run-manifest.json",
        testset_path=PROJECT_ROOT / DEFAULT_TESTSET,
        output_path=output_path,
        lineage_path=lineage_path_for(output_path),
        port=args.port,
        user_id=args.user_id,
        startup_timeout=args.startup_timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        preflight = validate_preflight(
            config.project_root,
            collection=config.collection,
            qdrant_url=config.qdrant_url,
            vector_size=config.vector_size,
            run_dir=config.checkpoint_path.parent,
            testset=config.testset_path,
        )
        result = run(config, preflight=preflight)
    except RunnerError as exc:
        print(f"RUNNER_ERROR: {_safe_error(exc)}", file=sys.stderr)
        return 2

    print(f"report: {result.output_path}")
    print(f"lineage: {result.lineage_path}")
    if result.runner_error:
        print(f"RUNNER_ERROR: {result.runner_error}", file=sys.stderr)
        return 2
    if result.evaluation_exit_code == 1:
        print("evaluation completed with failed cases (exit 1); report preserved")
    elif result.evaluation_exit_code == 2:
        print("evaluation exited 2; report/lineage preserved")
    else:
        print(f"evaluation completed (exit {result.evaluation_exit_code})")
    return result.evaluation_exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
