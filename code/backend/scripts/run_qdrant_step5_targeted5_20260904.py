#!/usr/bin/env python3
"""Prepare one-shot Step 5 validation for the five remaining RAG cases.

This launcher owns only an isolated backend process and new report/log files.
It never mutates Qdrant, never retries the evaluator, and never selects the
full test set.  The script is intentionally not executed by the offline
preparation checks in this task.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
REPORTS_DIR = BACKEND_DIR / "reports"
TESTSET_PATH = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"
QDRANT_URL = "http://127.0.0.1:16333"
COLLECTION = "kb_full_codex_20260830_24c7407e"
PORT = 18080
HUMAN_REPORT_PATH = REPORTS_DIR / "Qdrant_Step5修复后五题验收_20260904.md"

CASE_IDS: tuple[str, ...] = (
    "real_guangdao_2025_h1_revenue",
    "real_zhongguohe_2024_total_revenue_synonym",
    "real_huluwa_2025_h1_revenue_reason",
    "real_haiyue_2024_revenue_and_rd_cross_page",
    "real_dongshi_2024_vs_2023_revenue",
)

FIXED_ENVIRONMENT: dict[str, str] = {
    "KB_VECTOR_BACKEND": "qdrant",
    "KB_QDRANT_URL": QDRANT_URL,
    "KB_QDRANT_COLLECTION": COLLECTION,
    "KB_QDRANT_VECTOR_SIZE": "1024",
    "KB_QDRANT_CREATE_IF_MISSING": "false",
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


class Step5RunnerError(RuntimeError):
    """Infrastructure or launcher error, distinct from failed test cases."""


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_dotenv(path: Path) -> None:
    """Load missing environment values without printing any secret."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def _secret_values(environment: Mapping[str, str] | None = None) -> tuple[str, ...]:
    environment = environment or os.environ
    values: set[str] = set()
    for name, value in environment.items():
        if value and any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(value: Any, *, secrets: Sequence[str] = ()) -> str:
    text = str(value or "")
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"(?i)(\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|secret|password)\s*[:=]\s*)[^\s,;]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[redacted]",
        text,
    )
    return text


def safe_error(exc: BaseException) -> str:
    text = redact_text(exc, secrets=_secret_values())
    text = re.sub(r"\s+", " ", text).strip()
    return f"{type(exc).__name__}: {text[:320] or 'no details'}"


def log(path: Path, message: str) -> None:
    line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def port_listening(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def qdrant_preflight() -> dict[str, Any]:
    """Read collection status and vector size; never creates or mutates it."""
    endpoint = f"{QDRANT_URL}/collections/{quote(COLLECTION, safe='')}"
    request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("result") if isinstance(payload, dict) else None
    config = result.get("config", {}) if isinstance(result, dict) else {}
    params = config.get("params", {}) if isinstance(config, dict) else {}
    vectors = params.get("vectors", {}) if isinstance(params, dict) else {}
    size = vectors.get("size") if isinstance(vectors, dict) else None
    status = result.get("status") if isinstance(result, dict) else None
    if status and status != "green":
        raise Step5RunnerError(f"Qdrant collection status is {status!r}")
    if int(size or 0) != 1024:
        raise Step5RunnerError(f"Qdrant vector size is {size!r}, expected 1024")
    return {"status": status or "unspecified", "vector_size": int(size)}


def check_credentials() -> None:
    missing = [
        name
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID")
        if not os.getenv(name)
    ]
    if missing:
        raise Step5RunnerError("missing required credential/config names: " + ", ".join(missing))


def load_selected_cases() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(TESTSET_PATH.read_text(encoding="utf-8"))
    cases = {
        str(item.get("id")): item
        for item in (payload or {}).get("tests", [])
        if isinstance(item, dict) and item.get("id")
    }
    if len(CASE_IDS) != 5 or len(set(CASE_IDS)) != 5:
        raise Step5RunnerError("fixed case list is not exactly five unique ids")
    missing = [case_id for case_id in CASE_IDS if case_id not in cases]
    if missing:
        raise Step5RunnerError("testset missing selected case ids: " + ", ".join(missing))
    return {case_id: cases[case_id] for case_id in CASE_IDS}


def build_evaluator_command(
    python_executable: str | Path,
    report_path: Path,
    attempt_id: str,
) -> list[str]:
    """Build exactly one evaluator invocation containing only the fixed five IDs."""
    command = [
        str(python_executable),
        "scripts/evaluate_quality.py",
        "--rag",
        "testsets/rag_real_quality_v2.yaml",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--label",
        attempt_id,
        "--output",
        str(report_path),
    ]
    for case_id in CASE_IDS:
        command.extend(("--case-id", case_id))
    return command


def ensure_output_paths_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise Step5RunnerError("refusing to overwrite existing output: " + ", ".join(existing))


def wait_ready(process: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{PORT}/readyz"
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise Step5RunnerError(f"backend exited before readyz: {process.returncode}")
        try:
            request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and body.get("status") == "ok":
                return
            last_error = repr(body)
        except Exception as exc:  # noqa: BLE001
            last_error = safe_error(exc)
        time.sleep(0.5)
    raise Step5RunnerError(f"readyz timeout: {last_error}")


def terminate_backend(process: subprocess.Popen[bytes] | None) -> str | None:
    if process is None or process.poll() is not None:
        return None
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        return "kill fallback used"
    return None


def read_rag_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    reports = payload.get("reports", []) if isinstance(payload, dict) else []
    for report in reports:
        if isinstance(report, dict) and report.get("kind") == "rag":
            return report
    return None


def report_metrics(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    results = report.get("results") if isinstance(report.get("results"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    error_count = summary.get("error_count")
    if not isinstance(error_count, int):
        error_count = sum(1 for item in results if isinstance(item, dict) and item.get("error"))
    passed = report.get("passed")
    if not isinstance(passed, int):
        passed = sum(1 for item in results if isinstance(item, dict) and item.get("passed") is True)
    return {
        "passed": passed,
        "completed_total": report.get("completed_total", len(results)),
        "planned_total": report.get("planned_total", 5),
        "error_count": error_count,
        "run_status": report.get("run_status", "unavailable"),
        "unique_case_ids": len(
            {str(item.get("id")) for item in results if isinstance(item, dict) and item.get("id")}
        ),
    }


def fallback_report(attempt_id: str, error: str) -> dict[str, Any]:
    return {
        "kind": "rag",
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": attempt_id,
        "testset": str(TESTSET_PATH),
        "base_url": f"http://127.0.0.1:{PORT}",
        "selected_case_ids": list(CASE_IDS),
        "total": 0,
        "planned_total": 5,
        "completed_total": 0,
        "run_status": "failed",
        "passed": 0,
        "summary": {"error_count": 1},
        "results": [],
        "launcher_error": error,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_stable_metadata(
    attempt_id: str,
    report: Mapping[str, Any] | None,
    paths: Mapping[str, Path],
    *,
    port_released: bool,
    evaluation_exit_code: int | None,
    run_error: str | None,
) -> dict[str, Any]:
    metrics = report_metrics(report)
    return {
        "attempt_id": attempt_id,
        "selected_ids": list(CASE_IDS),
        "selected_ids_count": 5,
        "request_count": 5,
        "unique_case_ids": metrics["unique_case_ids"],
        "retries": 0,
        "full36_run": False,
        "passed": metrics["passed"],
        "passed_display": f"{metrics['passed']}/5",
        "error_count": metrics["error_count"],
        "run_status": metrics["run_status"],
        "completed_total": metrics["completed_total"],
        "planned_total": metrics["planned_total"],
        "port_18080_released": port_released,
        "evaluation_exit_code": evaluation_exit_code,
        "run_error": run_error,
        "report_path": str(paths["report"]),
        "lineage_path": str(paths["lineage"]),
        "runner_log": str(paths["runner_log"]),
        "backend_stdout": str(paths["backend_stdout"]),
        "backend_stderr": str(paths["backend_stderr"]),
    }


def write_human_report(
    path: Path,
    stable: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> None:
    if path.exists():
        raise Step5RunnerError(f"refusing to overwrite human report: {path}")
    rows = []
    by_id = {
        str(item.get("id")): item
        for item in (report or {}).get("results", [])
        if isinstance(item, dict) and item.get("id")
    }
    for case_id in CASE_IDS:
        item = by_id.get(case_id, {})
        answer = re.sub(r"\s+", " ", str(item.get("answer", ""))).strip()
        rows.append(
            f"| `{case_id}` | {item.get('passed', 'unavailable')} | "
            f"{item.get('elapsed_s', 'unavailable')} | {answer[:160]} |"
        )
    lines = [
        "# Qdrant Step 5 修复后五题验收",
        "",
        f"- attempt_id: `{stable['attempt_id']}`",
        f"- selected_ids=5, request_count=5, unique_case_ids={stable['unique_case_ids']}, "
        f"retries=0, full36_run=false, port_18080_released={str(stable['port_18080_released']).lower()}",
        f"- run_status: `{stable['run_status']}`; completed={stable['completed_total']}/{stable['planned_total']}; "
        f"passed={stable['passed_display']}; error_count={stable['error_count']}",
        f"- evaluator_exit_code: `{stable['evaluation_exit_code']}`; run_error: `{stable['run_error'] or ''}`",
        "",
        "| case_id | passed | elapsed_s | answer 摘要 |",
        "|---|---:|---:|---|",
        *rows,
        "",
        f"- machine report: `{stable['report_path']}`",
        f"- lineage: `{stable['lineage_path']}`",
        f"- runner/backend logs: `{stable['runner_log']}`, `{stable['backend_stdout']}`, `{stable['backend_stderr']}`",
        "- 本次启动器只允许五题一次性验收；不自动重试、不运行完整36题、不写入Qdrant。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    attempt_id = f"qdrant-step5-targeted5-20260904-{utc_stamp()}-{uuid.uuid4().hex[:10]}"
    paths = {
        "report": REPORTS_DIR / f"{attempt_id}.json",
        "lineage": REPORTS_DIR / f"{attempt_id}.lineage.json",
        "runner_log": REPORTS_DIR / f"{attempt_id}.runner.log",
        "backend_stdout": REPORTS_DIR / f"{attempt_id}.backend.stdout.log",
        "backend_stderr": REPORTS_DIR / f"{attempt_id}.backend.stderr.log",
    }
    ensure_output_paths_absent([*paths.values(), HUMAN_REPORT_PATH])
    paths["runner_log"].parent.mkdir(parents=True, exist_ok=True)
    paths["runner_log"].touch()
    log(paths["runner_log"], f"attempt_id={attempt_id}")
    log(paths["runner_log"], "selected_ids=5; request_count=5; retries=0; full36_run=false")
    log(paths["runner_log"], f"collection={COLLECTION}; qdrant={QDRANT_URL}; port={PORT}")

    process: subprocess.Popen[bytes] | None = None
    stdout_handle = None
    stderr_handle = None
    evaluation_exit_code: int | None = None
    run_error: str | None = None
    backend_ready = False
    qdrant_info: dict[str, Any] = {}
    report: dict[str, Any] | None = None
    try:
        selected_cases = load_selected_cases()
        if port_listening():
            raise Step5RunnerError(f"port {PORT} is already in use")
        docker = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if docker.returncode != 0:
            raise Step5RunnerError("Docker preflight failed")
        log(paths["runner_log"], "Docker preflight passed; version value not retained")
        qdrant_info = qdrant_preflight()
        log(
            paths["runner_log"],
            f"Qdrant preflight passed; status={qdrant_info['status']}; vector_size=1024",
        )
        check_credentials()
        log(paths["runner_log"], "LLM credential/config names present; values not printed")

        environment = os.environ.copy()
        environment.update(FIXED_ENVIRONMENT)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(BACKEND_DIR / "src"), environment.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        stdout_handle = paths["backend_stdout"].open("wb")
        stderr_handle = paths["backend_stderr"].open("wb")
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
                str(PORT),
            ],
            cwd=str(BACKEND_DIR),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        wait_ready(process)
        backend_ready = True
        log(paths["runner_log"], f"isolated backend readyz=ok; pid={process.pid}")

        command = build_evaluator_command(sys.executable, paths["report"], attempt_id)
        if command.count("--case-id") != 5 or set(command[command.index("--case-id") + 1 :: 2]) != set(CASE_IDS):
            raise Step5RunnerError("evaluator command is not exactly the fixed five-case invocation")
        log(paths["runner_log"], "evaluator invocation contains exactly five --case-id values; no retry option")
        completed = subprocess.run(
            command,
            cwd=str(BACKEND_DIR),
            env=environment,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        evaluation_exit_code = completed.returncode
        secrets = _secret_values(environment)
        output = redact_text(completed.stdout, secrets=secrets)
        error_output = redact_text(completed.stderr, secrets=secrets)
        if output:
            with paths["runner_log"].open("a", encoding="utf-8") as handle:
                handle.write(output)
        if error_output:
            with paths["runner_log"].open("a", encoding="utf-8") as handle:
                handle.write(error_output)
        log(paths["runner_log"], f"evaluator_exit={evaluation_exit_code}")
        report = read_rag_report(paths["report"])
        if report is None:
            raise Step5RunnerError("evaluator completed without a readable RAG report")
        if set(selected_cases) != set(CASE_IDS):
            raise Step5RunnerError("selected case validation drifted during run")
    except Exception as exc:  # noqa: BLE001
        run_error = safe_error(exc)
        log(paths["runner_log"], f"failure={run_error}")
    finally:
        termination = terminate_backend(process)
        if termination:
            log(paths["runner_log"], termination)
        released = not port_listening()
        log(paths["runner_log"], f"port_18080_released={str(released).lower()}")
        for stream in (stdout_handle, stderr_handle):
            if stream:
                stream.close()

    if report is None:
        report = fallback_report(attempt_id, run_error or "no readable evaluator report")
        write_json(paths["report"], {"reports": [report]})
    released = not port_listening()
    stable = build_stable_metadata(
        attempt_id,
        report,
        paths,
        port_released=released,
        evaluation_exit_code=evaluation_exit_code,
        run_error=run_error,
    )
    lineage = {
        **stable,
        "collection": COLLECTION,
        "qdrant_url": QDRANT_URL,
        "testset": str(TESTSET_PATH),
        "environment": dict(FIXED_ENVIRONMENT),
        "qdrant_preflight": qdrant_info,
        "backend_readyz": backend_ready,
        "stable_completion_marker": (
            f"selected_ids=5, request_count=5, unique_case_ids={stable['unique_case_ids']}, "
            f"retries=0, full36_run=false, port_18080_released={str(released).lower()}, "
            f"passed={stable['passed_display']}, error_count={stable['error_count']}"
        ),
    }
    write_json(paths["lineage"], lineage)
    write_human_report(HUMAN_REPORT_PATH, stable, report)
    log(
        paths["runner_log"],
        "stable completion marker: " + lineage["stable_completion_marker"],
    )
    # Exit 0 means the one-shot run produced a report, including evaluator
    # semantic failures (evaluate_quality uses exit 1 for failed cases).
    return 0 if report and run_error is None and evaluation_exit_code in (0, 1) else 2


if __name__ == "__main__":
    raise SystemExit(main())
