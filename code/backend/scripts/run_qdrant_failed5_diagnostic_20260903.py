#!/usr/bin/env python3
"""Run exactly the five remaining cases once with the existing diagnostic trace."""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
REPORTS_DIR = BACKEND_DIR / "reports"
TESTSET_PATH = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"
COLLECTION = "kb_full_codex_20260830_24c7407e"
QDRANT_URL = "http://127.0.0.1:16333"
PORT = 18080
CASE_IDS = [
    "real_guangdao_2025_h1_revenue",
    "real_zhongguohe_2024_total_revenue_synonym",
    "real_huluwa_2025_h1_revenue_reason",
    "real_haiyue_2024_revenue_and_rd_cross_page",
    "real_dongshi_2024_vs_2023_revenue",
]
FIXED_ENV = {
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
STAGES = [
    "raw_recall",
    "structured_fin_route",
    "post_merge",
    "rerank_after_model",
    "post_protect",
    "generation_verification",
    "safe_fallback",
]


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_dotenv(path: Path) -> None:
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


def log(path: Path, message: str) -> None:
    line = f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def qdrant_preflight() -> dict[str, Any]:
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
        raise RuntimeError(f"Qdrant collection status is {status!r}")
    if int(size or 0) != 1024:
        raise RuntimeError(f"Qdrant vector size is {size!r}, expected 1024")
    return {"status": status or "unspecified", "vector_size": int(size)}


def check_credentials() -> None:
    missing = [name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID") if not os.getenv(name)]
    if missing:
        raise RuntimeError("missing required credential/config names: " + ", ".join(missing))


def wait_ready(process: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{PORT}/readyz"
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend exited before readyz: {process.returncode}")
        try:
            request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and body.get("status") == "ok":
                return
            last_error = repr(body)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)[:240]
        time.sleep(0.5)
    raise RuntimeError(f"readyz timeout: {last_error}")


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


def candidate_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "doc_id" in value or "chunk_id" in value:
            found.append(value)
        for child in value.values():
            found.extend(candidate_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(candidate_dicts(child))
    return found


def candidate_summary(stage: dict[str, Any]) -> dict[str, Any]:
    data = stage.get("data") if isinstance(stage, dict) else {}
    candidates = candidate_dicts(data)
    ids: list[str] = []
    details: list[dict[str, Any]] = []
    for item in candidates[:40]:
        chunk_id = str(item.get("chunk_id") or "")
        doc_id = str(item.get("doc_id") or "")
        if chunk_id or doc_id:
            ids.append(chunk_id or doc_id)
            details.append(
                {
                    key: item.get(key)
                    for key in (
                        "chunk_id", "doc_id", "kb_id", "doc_title", "page", "page_start",
                        "vec_score", "rerank_score", "rrf_score", "company", "report_period",
                        "section_path", "financial_metrics", "source", "reason",
                    )
                    if key in item
                }
            )
    return {"available": bool(data), "candidate_count": len(candidates), "ids": ids[:40], "details": details}


def target_pages(case: dict[str, Any]) -> set[str]:
    return {str(value) for value in case.get("expected_pages") or []}


def matching_target_pages(
    summary: dict[str, Any], pages: set[str], source_stem: str
) -> set[str] | None:
    if not pages:
        return set()
    if not summary.get("details"):
        return None
    matches: set[str] = set()
    for item in summary.get("details", []):
        title = str(item.get("doc_title") or "")
        if source_stem and source_stem not in title and title not in source_stem:
            continue
        page = str(item.get("page") or item.get("page_start") or "")
        if page in pages:
            matches.add(page)
    return matches


def build_human_report(
    path: Path,
    report: dict[str, Any] | None,
    traces: list[dict[str, Any]],
    stable: dict[str, Any],
    cases: dict[str, dict[str, Any]],
) -> None:
    by_question = {
        str(record.get("basic", {}).get("question", "")): record for record in traces
    }
    lines = [
        "# Qdrant 剩余5题分阶段诊断",
        "",
        f"- attempt_id: `{stable['attempt_id']}`",
        f"- selected_ids=5, request_count=5, unique_case_ids={stable['unique_case_ids']}, "
        f"retries=0, full36_run=false, port_18080_released={str(stable['port_18080_released']).lower()}",
        f"- report_status: `{(report or {}).get('run_status', 'unavailable')}`; "
        f"completed={((report or {}).get('completed_total', 'unavailable'))}/"
        f"{((report or {}).get('planned_total', 'unavailable'))}; "
        f"error_count={((report or {}).get('summary') or {}).get('error_count', 'unavailable')}",
        "- 本报告只记录现有链路已产生的追踪数据；不可观测阶段标记为 unavailable。",
        "",
        "## 逐题归因",
        "",
    ]
    result_by_id = {str(item.get("id")): item for item in (report or {}).get("results", [])}
    for case_id in CASE_IDS:
        case = cases.get(case_id, {})
        result = result_by_id.get(case_id, {})
        trace = by_question.get(str(case.get("question", "")), {})
        stage_map = {str(stage.get("stage")): stage for stage in trace.get("stages", [])}
        pages = target_pages(case)
        source_stem = Path(str((case.get("expected_sources") or [""])[0])).stem
        target_page_hits = {
            name: matching_target_pages(candidate_summary(stage_map[name]), pages, source_stem)
            if name in stage_map else None
            for name in STAGES
        }
        presence = {
            name: (bool(target_page_hits[name]) if target_page_hits[name] is not None else None)
            for name in STAGES
        }
        present = [name for name in STAGES if presence[name] is True]
        last = present[-1] if present else "unavailable"
        first_loss = "unavailable"
        if present:
            last_index = STAGES.index(last)
            for name in STAGES[last_index + 1 :]:
                if presence[name] is False:
                    first_loss = name
                    break
        if not trace:
            cause = "trace unavailable；无法区分路由、重排、保护或核验阶段"
        elif not present:
            cause = "目标页未在可观测候选快照中出现；具体首次丢失阶段需结合原始字段复核"
        elif first_loss == "unavailable":
            cause = "目标页在已观测阶段未显示后续丢失；需结合生成/核验数据判断"
        else:
            cause = f"目标页在 `{first_loss}` 前后首次不再出现在候选快照中"
        lines.extend(
            [
                f"### `{case_id}`",
                f"- passed={result.get('passed', 'unavailable')}; answer={str(result.get('answer', ''))[:240]}",
                f"- 正确证据最后出现阶段: `{last}`；首次消失阶段: `{first_loss}`。",
                f"- 目标页命中（按 expected source + page 严格匹配）: "
                f"{', '.join(sorted(set().union(*(target_page_hits[name] or set() for name in STAGES)))) or 'none'}；"
                f"期望页: {','.join(sorted(pages)) or 'unavailable'}。",
                f"- 最可能原因（推断）: {cause}。事实阶段覆盖: {','.join(name for name in STAGES if name in stage_map) or 'unavailable'}。",
                "- 阶段候选摘要:",
            ]
        )
        for name in STAGES:
            if name not in stage_map:
                lines.append(f"  - `{name}`: unavailable（该阶段未由现有 trace 记录）")
            else:
                summary = candidate_summary(stage_map[name])
                ids = ", ".join(summary["ids"][:8]) or "none"
                lines.append(f"  - `{name}`: candidates={summary['candidate_count']}; ids={ids}")
        lines.append("")
    lines.extend(
        [
            "## 产物与边界",
            "",
            f"- machine report: `{stable['report_path']}`",
            f"- trace JSONL: `{stable['trace_path']}`; lines={stable['trace_lines']}; unique_trace_id={stable['unique_trace_ids']}",
            f"- lineage: `{stable['lineage_path']}`",
            f"- runner/backend logs: `{stable['runner_log']}`, `{stable['backend_stdout']}`, `{stable['backend_stderr']}`",
            "- 本轮未运行完整36题、未重试、未写入Qdrant；任何缺失阶段均不以推断补齐。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    attempt_id = f"qdrant-failed5-diagnostic-20260903-{utc_stamp()}-{uuid.uuid4().hex[:10]}"
    report_path = REPORTS_DIR / f"{attempt_id}.json"
    trace_path = REPORTS_DIR / f"{attempt_id}.trace.jsonl"
    runner_log = REPORTS_DIR / f"{attempt_id}.runner.log"
    backend_stdout = REPORTS_DIR / f"{attempt_id}.backend.stdout.log"
    backend_stderr = REPORTS_DIR / f"{attempt_id}.backend.stderr.log"
    lineage_path = REPORTS_DIR / f"{attempt_id}.lineage.json"
    human_path = REPORTS_DIR / "Qdrant_剩余5题分阶段诊断_20260903.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_paths = [report_path, trace_path, runner_log, backend_stdout, backend_stderr, lineage_path, human_path]
    if any(path.exists() for path in all_paths):
        raise RuntimeError("one or more output paths already exist; refusing overwrite")
    runner_log.touch()
    log(runner_log, f"attempt_id={attempt_id}")
    log(runner_log, "selected_ids=5; request_count=5; retries=0; full36_run=false")
    log(runner_log, f"collection={COLLECTION}; qdrant={QDRANT_URL}; port={PORT}")
    process: subprocess.Popen[bytes] | None = None
    stdout_handle = None
    stderr_handle = None
    eval_exit: int | None = None
    run_error: str | None = None
    ready = False
    qdrant_info: dict[str, Any] = {}
    cases = {str(item.get("id")): item for item in yaml.safe_load(TESTSET_PATH.read_text(encoding="utf-8")).get("tests", [])}
    try:
        if len(CASE_IDS) != 5 or len(set(CASE_IDS)) != 5:
            raise RuntimeError("fixed case list is not exactly five unique ids")
        missing = [case_id for case_id in CASE_IDS if case_id not in cases]
        if missing:
            raise RuntimeError("testset missing case ids: " + ", ".join(missing))
        if port_listening(PORT):
            raise RuntimeError(f"port {PORT} is already in use")
        docker = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=10)
        if docker.returncode != 0:
            raise RuntimeError("Docker preflight failed")
        log(runner_log, "Docker preflight passed; version value not retained")
        qdrant_info = qdrant_preflight()
        log(runner_log, f"Qdrant preflight passed; status={qdrant_info['status']}; vector_size=1024")
        check_credentials()
        log(runner_log, "LLM credential/config names present; values not printed")
        env = os.environ.copy()
        env.update(FIXED_ENV)
        env["PYTHONPATH"] = os.pathsep.join([str(BACKEND_DIR / "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
        env["KB_DIAGNOSTIC_TRACE_ENABLED"] = "1"
        env["KB_DIAGNOSTIC_TRACE_PATH"] = str(trace_path)
        stdout_handle = backend_stdout.open("wb")
        stderr_handle = backend_stderr.open("wb")
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=str(BACKEND_DIR), env=env, stdout=stdout_handle, stderr=stderr_handle,
        )
        wait_ready(process)
        ready = True
        log(runner_log, f"isolated backend readyz=ok; pid={process.pid}")
        command = [sys.executable, "scripts/evaluate_quality.py", "--rag", "testsets/rag_real_quality_v2.yaml", "--base-url", f"http://127.0.0.1:{PORT}", "--label", attempt_id, "--output", str(report_path)]
        for case_id in CASE_IDS:
            command.extend(["--case-id", case_id])
        log(runner_log, "evaluator invocation contains exactly five --case-id values; no retry option")
        completed = subprocess.run(command, cwd=str(BACKEND_DIR), env=env, capture_output=True, text=True, timeout=1800)
        eval_exit = completed.returncode
        with runner_log.open("a", encoding="utf-8") as handle:
            handle.write(completed.stdout)
            handle.write(completed.stderr)
        log(runner_log, f"evaluator_exit={eval_exit}")
    except Exception as exc:  # noqa: BLE001
        run_error = f"{type(exc).__name__}: {str(exc)[:320]}"
        log(runner_log, f"failure={run_error}")
    finally:
        termination = terminate_backend(process)
        if termination:
            log(runner_log, termination)
        released = not port_listening(PORT)
        log(runner_log, f"port_18080_released={str(released).lower()}")
        for stream in (stdout_handle, stderr_handle):
            if stream:
                stream.close()

    report: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            reports = payload.get("reports", [])
            report = next((item for item in reports if item.get("kind") == "rag"), None)
        except Exception as exc:  # noqa: BLE001
            run_error = run_error or f"report_parse_error: {exc}"
    traces: list[dict[str, Any]] = []
    if trace_path.is_file():
        for raw in trace_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                try:
                    traces.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
    trace_ids = {str(item.get("trace_id")) for item in traces if item.get("trace_id")}
    released = not port_listening(PORT)
    stable = {
        "attempt_id": attempt_id,
        "selected_ids": CASE_IDS,
        "selected_ids_count": 5,
        "request_count": 5,
        "unique_case_ids": len({str(item.get("id")) for item in (report or {}).get("results", [])}),
        "retries": 0,
        "full36_run": False,
        "port_18080_released": released,
        "report_path": str(report_path),
        "trace_path": str(trace_path),
        "lineage_path": str(lineage_path),
        "runner_log": str(runner_log),
        "backend_stdout": str(backend_stdout),
        "backend_stderr": str(backend_stderr),
        "trace_lines": len(traces),
        "unique_trace_ids": len(trace_ids),
        "run_error": run_error,
        "evaluation_exit_code": eval_exit,
    }
    lineage = {
        **stable,
        "run_status": (report or {}).get("run_status", "failed_preflight" if run_error else "unavailable"),
        "completed_total": (report or {}).get("completed_total"),
        "planned_total": (report or {}).get("planned_total"),
        "passed": (report or {}).get("passed"),
        "error_count": ((report or {}).get("summary") or {}).get("error_count"),
        "collection": COLLECTION,
        "qdrant_url": QDRANT_URL,
        "testset": str(TESTSET_PATH),
        "environment": FIXED_ENV,
        "qdrant_preflight": qdrant_info,
        "backend_readyz": ready,
        "trace_stage_contract": STAGES,
    }
    lineage_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_human_report(human_path, report, traces, stable, cases)
    log(runner_log, f"stable completion marker: selected_ids=5, request_count=5, unique_case_ids={stable['unique_case_ids']}, retries=0, full36_run=false, port_18080_released={str(released).lower()}")
    return 0 if report is not None and not run_error and eval_exit in (0, 1) else 2


if __name__ == "__main__":
    raise SystemExit(main())
