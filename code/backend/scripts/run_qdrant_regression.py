#!/usr/bin/env python3
"""Run a safe targeted-then-full Qdrant RAG regression.

The existing ``run_qdrant_full36.py`` remains the source of truth for the
formal preflight, isolated backend environment, readiness polling and child
process cleanup.  This wrapper only changes the evaluator case selection and
the gate between the two phases; it never writes to Qdrant.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]


def _load_full36_module() -> Any:
    try:
        import run_qdrant_full36 as module

        return module
    except ModuleNotFoundError:
        support_path = SCRIPT_DIR / "run_qdrant_full36.py"
        spec = importlib.util.spec_from_file_location(
            "run_qdrant_full36_regression_support", support_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load runner support: {support_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


full36 = _load_full36_module()

REGRESSION_SCHEMA_VERSION = 1
FULL_CASE_COUNT = 36
LINEAGE_FILENAME = "lineage.json"
TARGETED_FILENAME = "targeted.json"
FULL_FILENAME = "full.json"
SUMMARY_FILENAME = "summary.json"
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_RUNNER_ERROR = 2


class RegressionError(RuntimeError):
    """Input, identity, preflight or process orchestration failure."""


@dataclass(frozen=True)
class RegressionConfig:
    baseline_report: Path
    output_dir: Path
    project_root: Path = PROJECT_ROOT
    backend_dir: Path = BACKEND_DIR
    testset_path: Path = PROJECT_ROOT / full36.DEFAULT_TESTSET
    collection: str = full36.DEFAULT_COLLECTION
    qdrant_url: str = full36.DEFAULT_QDRANT_URL
    vector_size: int = full36.DEFAULT_VECTOR_SIZE
    port: int = full36.DEFAULT_PORT
    user_id: str | None = None
    startup_timeout: float = full36.DEFAULT_STARTUP_TIMEOUT
    resume: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class BaselineInfo:
    path: Path
    report: dict[str, Any]
    case_ids: frozenset[str]
    failed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegressionResult:
    exit_code: int
    run_status: str
    summary_path: Path | None
    lineage_path: Path | None
    targeted_case_ids: tuple[str, ...]
    error: str | None = None


def _safe_error(exc: BaseException) -> str:
    return full36._safe_error(exc)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RegressionError(f"{description} cannot be read: {_safe_error(exc)}") from exc
    if not isinstance(value, dict):
        raise RegressionError(f"{description} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Use the proven atomic UTF-8 writer from the full36 launcher."""
    full36._atomic_write_json(path, payload)


def _normal_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/").casefold()


def _is_expected_testset(value: Any) -> bool:
    normalized = _normal_path(value)
    return normalized == "testsets/rag_real_quality_v2.yaml" or normalized.endswith(
        "/testsets/rag_real_quality_v2.yaml"
    )


def _expected_label(collection: str) -> str:
    return f"qdrant-full-185-{collection}"


def load_baseline_report(
    path: Path,
    *,
    collection: str = full36.DEFAULT_COLLECTION,
) -> BaselineInfo:
    """Validate a completed 36-case report and extract only failed IDs."""
    path = path.resolve()
    if not path.is_file():
        raise RegressionError(f"baseline report does not exist: {path}")
    payload = _read_json(path, "baseline report")
    reports = payload.get("reports")
    if not isinstance(reports, list):
        raise RegressionError("baseline report.reports must be a list")
    rag_reports = [item for item in reports if isinstance(item, dict) and item.get("kind") == "rag"]
    if len(rag_reports) != 1:
        raise RegressionError("baseline report must contain exactly one rag report")
    report = rag_reports[0]
    if report.get("run_status") != "completed":
        raise RegressionError("baseline RAG report is not completed")
    if report.get("label") != _expected_label(collection):
        raise RegressionError("baseline report identity mismatch: label/collection")
    if not _is_expected_testset(report.get("testset")):
        raise RegressionError("baseline report identity mismatch: testset")
    if any(report.get(key) != FULL_CASE_COUNT for key in ("total", "planned_total", "completed_total")):
        raise RegressionError("baseline report must contain all 36 completed cases")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("error_count") != 0:
        raise RegressionError("baseline report must have error_count=0")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != FULL_CASE_COUNT:
        raise RegressionError("baseline report results must contain 36 cases")
    case_ids: list[str] = []
    failed: list[str] = []
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise RegressionError("baseline report contains an invalid case id")
        case_id = item["id"]
        case_ids.append(case_id)
        if item.get("passed") is False:
            failed.append(case_id)
    if len(set(case_ids)) != FULL_CASE_COUNT:
        raise RegressionError("baseline report contains duplicate case IDs")
    selected = report.get("selected_case_ids")
    if selected not in (None, []):
        raise RegressionError("baseline report identity mismatch: it is not a full run")
    return BaselineInfo(
        path=path,
        report=report,
        case_ids=frozenset(case_ids),
        failed_case_ids=tuple(sorted(failed)),
    )


def _sha256(path: Path) -> str:
    return full36.sha256_file(path)


def _user_id_digest(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def build_identity(config: RegressionConfig, baseline: BaselineInfo) -> dict[str, Any]:
    """Build the resume identity without writing a user ID or API key."""
    return {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "baseline_report_sha256": _sha256(baseline.path),
        "baseline_label": baseline.report.get("label"),
        "testset": _relative(config.testset_path, config.project_root),
        "testset_sha256": _sha256(config.testset_path),
        "collection": config.collection,
        "qdrant_url": full36.sanitize_url(config.qdrant_url),
        "vector_size": config.vector_size,
        "targeted_case_ids": list(baseline.failed_case_ids),
        "port": config.port,
        "user_id_sha256": _user_id_digest(config.user_id),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _phase_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "targeted": output_dir / TARGETED_FILENAME,
        "full": output_dir / FULL_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
        "lineage": output_dir / LINEAGE_FILENAME,
    }


def _resolve_output_dir(raw: Path | None) -> Path:
    if raw is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return (BACKEND_DIR / "reports" / f"qdrant-regression-{stamp}-{uuid.uuid4().hex[:10]}").resolve()
    return (raw if raw.is_absolute() else Path.cwd() / raw).resolve()


def prepare_output_dir(path: Path, *, resume: bool) -> None:
    if path.exists() and not path.is_dir():
        raise RegressionError(f"output-dir is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    entries = list(path.iterdir())
    if entries:
        raise RegressionError(f"output-dir is not empty; use --resume for this run: {path}")


def _load_resume_lineage(path: Path, identity: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    if not path.exists():
        if resume and any((path.parent / name).exists() for name in (TARGETED_FILENAME, FULL_FILENAME, SUMMARY_FILENAME)):
            raise RegressionError("resume identity unavailable: lineage.json is missing")
        return {"stages": {}}
    if not resume:
        raise RegressionError(f"lineage already exists; use --resume: {path.parent}")
    lineage = _read_json(path, "lineage")
    if lineage.get("identity") != dict(identity):
        raise RegressionError("resume identity mismatch")
    stages = lineage.get("stages")
    if not isinstance(stages, dict):
        raise RegressionError("lineage stages must be an object")
    return lineage


def _report_from_artifact(path: Path, description: str) -> dict[str, Any]:
    payload = _read_json(path, description)
    reports = payload.get("reports")
    if not isinstance(reports, list):
        raise RegressionError(f"{description}.reports must be a list")
    rag_reports = [item for item in reports if isinstance(item, dict) and item.get("kind") == "rag"]
    if len(rag_reports) != 1:
        raise RegressionError(f"{description} must contain exactly one rag report")
    return rag_reports[0]


def _validate_stage_identity(
    report: Mapping[str, Any],
    *,
    expected_case_ids: set[str],
    full: bool,
    config: RegressionConfig,
    description: str,
) -> None:
    if report.get("label") != _expected_label(config.collection):
        raise RegressionError(f"{description} identity mismatch: label/collection")
    if not _is_expected_testset(report.get("testset")):
        raise RegressionError(f"{description} identity mismatch: testset")
    expected_base_url = f"http://127.0.0.1:{config.port}"
    if full36.sanitize_url(str(report.get("base_url") or "")) != expected_base_url:
        raise RegressionError(f"{description} identity mismatch: base_url")
    selected = report.get("selected_case_ids")
    if full:
        if selected not in (None, []):
            raise RegressionError(f"{description} identity mismatch: selected cases")
    elif set(selected or []) != expected_case_ids:
        raise RegressionError(f"{description} identity mismatch: selected cases")


def load_stage_if_completed(
    path: Path,
    *,
    expected_case_ids: set[str],
    full: bool,
    config: RegressionConfig,
    description: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = _report_from_artifact(path, description)
    _validate_stage_identity(
        report,
        expected_case_ids=expected_case_ids,
        full=full,
        config=config,
        description=description,
    )
    if report.get("run_status") == "completed":
        return report
    return None


def targeted_gate(report: Mapping[str, Any], expected_case_ids: set[str]) -> tuple[bool, str]:
    """The only condition that permits the full 36-case phase."""
    results = report.get("results")
    summary = report.get("summary")
    actual_ids = {
        item.get("id")
        for item in results
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(results, list) else set()
    total = report.get("total")
    passed = report.get("passed")
    error_count = summary.get("error_count") if isinstance(summary, dict) else None
    if report.get("run_status") != "completed":
        return False, "targeted report is not completed"
    if error_count != 0:
        return False, f"targeted error_count={error_count!r}"
    if not isinstance(results, list) or len(results) != len(expected_case_ids) or actual_ids != expected_case_ids:
        return False, "targeted case set mismatch"
    if total != len(expected_case_ids) or passed != total:
        return False, f"targeted passed={passed!r}/total={total!r}"
    return True, "targeted gate passed"


def _build_phase_config(config: RegressionConfig, output_path: Path) -> Any:
    return full36.RunConfig(
        project_root=config.project_root,
        backend_dir=config.backend_dir,
        checkpoint_path=config.project_root / full36.DEFAULT_RUN_DIR / "checkpoint.json",
        verify_report_path=config.project_root / full36.DEFAULT_RUN_DIR / "verify-report.json",
        manifest_path=config.project_root / full36.DEFAULT_RUN_DIR / "run-manifest.json",
        testset_path=config.testset_path,
        output_path=output_path,
        lineage_path=output_path.with_suffix(".lineage.json"),
        port=config.port,
        user_id=config.user_id,
        startup_timeout=config.startup_timeout,
        collection=config.collection,
        qdrant_url=config.qdrant_url,
        vector_size=config.vector_size,
    )


def build_phase_evaluation_command(
    config: RegressionConfig,
    output_path: Path,
    *,
    case_ids: Sequence[str] | None = None,
) -> list[str]:
    """Reuse full36's fixed evaluator command and add repeatable --case-id."""
    command = full36.build_evaluation_command(_build_phase_config(config, output_path))
    if not case_ids:
        return command
    output_index = command.index("--output")
    selected: list[str] = []
    for case_id in sorted(case_ids):
        selected.extend(["--case-id", case_id])
    return [*command[:output_index], *selected, *command[output_index:]]


def _build_backend_popen_kwargs(config: RegressionConfig, environment: Mapping[str, str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": config.backend_dir,
        "env": dict(environment),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return kwargs


def _run_evaluator(command: Sequence[str], config: RegressionConfig, environment: Mapping[str, str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(command),
        cwd=config.backend_dir,
        env=dict(environment),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return int(completed.returncode), str(getattr(completed, "stdout", "") or ""), str(getattr(completed, "stderr", "") or "")


def _lineage_payload(
    config: RegressionConfig,
    baseline: BaselineInfo,
    identity: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    status: str,
    exit_code: int | None,
    error: str | None,
    commands: Mapping[str, Sequence[str]],
    diagnostics: Mapping[str, str],
) -> dict[str, Any]:
    secrets = list(full36._secret_values())
    if config.user_id:
        secrets.append(config.user_id)
    return {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "kind": "qdrant_regression_lineage",
        "identity": dict(identity),
        "collection": config.collection,
        "qdrant_url": full36.sanitize_url(config.qdrant_url),
        "vector_size": config.vector_size,
        "baseline_report": _relative(baseline.path, config.project_root),
        "testset": _relative(config.testset_path, config.project_root),
        "run_id": lineage.get("run_id"),
        "formal_run_id": lineage.get("formal_run_id"),
        "source_manifest_sha256": lineage.get("source_manifest_sha256"),
        "embedding_model": lineage.get("embedding_model"),
        "stages": lineage.get("stages", {}),
        "runner_status": status,
        "exit_code": exit_code,
        "error": full36.redact_sensitive_text(error, secrets=secrets) if error else None,
        "commands": {
            name: full36.redact_command(command, user_id=config.user_id)
            for name, command in commands.items()
        },
        "diagnostics": {
            name: full36.summarize_text(value, secrets=secrets)
            for name, value in diagnostics.items()
        },
    }


def _initial_lineage(identity: Mapping[str, Any], baseline: BaselineInfo) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "run_id": f"regression-{uuid.uuid4().hex}",
        "stages": {},
        "baseline_failed_case_ids": list(baseline.failed_case_ids),
    }


def _stage_summary(report: Mapping[str, Any] | None, *, path: Path, reused: bool, runs: int) -> dict[str, Any]:
    if report is None:
        return {"path": path.name, "status": "not_run", "reused": reused, "run_count": runs}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "path": path.name,
        "status": report.get("run_status"),
        "reused": reused,
        "run_count": runs,
        "total": report.get("total"),
        "passed": report.get("passed"),
        "error_count": summary.get("error_count"),
        "case_ids": sorted(
            item.get("id") for item in report.get("results", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ),
    }


def _print_dry_run(baseline: BaselineInfo) -> None:
    print("dry-run: no backend, LLM, Qdrant, or evaluator request will start")
    print(f"targeted: {len(baseline.failed_case_ids)} failed case(s)")
    for case_id in baseline.failed_case_ids:
        print(case_id)
    print("full36: run only if targeted report is completed, error_count=0, all targeted cases pass, and case IDs match")


def run_regression(config: RegressionConfig) -> RegressionResult:
    baseline = load_baseline_report(config.baseline_report, collection=config.collection)
    targeted_ids = tuple(baseline.failed_case_ids)
    if config.dry_run:
        _print_dry_run(baseline)
        return RegressionResult(EXIT_OK, "dry_run", None, None, targeted_ids)

    if not config.testset_path.is_file():
        raise RegressionError(f"testset does not exist: {config.testset_path}")
    # Existing full36 preflight is deliberately read-only and verifies the
    # completed formal collection identity before any child is started.
    preflight = full36.validate_preflight(
        config.project_root,
        collection=config.collection,
        qdrant_url=config.qdrant_url,
        vector_size=config.vector_size,
        run_dir=config.project_root / full36.DEFAULT_RUN_DIR,
        testset=config.testset_path,
    )
    prepare_output_dir(config.output_dir, resume=config.resume)
    paths = _phase_paths(config.output_dir)
    identity = build_identity(config, baseline)
    lineage = _load_resume_lineage(paths["lineage"], identity, resume=config.resume)
    if not lineage.get("identity"):
        lineage["identity"] = dict(identity)
    lineage.setdefault("run_id", f"regression-{uuid.uuid4().hex}")
    lineage["formal_run_id"] = preflight.run_id
    lineage["source_manifest_sha256"] = preflight.source_manifest_sha256
    lineage["embedding_model"] = preflight.embedding_model

    targeted_report: dict[str, Any] | None = None
    full_report: dict[str, Any] | None = None
    targeted_reused = False
    full_reused = False
    targeted_runs = int((lineage.get("stages", {}).get("targeted") or {}).get("run_count", 0))
    full_runs = int((lineage.get("stages", {}).get("full") or {}).get("run_count", 0))
    commands: dict[str, Sequence[str]] = {}
    diagnostics: dict[str, str] = {}
    runner_status = "runner_error"
    exit_code = EXIT_RUNNER_ERROR
    runner_error: str | None = None
    backend_process: Any | None = None
    backend_stdout = full36._StreamCapture()
    backend_stderr = full36._StreamCapture()
    environment = full36.build_backend_environment(
        config.backend_dir,
        collection=config.collection,
        qdrant_url=config.qdrant_url,
        vector_size=config.vector_size,
    )
    # Persist identity before starting a child.  If the host process is
    # interrupted after an evaluator has produced a report, --resume can
    # still authenticate and reuse that completed stage.
    try:
        _atomic_write_json(
            paths["lineage"],
            _lineage_payload(
                config,
                baseline,
                identity,
                lineage,
                status="in_progress",
                exit_code=None,
                error=None,
                commands={},
                diagnostics={},
            ),
        )
    except Exception as exc:
        raise RegressionError(f"cannot initialize lineage: {_safe_error(exc)}") from exc

    def start_backend() -> None:
        nonlocal backend_process
        if backend_process is not None:
            return
        backend_process = subprocess.Popen(
            full36.build_backend_command(config.port),
            **_build_backend_popen_kwargs(config, environment),
        )
        backend_stdout.start(backend_process.stdout)
        backend_stderr.start(backend_process.stderr)
        full36.wait_for_ready(
            backend_process,
            f"http://127.0.0.1:{config.port}",
            config.startup_timeout,
        )

    def run_stage(name: str, output_path: Path, case_ids: Sequence[str] | None) -> tuple[dict[str, Any], int]:
        nonlocal targeted_runs, full_runs
        command = build_phase_evaluation_command(config, output_path, case_ids=case_ids)
        commands[name] = command
        return_code, stdout, stderr = _run_evaluator(command, config, environment)
        diagnostics[f"{name}_stdout"] = stdout
        diagnostics[f"{name}_stderr"] = stderr
        if name == "targeted":
            targeted_runs += 1
        else:
            full_runs += 1
        return _report_from_artifact(output_path, f"{name} report"), return_code

    try:
        targeted_report = load_stage_if_completed(
            paths["targeted"],
            expected_case_ids=set(targeted_ids),
            full=False,
            config=config,
            description="targeted report",
        )
        targeted_reused = targeted_report is not None
        if targeted_report is None and targeted_ids:
            start_backend()
            targeted_report, _targeted_exit = run_stage("targeted", paths["targeted"], targeted_ids)
            _validate_stage_identity(
                targeted_report,
                expected_case_ids=set(targeted_ids),
                full=False,
                config=config,
                description="targeted report",
            )
        elif targeted_report is None:
            # A zero-failure baseline has no targeted request to execute.
            targeted_report = {
                "kind": "rag",
                "run_status": "completed",
                "label": _expected_label(config.collection),
                "testset": "testsets/rag_real_quality_v2.yaml",
                "base_url": f"http://127.0.0.1:{config.port}",
                "selected_case_ids": [],
                "total": 0,
                "planned_total": 0,
                "completed_total": 0,
                "passed": 0,
                "summary": {"error_count": 0},
                "results": [],
            }
            _atomic_write_json(paths["targeted"], {"reports": [targeted_report]})

        gate_ok, gate_reason = targeted_gate(targeted_report, set(targeted_ids))
        lineage["stages"] = {
            "targeted": {
                "run_count": targeted_runs,
                "status": targeted_report.get("run_status"),
                "gate_passed": gate_ok,
                "gate_reason": gate_reason,
            }
        }
        if not gate_ok:
            runner_status = "targeted_gate_failed"
            exit_code = EXIT_GATE_FAILED
            runner_error = gate_reason
        else:
            full_report = load_stage_if_completed(
                paths["full"],
                expected_case_ids=set(baseline.case_ids),
                full=True,
                config=config,
                description="full report",
            )
            full_reused = full_report is not None
            if full_report is None:
                start_backend()
                full_report, _full_exit = run_stage("full", paths["full"], None)
                _validate_stage_identity(
                    full_report,
                    expected_case_ids=set(baseline.case_ids),
                    full=True,
                    config=config,
                    description="full report",
                )
            summary = full_report.get("summary") if isinstance(full_report.get("summary"), dict) else {}
            full_complete = (
                full_report.get("run_status") == "completed"
                and summary.get("error_count") == 0
                and full_report.get("total") == FULL_CASE_COUNT
                and full_report.get("passed") == FULL_CASE_COUNT
                and {
                    item.get("id") for item in full_report.get("results", [])
                    if isinstance(item, dict)
                } == set(baseline.case_ids)
            )
            runner_status = "completed" if full_complete else "full_completed_with_failed_cases"
            exit_code = EXIT_OK if full_complete else EXIT_GATE_FAILED
            if not full_complete:
                runner_error = "full report did not pass all 36 cases"
            lineage["stages"]["full"] = {
                "run_count": full_runs,
                "status": full_report.get("run_status"),
                "passed_all": full_complete,
            }
    except KeyboardInterrupt:
        runner_status = "interrupted"
        exit_code = 130
        runner_error = "KeyboardInterrupt"
    except Exception as exc:
        runner_status = "runner_error"
        exit_code = EXIT_RUNNER_ERROR
        runner_error = _safe_error(exc)
    finally:
        if backend_process is not None:
            cleanup_error = full36.terminate_process(backend_process)
            if cleanup_error and not runner_error:
                runner_error = f"backend cleanup error: {cleanup_error}"
                runner_status = "runner_error"
                exit_code = EXIT_RUNNER_ERROR
        backend_stdout.join()
        backend_stderr.join()

        lineage["stages"] = lineage.get("stages", {})
        lineage_payload = _lineage_payload(
            config,
            baseline,
            identity,
            lineage,
            status=runner_status,
            exit_code=exit_code,
            error=runner_error,
            commands=commands,
            diagnostics={
                **diagnostics,
                "backend_stdout": backend_stdout.text(),
                "backend_stderr": backend_stderr.text(),
            },
        )
        _atomic_write_json(paths["lineage"], lineage_payload)
        summary_payload = {
            "schema_version": REGRESSION_SCHEMA_VERSION,
            "kind": "qdrant_regression",
            "run_status": runner_status,
            "exit_code": exit_code,
            "baseline_report": _relative(baseline.path, config.project_root),
            "collection": config.collection,
            "qdrant_url": full36.sanitize_url(config.qdrant_url),
            "targeted_case_ids": list(targeted_ids),
            "targeted": _stage_summary(
                targeted_report,
                path=paths["targeted"],
                reused=targeted_reused,
                runs=targeted_runs,
            ),
            "full": _stage_summary(
                full_report,
                path=paths["full"],
                reused=full_reused,
                runs=full_runs,
            ),
            "error": runner_error,
        }
        _atomic_write_json(paths["summary"], summary_payload)

    return RegressionResult(
        exit_code=exit_code,
        run_status=runner_status,
        summary_path=paths["summary"],
        lineage_path=paths["lineage"],
        targeted_case_ids=targeted_ids,
        error=runner_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run failed Qdrant RAG cases first; run full36 only after the targeted gate passes."
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=full36.DEFAULT_PORT)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--startup-timeout", type=float, default=full36.DEFAULT_STARTUP_TIMEOUT)
    return parser


def _config_from_args(args: argparse.Namespace) -> RegressionConfig:
    if args.port < 1 or args.port > 65535 or args.port == 8000:
        raise RegressionError("port must be between 1 and 65535 and cannot be 8000")
    if args.startup_timeout <= 0:
        raise RegressionError("startup-timeout must be positive")
    baseline = args.baseline_report if args.baseline_report.is_absolute() else Path.cwd() / args.baseline_report
    testset_path = PROJECT_ROOT / full36.DEFAULT_TESTSET
    return RegressionConfig(
        baseline_report=baseline.resolve(),
        output_dir=_resolve_output_dir(args.output_dir),
        testset_path=testset_path,
        port=args.port,
        user_id=args.user_id,
        startup_timeout=args.startup_timeout,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_regression(_config_from_args(args))
    except (RegressionError, full36.RunnerError) as exc:
        print(f"REGRESSION_ERROR: {_safe_error(exc)}", file=sys.stderr)
        return EXIT_RUNNER_ERROR
    if result.summary_path:
        print(f"summary: {result.summary_path}")
    if result.lineage_path:
        print(f"lineage: {result.lineage_path}")
    if result.error:
        print(f"REGRESSION: {result.error}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
