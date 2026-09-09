"""Offline tests for the targeted-then-full Qdrant regression wrapper."""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_qdrant_regression.py"
SPEC = importlib.util.spec_from_file_location("run_qdrant_regression_unit", SCRIPT)
assert SPEC and SPEC.loader
regression = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regression
SPEC.loader.exec_module(regression)


def _case_ids() -> list[str]:
    return [f"case-{index:02d}" for index in range(1, 37)]


def _write_baseline(tmp_path: Path, *, failed: int = 17, label: str | None = None) -> Path:
    ids = _case_ids()
    report = {
        "kind": "rag",
        "label": label or f"qdrant-full-185-{regression.full36.DEFAULT_COLLECTION}",
        "testset": "testsets/rag_real_quality_v2.yaml",
        "base_url": "http://127.0.0.1:18080",
        "selected_case_ids": None,
        "total": 36,
        "planned_total": 36,
        "completed_total": 36,
        "run_status": "completed",
        "passed": 36 - failed,
        "summary": {"error_count": 0},
        "results": [{"id": case_id, "passed": index >= failed} for index, case_id in enumerate(ids)],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"reports": [report]}), encoding="utf-8")
    return path


def _write_testset(tmp_path: Path) -> Path:
    path = tmp_path / "testsets" / "rag_real_quality_v2.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name: rag_real_quality_v2\ntests: []\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, baseline: Path, *, output: Path, **kwargs: Any) -> Any:
    return regression.RegressionConfig(
        baseline_report=baseline,
        output_dir=output,
        project_root=tmp_path,
        backend_dir=tmp_path,
        testset_path=_write_testset(tmp_path),
        **kwargs,
    )


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("backend started\n")
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


def _patch_infrastructure(monkeypatch: pytest.MonkeyPatch) -> _FakeProcess:
    process = _FakeProcess()
    monkeypatch.setattr(
        regression.full36,
        "validate_preflight",
        lambda *args, **kwargs: SimpleNamespace(
            run_id="run-formal-1",
            source_manifest_sha256="a" * 64,
            embedding_model="bge-m3",
        ),
    )
    monkeypatch.setattr(regression.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(regression.full36, "wait_for_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regression.full36,
        "get_git_metadata",
        lambda _root: regression.full36.GitMetadata("head", True),
    )
    return process


def _evaluation_report(config: Any, output_path: Path, case_ids: list[str], *, passed: bool) -> dict[str, Any]:
    is_full = not case_ids
    ids = _case_ids() if is_full else case_ids
    items = [{"id": case_id, "passed": passed} for case_id in ids]
    return {
        "reports": [{
            "kind": "rag",
            "label": f"qdrant-full-185-{config.collection}",
            "testset": "testsets/rag_real_quality_v2.yaml",
            "base_url": f"http://127.0.0.1:{config.port}",
            "selected_case_ids": None if is_full else ids,
            "total": len(ids),
            "planned_total": len(ids),
            "completed_total": len(ids),
            "run_status": "completed",
            "passed": sum(item["passed"] for item in items),
            "summary": {"error_count": 0},
            "results": items,
        }]
    }


def _patch_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    config: Any,
    *,
    targeted_passed: bool,
    full_passed: bool = True,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        selected = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--case-id"]
        report = _evaluation_report(
            config,
            output_path,
            selected,
            passed=targeted_passed if selected else full_passed,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(regression.subprocess, "run", fake_run)
    return calls


def test_dry_run_prints_failed_ids_without_starting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_baseline(tmp_path)
    config = _config(tmp_path, baseline, output=tmp_path / "run", dry_run=True)
    monkeypatch.setattr(
        regression.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry-run must not start a process"),
    )

    result = regression.run_regression(config)

    output = capsys.readouterr().out
    assert result.exit_code == 0
    assert result.run_status == "dry_run"
    assert len(result.targeted_case_ids) == 17
    assert all(case_id in output for case_id in result.targeted_case_ids)
    assert not (tmp_path / "run").exists()


def test_failed_targeted_gate_never_runs_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _write_baseline(tmp_path)
    config = _config(tmp_path, baseline, output=tmp_path / "run")
    process = _patch_infrastructure(monkeypatch)
    calls = _patch_evaluator(monkeypatch, config, targeted_passed=False)

    result = regression.run_regression(config)

    assert result.exit_code == regression.EXIT_GATE_FAILED
    assert result.run_status == "targeted_gate_failed"
    assert len(calls) == 1
    assert "--case-id" in calls[0]
    assert not (config.output_dir / regression.FULL_FILENAME).exists()
    summary = json.loads((config.output_dir / regression.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["full"]["run_count"] == 0
    assert process.terminated


def test_all_targeted_pass_runs_full_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _write_baseline(tmp_path)
    config = _config(tmp_path, baseline, output=tmp_path / "run")
    process = _patch_infrastructure(monkeypatch)
    calls = _patch_evaluator(monkeypatch, config, targeted_passed=True)

    result = regression.run_regression(config)

    assert result.exit_code == 0
    assert result.run_status == "completed"
    assert len(calls) == 2
    assert "--case-id" in calls[0]
    assert "--case-id" not in calls[1]
    assert process.terminated
    summary = json.loads((config.output_dir / regression.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["targeted"]["run_count"] == 1
    assert summary["full"]["run_count"] == 1


def test_targeted_gate_rejects_a_different_case_set_even_when_all_pass() -> None:
    expected = {"case-01", "case-02"}
    report = {
        "run_status": "completed",
        "summary": {"error_count": 0},
        "total": 2,
        "passed": 2,
        "results": [
            {"id": "case-01", "passed": True},
            {"id": "case-other", "passed": True},
        ],
    }

    allowed, reason = regression.targeted_gate(report, expected)

    assert allowed is False
    assert reason == "targeted case set mismatch"


def test_resume_reuses_completed_identity_matched_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _write_baseline(tmp_path)
    output = tmp_path / "run"
    first = _config(tmp_path, baseline, output=output)
    _patch_infrastructure(monkeypatch)
    calls = _patch_evaluator(monkeypatch, first, targeted_passed=True)
    assert regression.run_regression(first).exit_code == 0
    assert len(calls) == 2

    resumed = _config(tmp_path, baseline, output=output, resume=True)
    monkeypatch.setattr(
        regression.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("resume must not start backend"),
    )
    monkeypatch.setattr(
        regression.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("resume must not rerun evaluator"),
    )

    result = regression.run_regression(resumed)

    assert result.exit_code == 0
    summary = json.loads((output / regression.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["targeted"]["reused"] is True
    assert summary["full"]["reused"] is True
    assert summary["targeted"]["run_count"] == 1
    assert summary["full"]["run_count"] == 1


def test_ready_failure_still_cleans_child_in_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _write_baseline(tmp_path)
    config = _config(tmp_path, baseline, output=tmp_path / "run")
    process = _patch_infrastructure(monkeypatch)
    monkeypatch.setattr(
        regression.full36,
        "wait_for_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(regression.full36.RunnerError("not ready")),
    )
    monkeypatch.setattr(
        regression.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("evaluator must not run after readiness failure"),
    )

    result = regression.run_regression(config)

    assert result.exit_code == regression.EXIT_RUNNER_ERROR
    assert result.run_status == "runner_error"
    assert process.terminated
    assert "not ready" in result.error


def test_resume_rejects_mismatched_lineage_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _write_baseline(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / regression.LINEAGE_FILENAME).write_text(
        json.dumps({"identity": {"collection": "wrong"}, "stages": {}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, baseline, output=output, resume=True)
    monkeypatch.setattr(
        regression.full36,
        "validate_preflight",
        lambda *args, **kwargs: SimpleNamespace(
            run_id="run-formal-1", source_manifest_sha256="a" * 64, embedding_model="bge-m3"
        ),
    )

    with pytest.raises(regression.RegressionError, match="identity mismatch"):
        regression.run_regression(config)


def test_lineage_redacts_user_id_and_environment_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _write_baseline(tmp_path)
    config = _config(
        tmp_path,
        baseline,
        output=tmp_path / "run",
        user_id="user-secret-token",
    )
    _patch_infrastructure(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "api-secret-value")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        selected = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--case-id"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_evaluation_report(config, output_path, selected, passed=True)),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="api_key=api-secret-value token=user-secret-token",
            stderr="",
        )

    monkeypatch.setattr(regression.subprocess, "run", fake_run)
    regression.run_regression(config)

    lineage_text = (config.output_dir / regression.LINEAGE_FILENAME).read_text(encoding="utf-8")
    assert "api-secret-value" not in lineage_text
    assert "user-secret-token" not in lineage_text
    assert "[redacted]" in lineage_text


def test_baseline_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, label="chroma-baseline")

    with pytest.raises(regression.RegressionError, match="identity mismatch"):
        regression.load_baseline_report(baseline)
