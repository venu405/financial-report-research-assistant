"""Offline unit tests for the isolated Qdrant 36-case evaluation launcher."""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_qdrant_full36.py"
SPEC = importlib.util.spec_from_file_location("run_qdrant_full36_unit", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _write_run_fixture(root: Path) -> dict[str, Any]:
    run_dir = root / runner.DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True)
    backend_dir = root / "code" / "backend"
    backend_dir.mkdir(parents=True)
    testset = root / runner.DEFAULT_TESTSET
    testset.parent.mkdir(parents=True, exist_ok=True)
    testset.write_text("tests:\n  - id: one\n", encoding="utf-8")

    run_id = "run-test-123"
    last_verification = {
        "status": "verified",
        "collection": runner.DEFAULT_COLLECTION,
        "vector_size": runner.DEFAULT_VECTOR_SIZE,
    }
    checkpoint = {
        "status": "completed",
        "completed_files": 185,
        "failed_files": 0,
        "remaining_files": 0,
        "collection_name": runner.DEFAULT_COLLECTION,
        "run_id": run_id,
        "last_verification": last_verification,
        "t1_after_verification": {
            "collection": runner.T1_COLLECTION,
            "points": 100,
            "vector_size": runner.DEFAULT_VECTOR_SIZE,
        },
    }
    manifest = {
        "status": "completed",
        "collection_name": runner.DEFAULT_COLLECTION,
        "collection_base": runner.DEFAULT_COLLECTION,
        "embedding_dimension": runner.DEFAULT_VECTOR_SIZE,
        "embedding_model": "bge-m3",
        "qdrant_url": runner.DEFAULT_QDRANT_URL,
        "source_manifest_sha256": "a" * 64,
        "run_id": run_id,
        "config": {
            "collection": runner.DEFAULT_COLLECTION,
            "embedding_dimension": runner.DEFAULT_VECTOR_SIZE,
            "embedding_model": "bge-m3",
            "qdrant_url": runner.DEFAULT_QDRANT_URL,
        },
    }
    verify_report = {
        "status": "verified",
        "collection": runner.DEFAULT_COLLECTION,
        "vector_size": runner.DEFAULT_VECTOR_SIZE,
        "run_id": run_id,
    }
    (run_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "verify-report.json").write_text(
        json.dumps(verify_report), encoding="utf-8"
    )
    return {
        "run_dir": run_dir,
        "backend_dir": backend_dir,
        "testset": testset,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "verify_report": verify_report,
    }


def _config(root: Path, fixture: dict[str, Any], output: Path, **kwargs: Any) -> Any:
    return runner.RunConfig(
        project_root=root,
        backend_dir=fixture["backend_dir"],
        checkpoint_path=fixture["run_dir"] / "checkpoint.json",
        verify_report_path=fixture["run_dir"] / "verify-report.json",
        manifest_path=fixture["run_dir"] / "run-manifest.json",
        testset_path=fixture["testset"],
        output_path=output,
        lineage_path=runner.lineage_path_for(output),
        **kwargs,
    )


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("uvicorn started\n")
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


def _patch_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse({"status": "ok"}),
    )
    monkeypatch.setattr(runner, "get_git_metadata", lambda _root: runner.GitMetadata("head", True))


def _fake_popen_factory(process: _FakeProcess, captured: dict[str, Any]):
    def _fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return process

    return _fake_popen


def test_environment_variables_are_qdrant_only_and_llm_settings_are_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_run_fixture(tmp_path)
    output = tmp_path / "reports" / "result.json"
    config = _config(tmp_path, fixture, output, user_id="eval-user", port=18081)
    process = _FakeProcess()
    captured: dict[str, Any] = {}
    monkeypatch.setenv("LLM_API_KEY", "do-not-print-this")
    monkeypatch.setenv("KB_OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(process, captured))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="evaluation ok", stderr=""
        ),
    )
    _patch_ready(monkeypatch)

    result = runner.run(config)

    environment = captured["env"]
    assert environment["KB_VECTOR_BACKEND"] == "qdrant"
    assert environment["KB_QDRANT_URL"] == runner.DEFAULT_QDRANT_URL
    assert environment["KB_QDRANT_COLLECTION"] == runner.DEFAULT_COLLECTION
    assert environment["KB_QDRANT_VECTOR_SIZE"] == "1024"
    assert environment["KB_QDRANT_CREATE_IF_MISSING"] == "false"
    assert environment["LLM_API_KEY"] == "do-not-print-this"
    assert str(fixture["backend_dir"] / "src") in environment["PYTHONPATH"]
    assert result.evaluation_exit_code == 0
    assert process.terminated and not process.killed


def test_preflight_rejects_non_completed_checkpoint(tmp_path: Path) -> None:
    fixture = _write_run_fixture(tmp_path)
    checkpoint_path = fixture["run_dir"] / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["failed_files"] = 1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(runner.PreflightError, match="failed_files mismatch"):
        runner.validate_preflight(tmp_path)


def test_default_output_is_unique_and_sidecar_collision_is_considered(tmp_path: Path) -> None:
    stamp = datetime(2026, 8, 31, tzinfo=timezone.utc)
    first = runner.default_output_path(tmp_path, now=stamp)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("existing", encoding="utf-8")
    second = runner.default_output_path(tmp_path, now=stamp)
    assert second != first
    assert second.name.endswith("-1.json")

    runner.lineage_path_for(second).write_text("existing sidecar", encoding="utf-8")
    third = runner.default_output_path(tmp_path, now=stamp)
    assert third != second
    assert third.name.endswith("-2.json")


def test_lineage_redacts_api_keys_and_user_tokens(tmp_path: Path) -> None:
    fixture = _write_run_fixture(tmp_path)
    output = tmp_path / "report.json"
    config = _config(tmp_path, fixture, output, user_id="user-secret-token")
    preflight = runner.validate_preflight(tmp_path)
    lineage = runner.build_lineage(
        config=config,
        preflight=preflight,
        testset_sha256="b" * 64,
        git=runner.GitMetadata("head", True),
        evaluation_command=["python", "--user-id", "user-secret-token"],
        started_at="start",
        ended_at="end",
        evaluation_started_at="eval-start",
        evaluation_exit_code=1,
        runner_status="evaluation_completed_with_failed_cases",
        runner_error="Authorization: Bearer user-secret-token",
        evaluation_stdout="api_key=super-secret-key",
        evaluation_stderr="token=super-secret-token",
        backend_stdout="",
        backend_stderr="",
    )
    encoded = json.dumps(lineage, ensure_ascii=False)
    assert "super-secret-key" not in encoded
    assert "user-secret-token" not in encoded
    assert "[redacted]" in encoded
    assert lineage["evaluation_exit_code"] == 1


def test_backend_is_cleaned_in_finally_when_ready_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_run_fixture(tmp_path)
    output = tmp_path / "report.json"
    config = _config(tmp_path, fixture, output, port=18082, startup_timeout=0.1)
    process = _FakeProcess()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(process, captured))
    monkeypatch.setattr(
        runner,
        "wait_for_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runner.RunnerError("ready timeout")),
    )
    monkeypatch.setattr(runner, "get_git_metadata", lambda _root: runner.GitMetadata("head", True))

    result = runner.run(config)

    assert result.runner_status == "runner_error"
    assert result.evaluation_exit_code is None
    assert process.terminated
    assert not process.killed
    assert config.lineage_path.is_file()
    lineage = json.loads(config.lineage_path.read_text(encoding="utf-8"))
    assert lineage["evaluation_exit_code"] is None


def test_evaluation_exit_one_keeps_report_and_records_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_run_fixture(tmp_path)
    output = tmp_path / "reports" / "qdrant-result.json"
    config = _config(tmp_path, fixture, output, port=18083)
    process = _FakeProcess()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(process, captured))

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"reports": []}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command, returncode=1, stdout="one case failed", stderr="evaluation warning"
        )

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    _patch_ready(monkeypatch)

    result = runner.run(config)

    assert result.runner_status == "evaluation_completed_with_failed_cases"
    assert result.runner_error is None
    assert result.evaluation_exit_code == 1
    assert output.is_file()
    lineage = json.loads(config.lineage_path.read_text(encoding="utf-8"))
    assert lineage["evaluation_exit_code"] == 1
    assert lineage["runner_status"] == "evaluation_completed_with_failed_cases"
    assert "one case failed" in lineage["evaluation_stdout_summary"]
    assert process.terminated and not process.killed


# ---------------------------------------------------------------------------
# 定向6/6 与全量36题不一致的回归防护（2026-09-03）
#
# 根因：build_backend_environment() 原先只强制 Qdrant 选择器，检索/路由/重排
# 配置全靠父进程继承。KB_STRUCTURED_FIN_ROUTE 在 retriever.py 中默认关闭，
# 父进程没导出时全量评测就在"财务结构化路由关闭"下运行 —— 文档级召回正常
# （recall@1=1.0）但正确块进不了 top-K（page_recall=0.0），数值核验无证据
# 因而安全拒答。定向脚本显式设了该变量，于是同样的题定向过、全量挂。
#
# 下面这些断言保证正式检索配置**始终**随全量链路一起下发，防止再次漏传。
# ---------------------------------------------------------------------------


# 显式列出必须下发的正式检索配置。这里**故意硬编码**而不遍历
# runner.FIXED_RETRIEVAL_ENVIRONMENT：若遍历常量本身，一旦有人把常量清空，
# 循环体不执行，测试就会"空转通过"，反而保护不住这条防线。
EXPECTED_RETRIEVAL_ENVIRONMENT: dict[str, str] = {
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


def test_backend_environment_imposes_fixed_retrieval_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """全量链路必须下发正式检索配置，不能只靠父进程继承。"""
    for key in EXPECTED_RETRIEVAL_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)

    fixture = _write_run_fixture(tmp_path)
    environment = runner.build_backend_environment(fixture["backend_dir"])

    for key, value in EXPECTED_RETRIEVAL_ENVIRONMENT.items():
        assert environment.get(key) == value, f"{key} 未随全量链路下发"


def test_backend_environment_enables_structured_fin_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """财务结构化路由必须显式开启；retriever.py 中它默认是关闭的。"""
    monkeypatch.delenv("KB_STRUCTURED_FIN_ROUTE", raising=False)

    fixture = _write_run_fixture(tmp_path)
    environment = runner.build_backend_environment(fixture["backend_dir"])

    # 这是定向6/6 与全量拒答的分水岭，单独断言一次
    assert environment.get("KB_STRUCTURED_FIN_ROUTE") == "1"


def test_backend_environment_overrides_stale_parent_retrieval_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """父进程若带着过期/错误的检索配置，必须被正式配置覆盖。"""
    monkeypatch.setenv("KB_STRUCTURED_FIN_ROUTE", "0")
    monkeypatch.setenv("KB_MAX_HITS_PER_DOC", "1")

    fixture = _write_run_fixture(tmp_path)
    environment = runner.build_backend_environment(fixture["backend_dir"])

    assert environment["KB_STRUCTURED_FIN_ROUTE"] == "1"
    assert environment["KB_MAX_HITS_PER_DOC"] == "12"


def test_backend_environment_still_inherits_llm_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """补检索配置不能顺带吞掉继承来的 LLM 凭据。"""
    monkeypatch.setenv("LLM_API_KEY", "do-not-print-this")

    fixture = _write_run_fixture(tmp_path)
    environment = runner.build_backend_environment(fixture["backend_dir"])

    assert environment["LLM_API_KEY"] == "do-not-print-this"


def test_fixed_retrieval_environment_matches_targeted_script(tmp_path: Path) -> None:
    """全量配置必须与定向验证脚本的 FixedConfig 逐项一致，否则又会出现两套口径。"""
    targeted = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_qdrant_targeted8_accuracy_20260902.ps1"
    )
    assert targeted.is_file(), "定向脚本缺失，无法校验配置一致性"
    source = targeted.read_text(encoding="utf-8-sig")

    import re

    declared = dict(
        re.findall(r'KB_([A-Z_]+)\s*=\s*"([^"]*)"', source)
    )
    expected = {
        key.removeprefix("KB_"): value
        for key, value in EXPECTED_RETRIEVAL_ENVIRONMENT.items()
    }
    for key, value in expected.items():
        assert key in declared, f"定向脚本缺少 {key}"
        assert declared[key] == value, (
            f"{key} 口径不一致：全量={value} 定向={declared[key]}"
        )


def test_lineage_records_retrieval_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lineage 必须记录实际生效的检索配置，便于复现与归因。"""
    fixture = _write_run_fixture(tmp_path)
    output = tmp_path / "reports" / "result.json"
    config = _config(tmp_path, fixture, output, port=18084)
    process = _FakeProcess()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(process, captured))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    _patch_ready(monkeypatch)

    runner.run(config)

    lineage = json.loads(config.lineage_path.read_text(encoding="utf-8"))
    recorded = lineage["backend"]["environment"]
    for key, value in EXPECTED_RETRIEVAL_ENVIRONMENT.items():
        assert recorded.get(key) == value, f"lineage 未记录 {key}"
    # 敏感凭据不得进入 lineage
    assert "LLM_API_KEY" not in recorded
