"""Static contract tests for the Step 5 targeted-five launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_qdrant_step5_targeted5_20260904.py"
SPEC = importlib.util.spec_from_file_location("step5_targeted5_launcher", SCRIPT)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_launcher_has_exactly_the_fixed_five_cases_and_no_diagnostic_output():
    assert len(launcher.CASE_IDS) == 5
    assert len(set(launcher.CASE_IDS)) == 5
    assert all(case_id.startswith("real_") for case_id in launcher.CASE_IDS)
    assert "failed5-diagnostic" not in str(launcher.HUMAN_REPORT_PATH)
    assert "full36" not in launcher.CASE_IDS


def test_fixed_environment_matches_formal_qdrant_validation_settings():
    expected = {
        "KB_VECTOR_BACKEND": "qdrant",
        "KB_QDRANT_URL": "http://127.0.0.1:16333",
        "KB_QDRANT_COLLECTION": "kb_full_codex_20260830_24c7407e",
        "KB_QDRANT_VECTOR_SIZE": "1024",
        "KB_QDRANT_CREATE_IF_MISSING": "false",
        "KB_EMBEDDING_MODE": "bge_m3",
        "KB_EMBEDDING_MODEL": "bge-m3",
        "KB_TOP_K": "5",
        "KB_RECALL_K": "20",
        "KB_MAX_HITS_PER_DOC": "12",
        "KB_STRUCTURED_FIN_ROUTE": "1",
        "KB_FIN_ROUTE_TOP_N": "8",
        "KB_RERANK_MODE": "llm",
    }
    for key, value in expected.items():
        assert launcher.FIXED_ENVIRONMENT[key] == value


def test_evaluator_command_contains_only_one_invocation_and_five_case_flags(tmp_path):
    command = launcher.build_evaluator_command(
        "python",
        tmp_path / "result.json",
        "qdrant-step5-targeted5-20260904-test",
    )
    assert command[:4] == [
        "python",
        "scripts/evaluate_quality.py",
        "--rag",
        "testsets/rag_real_quality_v2.yaml",
    ]
    assert command.count("--case-id") == 5
    values = [command[index + 1] for index, value in enumerate(command) if value == "--case-id"]
    assert values == list(launcher.CASE_IDS)
    assert "--retry" not in command
    assert "--ocr" not in command


def test_report_metrics_and_stable_marker_preserve_semantic_failures(tmp_path):
    report = {
        "run_status": "completed",
        "completed_total": 5,
        "planned_total": 5,
        "passed": 3,
        "summary": {"error_count": 0},
        "results": [
            {"id": launcher.CASE_IDS[0], "passed": True},
            {"id": launcher.CASE_IDS[1], "passed": True},
            {"id": launcher.CASE_IDS[2], "passed": True},
            {"id": launcher.CASE_IDS[3], "passed": False},
            {"id": launcher.CASE_IDS[4], "passed": False},
        ],
    }
    metrics = launcher.report_metrics(report)
    assert metrics == {
        "passed": 3,
        "completed_total": 5,
        "planned_total": 5,
        "error_count": 0,
        "run_status": "completed",
        "unique_case_ids": 5,
    }
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "report",
            "lineage",
            "runner_log",
            "backend_stdout",
            "backend_stderr",
        )
    }
    stable = launcher.build_stable_metadata(
        "qdrant-step5-targeted5-20260904-test",
        report,
        paths,
        port_released=True,
        evaluation_exit_code=1,
        run_error=None,
    )
    assert stable["passed_display"] == "3/5"
    assert stable["error_count"] == 0
    assert stable["request_count"] == 5
    assert stable["retries"] == 0
    assert stable["full36_run"] is False
    assert stable["port_18080_released"] is True


def test_existing_human_report_is_rejected_without_overwrite(tmp_path):
    target = tmp_path / "Qdrant_Step5修复后五题验收_20260904.md"
    target.write_text("keep", encoding="utf-8")
    try:
        launcher.ensure_output_paths_absent([target])
    except launcher.Step5RunnerError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("existing human report must be rejected")
