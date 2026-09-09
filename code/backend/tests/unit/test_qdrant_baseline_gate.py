from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_qdrant_baseline import BaselineGateError, check_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE = PROJECT_ROOT / "code/backend/testsets/qdrant_rag_baseline_34of36.json"
CURRENT_REPORT = PROJECT_ROOT / (
    "code/backend/reports/"
    "qdrant-full36-after-raw-narrative-fix-venv-20260906.json"
)
CURRENT_LINEAGE = PROJECT_ROOT / (
    "code/backend/reports/"
    "qdrant-full36-after-raw-narrative-fix-venv-20260906.lineage.json"
)


def _load_report() -> dict:
    return json.loads(CURRENT_REPORT.read_text(encoding="utf-8"))


def _set_passed(report: dict, case_id: str, passed: bool) -> None:
    for item in report["reports"][0]["results"]:
        if item["id"] == case_id:
            item["passed"] = passed
            report["reports"][0]["passed"] = sum(
                candidate.get("passed") is True
                for candidate in report["reports"][0]["results"]
            )
            return
    raise AssertionError(f"missing test case in fixture: {case_id}")


def _candidate(tmp_path: Path, mutate) -> tuple[Path, Path]:
    payload = _load_report()
    mutate(payload)
    report_path = tmp_path / "candidate.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    lineage_path = tmp_path / "candidate.lineage.json"
    lineage_path.write_text(CURRENT_LINEAGE.read_text(encoding="utf-8"), encoding="utf-8")
    return report_path, lineage_path


def _run(tmp_path: Path, mutate) -> dict:
    report_path, lineage_path = _candidate(tmp_path, mutate)
    return check_candidate(BASELINE, report_path, lineage_path, project_root=PROJECT_ROOT)


def test_formal_34_source_and_current_candidate_pass() -> None:
    result = check_candidate(
        BASELINE,
        CURRENT_REPORT,
        CURRENT_LINEAGE,
        project_root=PROJECT_ROOT,
    )

    assert result == {
        "marker": "QDRANT_BASELINE_GATE: PASS",
        "baseline_passed": 34,
        "candidate_passed": 34,
        "candidate_total": 36,
        "newly_passed": [],
        "report": str(CURRENT_REPORT),
    }


def test_missing_any_baseline_id_fails_even_when_score_is_preserved(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        report = payload["reports"][0]
        _set_passed(payload, "real_dongshi_2024_revenue", False)
        _set_passed(payload, "real_zhongguohe_2024_total_revenue_synonym", True)
        assert report["passed"] == 34

    with pytest.raises(BaselineGateError, match="回退了基线通过ID"):
        _run(tmp_path, mutate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["testset"].__setitem__("sha256", "0" * 64),
            "题集SHA256",
        ),
        (
            lambda manifest: manifest["source_report"].__setitem__("sha256", "0" * 64),
            "源报告SHA256",
        ),
        (
            lambda manifest: manifest["retrieval_config"].__setitem__("KB_TOP_K", "6"),
            "基线检索配置指纹",
        ),
    ],
)
def test_testset_source_hash_or_config_mismatch_fails(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = json.loads(BASELINE.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path = tmp_path / "baseline.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BaselineGateError, match=message):
        check_candidate(
            manifest_path,
            CURRENT_REPORT,
            CURRENT_LINEAGE,
            project_root=PROJECT_ROOT,
        )


def test_candidate_lineage_config_mismatch_fails(tmp_path: Path) -> None:
    report_path, lineage_path = _candidate(tmp_path, lambda _: None)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["backend"]["environment"]["KB_TOP_K"] = "6"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    with pytest.raises(BaselineGateError, match="KB_TOP_K"):
        check_candidate(BASELINE, report_path, lineage_path, project_root=PROJECT_ROOT)


def test_duplicate_baseline_id_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads(BASELINE.read_text(encoding="utf-8"))
    manifest["baseline_passed_case_ids"].append(
        manifest["baseline_passed_case_ids"][0]
    )
    manifest_path = tmp_path / "baseline.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BaselineGateError, match="排序且无重复"):
        check_candidate(
            manifest_path,
            CURRENT_REPORT,
            CURRENT_LINEAGE,
            project_root=PROJECT_ROOT,
        )
