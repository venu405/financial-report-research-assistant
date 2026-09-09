from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_qdrant_31of36_baseline import BaselineGateError, check_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE = PROJECT_ROOT / "code/backend/testsets/qdrant_rag_baseline_31of36.json"
CURRENT_REPORT = PROJECT_ROOT / (
    "code/backend/reports/"
    "qdrant-full-185-kb_full_codex_20260830_24c7407e-20260903T111002919444Z.json"
)
CURRENT_LINEAGE = PROJECT_ROOT / (
    "code/backend/reports/"
    "qdrant-full-185-kb_full_codex_20260830_24c7407e-20260903T111002919444Z.lineage.json"
)


def _load_current_report() -> dict:
    return json.loads(CURRENT_REPORT.read_text(encoding="utf-8"))


def _candidate(tmp_path: Path, mutate) -> tuple[Path, Path]:
    payload = _load_current_report()
    report = payload["reports"][0]
    mutate(report)
    report_path = tmp_path / "candidate.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    lineage_path = tmp_path / "candidate.lineage.json"
    lineage_path.write_text(CURRENT_LINEAGE.read_text(encoding="utf-8"), encoding="utf-8")
    return report_path, lineage_path


def _set_passed(report: dict, case_id: str, passed: bool) -> None:
    for item in report["results"]:
        if item["id"] == case_id:
            item["passed"] = passed
            break
    else:
        raise AssertionError(f"missing test case in fixture: {case_id}")
    report["passed"] = sum(item.get("passed") is True for item in report["results"])


def _run(tmp_path: Path, mutate) -> dict:
    report_path, lineage_path = _candidate(tmp_path, mutate)
    return check_candidate(BASELINE, report_path, lineage_path, project_root=PROJECT_ROOT)


def test_current_formal_31of36_report_passes() -> None:
    result = check_candidate(
        BASELINE,
        CURRENT_REPORT,
        CURRENT_LINEAGE,
        project_root=PROJECT_ROOT,
    )

    assert result["marker"] == "QDRANT_31OF36_BASELINE_GATE: PASS"
    assert result["baseline_passed"] == 31
    assert result["candidate_passed"] == 31
    assert result["candidate_total"] == 36


def test_candidate_with_only_30_passed_fails(tmp_path: Path) -> None:
    with pytest.raises(BaselineGateError, match="低于最低门槛"):
        _run(tmp_path, lambda report: _set_passed(report, "real_dongshi_2024_revenue", False))


def test_same_score_with_one_baseline_pass_lost_fails(tmp_path: Path) -> None:
    def mutate(report: dict) -> None:
        _set_passed(report, "real_dongshi_2024_revenue", False)
        _set_passed(report, "real_dongshi_2024_vs_2023_revenue", True)

    with pytest.raises(BaselineGateError, match="回退了基线通过ID"):
        _run(tmp_path, mutate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["results"].pop(),
            "必须包含36条results",
        ),
        (
            lambda report: report["results"].__setitem__(1, copy.deepcopy(report["results"][0])),
            "重复case ID",
        ),
        (
            lambda report: report["summary"].__setitem__("error_count", 1),
            "error_count",
        ),
    ],
)
def test_missing_duplicate_or_error_report_fails(tmp_path: Path, mutation, message: str) -> None:
    with pytest.raises(BaselineGateError, match=message):
        _run(tmp_path, mutation)


def test_retrieval_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    def mutate(_report: dict) -> None:
        return None

    report_path, lineage_path = _candidate(tmp_path, mutate)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["backend"]["environment"]["KB_TOP_K"] = "6"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    with pytest.raises(BaselineGateError, match="KB_TOP_K"):
        check_candidate(BASELINE, report_path, lineage_path, project_root=PROJECT_ROOT)


def test_preserving_baseline_and_adding_a_passed_case_succeeds(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        lambda report: _set_passed(report, "real_dongshi_2024_vs_2023_revenue", True),
    )

    assert result["candidate_passed"] == 32
    assert result["newly_passed"] == ["real_dongshi_2024_vs_2023_revenue"]
