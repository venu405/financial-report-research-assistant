from __future__ import annotations

import json

import pytest

from scripts.check_rag_anti_regression_report import ReportCheckError, check_report

CASE_IDS = (
    "real_jinzhou_port_2025_plan_not_promise",
    "real_longyu_2024_revenue",
    "real_huawei_2024_revenue",
    "real_shenlian_2025_h1_operating_cash",
)


def _payload(
    *,
    run_status: str = "completed",
    error_count: int = 0,
    passed: bool = True,
    citation_kb_ids: list[str] | None = None,
    omit_case: str | None = None,
) -> dict:
    if citation_kb_ids is None:
        citation_kb_ids = ["cninfo_report"]
    results = [
        {
            "id": case_id,
            "passed": passed,
            "citation_kb_ids": citation_kb_ids,
        }
        for case_id in CASE_IDS
        if case_id != omit_case
    ]
    return {
        "reports": [
            {
                "run_status": run_status,
                "summary": {"error_count": error_count},
                "results": results,
            }
        ]
    }


def _write_report(tmp_path, payload: dict, name: str = "report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_valid_current_evaluate_quality_shape_passes(tmp_path):
    check_report(_write_report(tmp_path, _payload()))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_status": "in_progress"}, "run_status"),
        ({"error_count": 1}, "错误数"),
        ({"passed": False}, "关键 case 未通过"),
        ({"omit_case": CASE_IDS[1]}, "缺少关键 case"),
        ({"citation_kb_ids": ["cninfo_report", "other_kb"]}, "跨 KB"),
    ],
)
def test_invalid_report_contract_returns_clear_reason(tmp_path, changes, message):
    payload = _payload(**changes)
    with pytest.raises(ReportCheckError, match=message):
        check_report(_write_report(tmp_path, payload))


def test_missing_or_corrupt_report_fails(tmp_path):
    with pytest.raises(ReportCheckError, match="报告不存在"):
        check_report(tmp_path / "missing.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ReportCheckError, match="JSON 损坏"):
        check_report(corrupt)
