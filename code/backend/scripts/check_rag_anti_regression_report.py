"""Validate an existing evaluate_quality.py RAG report without running evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_CASE_IDS = (
    "real_jinzhou_port_2025_plan_not_promise",
    "real_longyu_2024_revenue",
    "real_huawei_2024_revenue",
    "real_shenlian_2025_h1_operating_cash",
)
ALLOWED_KB_ID = "cninfo_report"


class ReportCheckError(ValueError):
    """A report failed the anti-regression contract."""


def _report_entries(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReportCheckError("报告顶层不是 JSON 对象")
    reports = payload.get("reports")
    if reports is None:
        reports = [payload]
    if not isinstance(reports, list) or not reports:
        raise ReportCheckError("报告缺少非空 reports 数组")
    entries = [item for item in reports if isinstance(item, dict)]
    if len(entries) != len(reports):
        raise ReportCheckError("reports 数组包含非对象项")
    return entries


def _select_rag_report(payload: Any) -> dict[str, Any]:
    entries = _report_entries(payload)
    required = set(REQUIRED_CASE_IDS)
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        results = entry.get("results")
        if not isinstance(results, list):
            continue
        result_ids = {
            item.get("id")
            for item in results
            if isinstance(item, dict)
        }
        if result_ids & required:
            candidates.append(entry)
    if len(candidates) == 0:
        raise ReportCheckError("报告中找不到四个关键 RAG case")
    if len(candidates) > 1:
        raise ReportCheckError("报告包含多个候选四题 RAG 报告，无法安全选择")
    return candidates[0]


def check_report(path: str | Path) -> None:
    """Raise ReportCheckError unless the existing report passes the contract."""
    report_path = Path(path)
    if not report_path.is_file():
        raise ReportCheckError(f"报告不存在: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise ReportCheckError(f"报告无法读取: {report_path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ReportCheckError(f"报告 JSON 损坏: 第 {exc.lineno} 行第 {exc.colno} 列") from exc

    report = _select_rag_report(payload)
    if report.get("run_status") != "completed":
        raise ReportCheckError(
            f"报告 run_status 不是 completed: {report.get('run_status')!r}"
        )

    summary = report.get("summary")
    if not isinstance(summary, dict) or "error_count" not in summary:
        raise ReportCheckError("报告缺少 summary.error_count")
    if summary.get("error_count") != 0:
        raise ReportCheckError(f"报告错误数非 0: {summary.get('error_count')!r}")

    results = report.get("results")
    if not isinstance(results, list):
        raise ReportCheckError("报告缺少 results 数组")
    by_id: dict[str, dict[str, Any]] = {}
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            if item["id"] in by_id:
                raise ReportCheckError(f"报告存在重复 case: {item['id']}")
            by_id[item["id"]] = item

    missing = [case_id for case_id in REQUIRED_CASE_IDS if case_id not in by_id]
    if missing:
        raise ReportCheckError(f"报告缺少关键 case: {', '.join(missing)}")

    for case_id in REQUIRED_CASE_IDS:
        case = by_id[case_id]
        if case.get("passed") is not True:
            raise ReportCheckError(f"关键 case 未通过: {case_id}")
        if "citation_kb_ids" not in case:
            raise ReportCheckError(f"关键 case 缺少 citation_kb_ids: {case_id}")
        citation_kb_ids = case["citation_kb_ids"]
        if not isinstance(citation_kb_ids, list):
            raise ReportCheckError(f"关键 case citation_kb_ids 格式错误: {case_id}")
        foreign = [
            str(kb_id)
            for kb_id in citation_kb_ids
            if kb_id is not None and str(kb_id).strip() != ALLOWED_KB_ID
        ]
        if foreign:
            raise ReportCheckError(
                f"关键 case 存在跨 KB 引用: {case_id} -> {', '.join(foreign)}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读检查既有 RAG 评测报告")
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args(argv)
    try:
        check_report(args.report_path)
    except ReportCheckError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    print("通过：四个关键 RAG case、run_status、错误数和 KB 隔离均符合门禁")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
