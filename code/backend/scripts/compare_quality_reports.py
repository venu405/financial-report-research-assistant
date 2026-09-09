#!/usr/bin/env python3
"""比较两份 evaluate_quality RAG 报告，输出指标差、风险警告与可选质量门槛。

用法示例：
  python scripts/compare_quality_reports.py \
      --baseline reports/rag_law.json --candidate reports/rag_law_v2.json \
      --require-gain "retrieval.contexts.hit@5=0.10" \
      --limit-regression "escalate_rate=0.05" \
      --limit-regression "latency.p95_s=3.0"

设计原则（防误导）：
  - 分母不同 / 案例集合不同 / 指标缺失 -> 明确警告；无任何可比指标时退出码 2
  - 样本量 < 10 -> 警告"差异不具统计显著性"，绝不伪造显著性检验
  - 门槛在指标缺失或不可比时判失败（宁可不通过，不静默放行）
  - 相对变化在基线为 0 时记 null（避免除零放大）

退出码：0=可比且门槛通过（或未配置门槛）；1=门槛未通过；2=输入错误或无可比指标。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

SMALL_SAMPLE = 10
_MISSING = object()

# 汇总里与检索层指标同名的分母键后缀
_DENOMINATOR_SUFFIX = "_denominator"
_RETRIEVAL_METRIC_RE = re.compile(r"^(hit@\d+|recall@\d+|evidence_rate)$")


def load_report(path: Path) -> dict[str, Any]:
    """读取报告 JSON；结构不符时抛 ValueError（由 main 转 exit 2）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取报告 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"报告不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"报告顶层必须是 JSON 对象: {path}")
    return data


def locate_rag_report(report: dict[str, Any]) -> dict[str, Any] | None:
    """兼容两种形态：新报告（RAG 汇总在顶层）与旧包装 {"reports": [...]}。"""
    if "summary" in report and report.get("kind", "rag") == "rag":
        return report
    for item in report.get("reports") or []:
        if isinstance(item, dict) and item.get("kind") == "rag":
            return item
    return None


def dig(obj: dict[str, Any], dotted: str) -> Any:
    """按点分路径取值。键不存在返回 _MISSING；值为 null 返回 None（区分二者）。"""
    current: Any = obj
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def metric_direction(name: str) -> str:
    """指标方向：high=越高越好；low=越低越好；none=中性（只展示不判优劣）。"""
    if name.startswith("latency."):
        return "low"
    if name == "escalate_rate":
        return "low"
    if name == "avg_citations":
        return "none"
    return "high"


def discover_metric_names(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """指标并集：固定答案层指标 + 双方 summary.retrieval.* 实际出现的检索指标。"""
    names = [
        "pass_rate",
        "answerable.accuracy",
        "escalate_rate",
        "avg_citations",
        "latency.p50_s",
        "latency.p95_s",
        "latency.max_s",
        "latency.avg_s",
    ]
    pools: set[tuple[str, str]] = set()
    for summary_source in (baseline, candidate):
        retrieval = summary_source.get("retrieval") or {}
        for pool, pool_data in retrieval.items():
            if not isinstance(pool_data, dict):
                continue
            for key in pool_data:
                if _RETRIEVAL_METRIC_RE.match(key):
                    pools.add((str(pool), key))
            page = pool_data.get("page")
            if isinstance(page, dict) and "hit_rate" in page:
                pools.add((str(pool), "page.hit_rate"))
    for pool, key in sorted(pools):
        names.append(f"retrieval.{pool}.{key}")
    return names


def denominator_path(metric: str) -> str | None:
    """检索指标 -> 对应分母键路径；答案层指标分母是 summary.total。"""
    if metric.startswith("retrieval."):
        if metric.endswith(".page.hit_rate"):
            return metric[: -len("hit_rate")] + "denominator"
        return metric + _DENOMINATOR_SUFFIX
    return "total"


def _case_ids(report: dict[str, Any]) -> list[str] | None:
    """案例标识：优先 id，缺失时退回 question（旧报告无 id）。"""
    results = report.get("results") or []
    if not results:
        return None
    if all(item.get("id") for item in results):
        return [str(item["id"]) for item in results]
    return [str(item.get("question") or "") for item in results]


def _describe_delta(row: dict[str, Any]) -> str:
    if row["status"] != "ok" or row["baseline"] is None or row["candidate"] is None:
        return row["status"]
    direction = row["direction"]
    if direction == "none" or row["abs_delta"] == 0:
        return "持平" if row["abs_delta"] == 0 else "变化"
    improved = row["abs_delta"] > 0 if direction == "high" else row["abs_delta"] < 0
    return "改善" if improved else "恶化"


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_path: str = "",
    candidate_path: str = "",
) -> dict[str, Any]:
    """核心比较（纯函数）。绝不抛异常，不可比情况写入 warnings/comparable=False。"""
    warnings: list[str] = []
    baseline_rag = locate_rag_report(baseline)
    candidate_rag = locate_rag_report(candidate)
    for label, report, path in (
        ("基线", baseline_rag, baseline_path),
        ("候选", candidate_rag, candidate_path),
    ):
        if report is None:
            warnings.append(f"{label}报告缺少 RAG 汇总（旧版报告无 summary）：{path or '(未提供路径)'}")
    if baseline_rag is None or candidate_rag is None:
        return {
            "comparable": False,
            "warnings": warnings,
            "metrics": [],
            "case_set": None,
            "baseline": _report_header(baseline_rag, baseline_path),
            "candidate": _report_header(candidate_rag, candidate_path),
        }

    base_summary = baseline_rag.get("summary") or {}
    cand_summary = candidate_rag.get("summary") or {}

    base_testset = str(baseline_rag.get("testset") or "")
    cand_testset = str(candidate_rag.get("testset") or "")
    if base_testset and cand_testset and base_testset != cand_testset:
        warnings.append(
            f"题集不同：基线={base_testset}，候选={cand_testset}；指标差异可能来自题集而非被测改动"
        )

    base_ids, cand_ids = _case_ids(baseline_rag), _case_ids(candidate_rag)
    only_baseline = sorted(set(base_ids or []) - set(cand_ids or [])) if base_ids and cand_ids else []
    only_candidate = sorted(set(cand_ids or []) - set(base_ids or [])) if base_ids and cand_ids else []
    if only_baseline or only_candidate:
        warnings.append(
            f"案例集合不同：仅基线 {len(only_baseline)} 例、仅候选 {len(only_candidate)} 例"
            f"（示例：{ (only_baseline + only_candidate)[:3] }）；命中类指标不可直接对比"
        )

    metrics: list[dict[str, Any]] = []
    for name in discover_metric_names(base_summary, cand_summary):
        base_value = dig(base_summary, name)
        cand_value = dig(cand_summary, name)
        if base_value is _MISSING and cand_value is _MISSING:
            continue
        if base_value is _MISSING:
            status = "missing_baseline"
        elif cand_value is _MISSING:
            status = "missing_candidate"
        elif base_value is None or cand_value is None:
            status = "not_scoreable"
        else:
            status = "ok"
        row: dict[str, Any] = {
            "name": name,
            "direction": metric_direction(name),
            "baseline": base_value if base_value is not _MISSING else None,
            "candidate": cand_value if cand_value is not _MISSING else None,
            "status": status,
        }
        if status == "ok":
            delta = round(float(cand_value) - float(base_value), 4)  # type: ignore[arg-type]
            row["abs_delta"] = delta
            base_float = float(base_value)  # type: ignore[arg-type]
            row["rel_change"] = (
                round((float(cand_value) - base_float) / base_float, 4) if base_float != 0 else None  # type: ignore[arg-type]
            )
        else:
            row["abs_delta"] = None
            row["rel_change"] = None
        row["verdict"] = _describe_delta(row)
        metrics.append(row)

        denom_path = denominator_path(name)
        if denom_path:
            base_denom = dig(base_summary, denom_path)
            cand_denom = dig(cand_summary, denom_path)
            if (
                base_denom is not _MISSING
                and cand_denom is not _MISSING
                and base_denom is not None
                and cand_denom is not None
                and base_denom != cand_denom
            ):
                warnings.append(
                    f"{name} 分母不同：基线 n={base_denom}，候选 n={cand_denom}"
                    "（真值覆盖或案例数不同），差异解释需谨慎"
                )

    comparable_rows = [row for row in metrics if row["status"] == "ok"]
    if not comparable_rows:
        warnings.append("两份报告没有任何可比较的指标（真值缺失或报告版本过旧）")

    base_total = base_summary.get("total")
    cand_total = cand_summary.get("total")
    for label, summary in (("基线", base_summary), ("候选", cand_summary)):
        if summary.get("error_count"):
            warnings.append(f"{label}报告存在 {summary['error_count']} 个请求错误案例，指标可能受基础设施故障影响")
    if (
        isinstance(base_total, int)
        and isinstance(cand_total, int)
        and min(base_total, cand_total) < SMALL_SAMPLE
    ):
        warnings.append(
            f"样本量小（基线 n={base_total}，候选 n={cand_total}，< {SMALL_SAMPLE}），"
            "差异不具统计显著性，仅作参考"
        )

    return {
        "comparable": bool(comparable_rows),
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "baseline": _report_header(baseline_rag, baseline_path),
        "candidate": _report_header(candidate_rag, candidate_path),
        "case_set": {
            "baseline_n": base_total,
            "candidate_n": cand_total,
            "only_in_baseline": only_baseline,
            "only_in_candidate": only_candidate,
        },
        "metrics": metrics,
        "warnings": warnings,
    }


def _report_header(report: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if report is None:
        return {"path": path, "testset": None, "label": None, "generated_at": None}
    return {
        "path": path,
        "testset": report.get("testset"),
        "label": report.get("label") or None,
        "generated_at": report.get("generated_at"),
    }


def _metric_row(diff: dict[str, Any], name: str) -> dict[str, Any]:
    for row in diff.get("metrics") or []:
        if row["name"] == name:
            return row
    raise ValueError(f"未知指标：{name}（不在两份报告的可比指标中）")


def parse_gate(spec: str) -> tuple[str, float]:
    """解析 'metric=value' 门槛声明。"""
    name, sep, raw_value = spec.partition("=")
    if not sep or not name.strip() or not raw_value.strip():
        raise ValueError(f"门槛格式错误（应为 指标=数值）：{spec}")
    try:
        return name.strip(), float(raw_value)
    except ValueError as exc:
        raise ValueError(f"门槛数值不合法：{spec}") from exc


def evaluate_gates(
    diff: dict[str, Any],
    require_gain: list[str],
    limit_regression: list[str],
) -> dict[str, Any]:
    """质量门槛：require-gain 必须改善至少 v；limit-regression 最多恶化 v。

    指标缺失/不可比/方向中性 -> 该门槛判失败（不静默放行）。
    """
    failures: list[str] = []
    for spec in require_gain:
        name, value = parse_gate(spec)
        row = _metric_row(diff, name)
        if row["direction"] == "none":
            failures.append(f"门槛 {spec}：{name} 是中性指标，不能作为门槛")
            continue
        if row["status"] != "ok":
            failures.append(f"门槛 {spec}：{name} 不可比较（{row['status']}），无法判定改善")
            continue
        gain = row["abs_delta"] if row["direction"] == "high" else -row["abs_delta"]
        if gain < value:
            failures.append(
                f"门槛未达成的要求改善 {name} ≥ {value}，实际变化 {row['abs_delta']:+.4f}"
            )
    for spec in limit_regression:
        name, value = parse_gate(spec)
        row = _metric_row(diff, name)
        if row["direction"] == "none":
            failures.append(f"门槛 {spec}：{name} 是中性指标，不能作为门槛")
            continue
        if row["status"] != "ok":
            failures.append(f"门槛 {spec}：{name} 不可比较（{row['status']}），无法限制恶化")
            continue
        regression = -row["abs_delta"] if row["direction"] == "high" else row["abs_delta"]
        if regression > value:
            failures.append(
                f"门槛超限的 {name} 最多恶化 {value}，实际变化 {row['abs_delta']:+.4f}"
            )
    return {
        "configured": bool(require_gain or limit_regression),
        "passed": not failures,
        "failures": failures,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary(diff: dict[str, Any], gates: dict[str, Any]) -> None:
    header_b, header_c = diff["baseline"], diff["candidate"]
    print(f"基线：{header_b.get('testset') or '?'} label={header_b.get('label') or '-'} ({header_b.get('path')})")
    print(f"候选：{header_c.get('testset') or '?'} label={header_c.get('label') or '-'} ({header_c.get('path')})")
    for warning in diff["warnings"]:
        print(f"⚠ {warning}")
    if not diff["metrics"]:
        print("（无可展示指标）")
    for row in diff["metrics"]:
        line = (
            f"{row['name']:<40} {_format_value(row['baseline']):>10} -> "
            f"{_format_value(row['candidate']):>10}"
        )
        if row["status"] == "ok":
            rel = (
                f"{row['rel_change'] * 100:+.1f}%"
                if row["rel_change"] is not None
                else "(基线为0，无法计算相对变化)"
            )
            line += f"  Δ{row['abs_delta']:+.4f} ({rel})  {row['verdict']}"
        else:
            line += f"  [{row['status']}]"
        print(line)
    if gates["configured"]:
        print(f"质量门槛：{'通过' if gates['passed'] else '未通过'}")
        for failure in gates["failures"]:
            print(f"✗ {failure}")


def configure_console_encoding() -> None:
    """让 Windows 的 GBK 控制台也能稳定输出中文及状态符号。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    # 展示层编码问题不能阻断 JSON 对比报告落盘。
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description=(
            "比较两份 evaluate_quality RAG 报告（A/B）。输出绝对差与相对变化，"
            "对分母不同/案例集合不同/指标缺失/小样本给出警告；"
            "可选 --require-gain / --limit-regression 质量门槛。"
            "退出码：0=可比且门槛通过；1=门槛未通过；2=输入错误或无可比指标。"
        )
    )
    parser.add_argument("--baseline", type=Path, required=True, help="基线报告 JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="候选报告 JSON")
    parser.add_argument(
        "--require-gain", action="append", default=[], metavar="METRIC=ABS",
        help="候选必须比基线改善至少 ABS（可重复），如 'retrieval.contexts.hit@5=0.10'",
    )
    parser.add_argument(
        "--limit-regression", action="append", default=[], metavar="METRIC=ABS",
        help="候选相对基线最多恶化 ABS（可重复），如 'latency.p95_s=3.0'",
    )
    parser.add_argument("--output", type=Path, help="差异报告 JSON 输出路径")
    args = parser.parse_args(argv)

    try:
        baseline = load_report(args.baseline)
        candidate = load_report(args.candidate)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    diff = compare_reports(
        baseline, candidate, baseline_path=str(args.baseline), candidate_path=str(args.candidate)
    )
    try:
        gates = evaluate_gates(diff, args.require_gain, args.limit_regression)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print_summary(diff, gates)
    if args.output:
        payload = dict(diff)
        payload["gates"] = gates
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"差异报告已写入 {args.output}")

    if not diff["comparable"]:
        return 2
    if gates["configured"] and not gates["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
