#!/usr/bin/env python3
"""模拟「改题集」能带来多少通过题的增收，把题集问题与系统问题分开。

背景：RAG 评测分下降时，必须先分清是「系统答错了」还是「题集判据写窄了」。
本脚本不重新跑模型，只在既有报告的答案上重算判据，量化三种改题策略的增收。

三种策略：
  A 修正判据缺陷   —— 只放开「单位换算的四舍五入」
                      （如 22,182.87万元 与 221,828,689.43 等价却被判不等）
  B A + 放水       —— 再把 expect_keywords_all 从「全部命中」降到「命中 1 个」
  C B + 放弃拒答题 —— 再把 escalate 题的 answerable/min_citations/inline_citations
                      一律视为通过（纯数字游戏，仅用于说明上限，不建议使用）

注意：B/C 会把「系统只答对一半」甚至「答错口径」的题判为通过，
用来刷分可以，用来衡量真实能力会掩盖幻觉。

用法：
    .venv/Scripts/python.exe scripts/sim_testset_relax.py \
        --report reports/qdrant-full36-pctfix-20260901.json \
        --testset testsets/rag_real_quality_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

import yaml  # noqa: E402

from evaluate_quality import _numeric_tokens, numeric_semantically_matches  # noqa: E402

# 相对误差阈值：单位换算后允许四舍五入（22,182.87万元 -> 221,828,700 元，误差 ~5e-8）
ROUND_TOL = Decimal("0.001")


def loose_match(expected: str, answer: str) -> bool:
    """在既有严格匹配之上，再允许单位换算造成的四舍五入差异。"""
    if numeric_semantically_matches(expected, answer):
        return True
    expected_tokens = _numeric_tokens(expected)
    answer_tokens = _numeric_tokens(answer)
    if not expected_tokens or not answer_tokens:
        return False
    for exp in expected_tokens:
        for act in answer_tokens:
            # 百分比不做放宽：22.32 与 22.32% 语义不同，放开会把口径错误放进来
            if exp["unit"] in ("%", "％") or act["unit"] in ("%", "％"):
                continue
            if act["canonical"] == 0:
                continue
            rel = abs(exp["canonical"] - act["canonical"]) / abs(act["canonical"])
            if rel < ROUND_TOL:
                return True
    return False


def all_pass(checks: dict) -> bool:
    return all(v is not False for v in checks.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--testset", default="testsets/rag_real_quality_v2.yaml")
    parser.add_argument("--testset-root", default=str(BACKEND_DIR))
    args = parser.parse_args()

    root = Path(args.testset_root)
    report_path = root / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["reports"][0]["results"]
    cases = {t["id"]: t for t in yaml.safe_load(
        (root / args.testset).read_text(encoding="utf-8"))["tests"]}

    rows = []
    for r in results:
        case = cases.get(r["id"], {})
        checks = dict(r["checks"])
        answer = r["answer"]

        # ---- 策略 A：放开单位换算四舍五入 ----
        a_checks = dict(checks)
        if "expect_keyword" in case and not checks.get("keyword", True):
            a_checks["keyword"] = loose_match(case["expect_keyword"], answer)

        # ---- 策略 B：keywords_all 降到命中 1 个 ----
        b_checks = dict(a_checks)
        if "expect_keywords_all" in case and not checks.get("keywords_all", True):
            hits = sum(1 for w in case["expect_keywords_all"] if loose_match(w, answer))
            b_checks["keywords_all"] = hits >= 1

        # ---- 策略 C：连拒答题的引用/可答性检查也免掉 ----
        c_checks = dict(b_checks)
        if r.get("escalate"):
            for key in ("answerable", "min_citations", "inline_citations"):
                if c_checks.get(key) is False:
                    c_checks[key] = True

        rows.append({
            "id": r["id"],
            "base": r["passed"],
            "A": all_pass(a_checks),
            "B": all_pass(b_checks),
            "C": all_pass(c_checks),
            "escalate": bool(r.get("escalate")),
            "hit@5": r.get("retrieval", {}).get("contexts", {}).get("hit@5"),
            "evidence_rate": r.get("retrieval", {}).get("contexts", {}).get("evidence_rate"),
            "failed_checks": [k for k, v in checks.items() if v is False],
        })

    total = len(rows)
    base = sum(1 for x in rows if x["base"])
    print(f"报告: {report_path.name}   题数: {total}")
    print(f"  现状(BASE)                                : {base}/{total}")
    print(f"  A 只修正判据缺陷(单位换算四舍五入)        : "
          f"{sum(1 for x in rows if x['A'])}/{total}   (+{sum(1 for x in rows if x['A']) - base})")
    print(f"  B A + keywords_all 降到命中1个(放水)      : "
          f"{sum(1 for x in rows if x['B'])}/{total}   (+{sum(1 for x in rows if x['B']) - base})")
    print(f"  C B + 免掉拒答题检查(纯数字游戏)          : "
          f"{sum(1 for x in rows if x['C'])}/{total}   (+{sum(1 for x in rows if x['C']) - base})")

    print("\n--- 失败题明细 ---")
    for x in rows:
        if x["base"]:
            continue
        gain = "A" if x["A"] else ("B" if x["B"] else ("C" if x["C"] else "-"))
        kind = "拒答" if x["escalate"] else "作答"
        print(f"{x['id']:52s} {kind} hit@5={str(x['hit@5']):5s} "
              f"ev={x['evidence_rate']} 增收={gain:2s} bad={x['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
