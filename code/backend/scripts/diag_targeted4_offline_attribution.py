"""离线归因：把历史基线（8-31 24/36）通过的正确答案喂给当前核验代码。

目的：判断本次 targeted4 的 0/4 是「核验层回归」还是「检索/生成波动」。
只读隔离 Chroma 副本，不调用任何 LLM，不调用后端服务，不写报告。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(BACKEND / "src") not in sys.path:
    sys.path.insert(0, str(BACKEND / "src"))

from scripts.diag_verify_real_block import load_real_blocks  # noqa: E402
from services.kb.qa_graph import (  # noqa: E402
    _extract_numeric_claims,
    _metric_alias_for_numeric_claim,
    _strict_revenue_field,
    _table_body_evidence,
    _verify_numeric_claims,
)

# (case_id, 文档关键词, 页码, 问句, 历史通过的正确答案)
CASES: list[tuple[str, str, int, str, str]] = [
    (
        "real_longyu_2024_revenue",
        "ST龙宇",
        7,
        "上海龙宇数据2024年营业收入是多少？",
        "上海龙宇数据2024年营业收入为1,404,920,973.02元，较上年同期减少55.02%。",
    ),
    (
        "real_huawei_2024_revenue",
        "ST华微",
        6,
        "吉林华微电子2024年营业收入是多少？",
        "吉林华微电子2024年度营业总收入为2,057,608,183.78元。",
    ),
    (
        "real_shenlian_2025_h1_operating_cash",
        "申联生物",
        7,
        "申联生物2025年上半年经营活动现金净流量是多少？",
        "申联生物2025年上半年经营活动产生的现金流量净额为-39,389,053.48元。",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chroma-dir",
        default=str(BACKEND / ".rag_eval" / "chroma_repro_26_current_20260831" / "chroma_data"),
    )
    parser.add_argument("--collection", default="enterprise_kb")
    parser.add_argument("--show-chars", type=int, default=260)
    args = parser.parse_args()

    regression = 0
    for case_id, keyword, page, question, historical_answer in CASES:
        print("=" * 78)
        print(f"CASE {case_id}")
        print(f"  Q: {question}")
        blocks = load_real_blocks(args.chroma_dir, args.collection, keyword, page)
        texts: list[str] = []
        for block in blocks:
            texts.extend(_table_body_evidence(block))
        print(f"  真实块 {len(blocks)} 个 -> 证据条数 {len(texts)}")
        if not texts:
            print("  !! 该页无可用表格证据，跳过")
            continue

        required = _strict_revenue_field(question)
        print(f"  _strict_revenue_field(question) = {required!r}")

        for text in texts[:6]:
            snippet = text[: args.show_chars].replace("\n", " | ")
            print(f"   证据: {snippet}")

        claims = _extract_numeric_claims(historical_answer)
        for claim in claims:
            alias = _metric_alias_for_numeric_claim(historical_answer, claim)
            print(
                f"  答案 claim raw={claim.raw!r} number={claim.number} "
                f"unit={claim.unit!r} alias={alias!r}"
            )

        ok, unsupported = _verify_numeric_claims(
            historical_answer, texts, question=question
        )
        verdict = "PASS" if ok else "REJECT"
        print(f"  >> 历史正确答案在当前代码下的核验结果: {verdict}")
        for claim in unsupported:
            print(
                f"       未通过 claim raw={claim.raw!r} number={claim.number} "
                f"metric={claim.metric!r} line_metric={claim.line_metric!r} "
                f"period={claim.report_period!r}"
            )
        if not ok:
            regression += 1

    print("=" * 78)
    print(f"被当前核验代码拒绝的历史正确答案数: {regression} / {len(CASES)}")
    print(
        "解读: 若 >0，说明 targeted4 的失败主因是核验层回归（代码变化导致），"
        "而非检索未召回或模型波动。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
