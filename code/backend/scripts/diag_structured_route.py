#!/usr/bin/env python3
"""诊断：财务问答的「结构化元数据路由」可行性量化。

背景（本轮最关键发现）：
  财务数字问答失败的真正原因不是"文档没召回"，而是"块级定位失效"——
  正确文档稳定排在 top-10，但装着正确数字的那一块连 top-100 都进不去。
  原因：表格数字块在语义向量空间里与问句并不相似，而 BM25 也捞不到
  （问句里根本没有那串数字）。纯语义/关键词排序天然定位不到
  "某页主要会计数据表里的某一行"。

  但索引里已经存在可用的结构化元数据：section_path / is_table / table_name /
  report_period / doc_id。本脚本验证：用这些元数据做前置过滤，能把候选池
  从 7 万块收敛到多小，以及正确块的命中率是多少。

用法：
    .venv/Scripts/python.exe scripts/diag_structured_route.py
    .venv/Scripts/python.exe scripts/diag_structured_route.py --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_evidence_integrity import canonical, scroll_all  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTSET = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"

# 财报里"主要会计数据"表的章节标题写法（与 section_path 做子串匹配）
KEY_SECTIONS = ("主要会计数据", "主要财务指标")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d+")


def period_from_question(question: str) -> str:
    """从问句推断 report_period 元数据取值（与 ingest 写入格式保持一致）。"""
    q = str(question or "")
    m = re.search(r"(20\d{2})\s*年", q)
    if not m:
        return ""
    year = m.group(1)
    if "半年" in q or "中期" in q or "1-6" in q or "半年度" in q:
        return f"{year}年半年度"
    return f"{year}年度"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import yaml

    tests = (yaml.safe_load(TESTSET.read_text(encoding="utf-8")) or {}).get("tests") or []

    print(f"扫描 {args.collection} ...", flush=True)
    points = scroll_all(args.collection)
    payloads = [p.get("payload") or {} for p in points]
    print(f"共 {len(payloads)} 块\n", flush=True)

    table_chunks = [
        pl for pl in payloads
        if pl.get("is_table") and any(k in str(pl.get("section_path") or "") for k in KEY_SECTIONS)
    ]
    print(f"「主要会计数据/主要财务指标」表格块：{len(table_chunks)} "
          f"({len(table_chunks) / max(len(payloads), 1):.1%})")
    print("  按 report_period 分布：",
          dict(Counter(str(pl.get("report_period")) for pl in table_chunks).most_common(8)))
    print()

    stats = Counter()
    for t in tests:
        evidences = t.get("expected_evidence") or []
        if not evidences:
            continue
        question = t.get("question") or ""
        period = period_from_question(question)
        if not period:
            stats["skip_no_period"] += 1
            continue

        # 第一级：期间 + 章节
        pool = [pl for pl in table_chunks if str(pl.get("report_period")) == period]
        if not pool:
            stats["period_pool_empty"] += 1
            print(f"[期间池为空] {t.get('id')} period={period}")
            continue

        # 第二级：叠加"证据所在文档"过滤（模拟公司已由检索确定）
        # 只用"含数字的最具体证据项"判定命中。
        # 注意：expected_evidence 里常有 '主要会计数据' 这类通用短语（全库 424 块都有），
        # 拿它做子串匹配会把候选池虚报到全库规模，必须排除。
        specific = [ev for ev in evidences if NUM_RE.search(ev)] or evidences
        hit_chunks: list[dict] = []
        for ev in specific:
            can = canonical(ev)
            hit_chunks.extend(pl for pl in pool if can in canonical(pl.get("text", "")))
        if not hit_chunks:
            stats["not_in_period_pool"] += 1
            print(f"[期间池未命中] {t.get('id')} period={period} 池={len(pool)}")
            continue

        doc_ids = {pl.get("doc_id") for pl in hit_chunks}
        per_doc = [pl for pl in pool if pl.get("doc_id") in doc_ids]
        stats["routed"] += 1
        print(f"[{t.get('id')}]")
        print(f"    期间池={len(pool)} -> 叠加公司后={len(per_doc)} "
              f"(命中块 {len(hit_chunks)} 个, page={sorted({h.get('page') for h in hit_chunks})})")
        if args.verbose:
            for pl in per_doc[:6]:
                txt = re.sub(r"\s+", " ", str(pl.get("text") or ""))[:90]
                print(f"        p{pl.get('page')} | {txt}")

    print("\n" + "=" * 78)
    print("路由可行性汇总：", dict(stats))
    total = sum(v for k, v in stats.items() if k in ("routed", "not_in_period_pool", "period_pool_empty"))
    if total:
        print(f"期间+章节路由命中率：{stats['routed']}/{total} "
              f"({stats['routed'] / total:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
