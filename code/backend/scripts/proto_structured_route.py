#!/usr/bin/env python3
"""原型验证：用真实 Qdrant 元数据过滤 API 实现财务结构化路由。

不改动任何生产代码，独立验证三件事：
  1. is_table / report_period / section_path 这些字段能不能被 Qdrant 直接过滤（无需建索引）
  2. 过滤后候选池有多小
  3. 在过滤后的小池子里做向量排序，正确块能不能排进 top-K

结论用于决定要不要把这条通道做进 retriever.py。

用法：
    .venv/Scripts/python.exe scripts/proto_structured_route.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_evidence_integrity import canonical, scroll_all  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTSET = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"
COLLECTION = "kb_full_codex_20260830_24c7407e"
QDRANT = "http://127.0.0.1:16333"

# 与 diag_structured_route.py 保持一致
KEY_SECTIONS = ("主要会计数据", "主要财务指标")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d+")


def period_from_question(question: str) -> str:
    q = str(question or "")
    m = re.search(r"(20\d{2})\s*年", q)
    if not m:
        return ""
    year = m.group(1)
    if "半年" in q or "中期" in q or "1-6" in q or "半年度" in q:
        return f"{year}年半年度"
    return f"{year}年度"


def main() -> int:
    import yaml

    tests = (yaml.safe_load(TESTSET.read_text(encoding="utf-8")) or {}).get("tests") or []

    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402

    store = QdrantVectorStore(
        url=QDRANT, collection_name=COLLECTION, vector_size=1024, create_if_missing=False
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")

    print("=" * 78)
    print("步骤 1：枚举 section_path 中含「主要会计数据」的全部取值")
    points = scroll_all(COLLECTION)
    payloads = [p.get("payload") or {} for p in points]
    section_values = sorted({
        str(pl.get("section_path"))
        for pl in payloads
        if any(k in str(pl.get("section_path") or "") for k in KEY_SECTIONS)
    })
    print(f"  全库 {len(payloads)} 块；命中章节的 section_path 取值共 {len(section_values)} 种")
    for v in section_values[:12]:
        print(f"      {v!r}")
    if not section_values:
        print("  ！没有任何 section_path 命中，路由不可行")
        return 1

    print()
    print("=" * 78)
    print("步骤 2：用 Qdrant where 过滤（is_table=True + report_period + section_path $in）")
    targets = [
        "real_kelida_2024_net_profit",
        "real_zhongcheng_2024_net_profit",
        "real_shenlian_2025_h1_rnd_ratio_synonym",
        "real_gree_2023_core_table_fields",
        "real_huawei_2024_revenue",
        "real_jiuyou_2024_net_profit",
    ]
    ok = 0
    for t in tests:
        if t.get("id") not in targets:
            continue
        question = t.get("question") or ""
        period = period_from_question(question)
        vec = emb.embed_query(question)

        # 阶段一：常规向量检索，确定"问的是哪几家公司"（取 top-20 涉及的 doc_id）
        plain = store.search(vec, top_k=20, kb_id="cninfo_report")
        doc_ids = sorted({
            str((h.get("metadata") or {}).get("doc_id"))
            for h in plain
            if (h.get("metadata") or {}).get("doc_id")
        })

        # 阶段二：元数据过滤（公司 + 期间 + 会计数据章节），在小池子里重新排序
        where = {
            "$and": [
                {"is_table": True},
                {"report_period": period},
                {"section_path": {"$in": section_values}},
                {"doc_id": {"$in": doc_ids}},
            ]
        }
        try:
            hits = store.search(vec, top_k=20, where=where, kb_id="cninfo_report")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{t['id']}] 过滤查询失败: {type(exc).__name__}: {exc}")
            continue

        spec = [ev for ev in (t.get("expected_evidence") or []) if NUM_RE.search(ev)]
        spec = spec or (t.get("expected_evidence") or [])
        keys = [canonical(ev) for ev in spec]
        nums = [
            n for ev in spec for n in NUM_RE.findall(ev)
            if len(n.replace(",", "").replace(".", "").replace("-", "")) >= 4
        ]

        rank = None
        for i, h in enumerate(hits, 1):
            txt = canonical((h.get("metadata") or {}).get("text") or h.get("text") or "")
            if any(k in txt for k in keys) or any(canonical(n) in txt for n in nums):
                rank = i
                break
        pool = len(hits)
        if rank:
            ok += 1
        print(f"  [{t['id']}] period={period} 公司doc={len(doc_ids)} 过滤后候选={pool} "
              f"正确块名次={'#' + str(rank) if rank else '未进 top-20'} "
              f"期望页={t.get('expected_pages')}")

    print()
    print("=" * 78)
    print(f"结构化路由原型：{ok}/{len(targets)} 题正确块进入过滤后 top-20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
