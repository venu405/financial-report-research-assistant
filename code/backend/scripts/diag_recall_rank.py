#!/usr/bin/env python3
"""诊断：期望证据块在召回排序中的真实名次。

区分两类检索失败：
  * 期望块排在 6~30 名 → 排序问题（进了池子但没进 top-5 / 被 rerank 挤掉）
  * 期望块 >30 名或完全不在 → 召回问题（向量/BM25 根本没捞到）

用法：
    .venv/Scripts/python.exe scripts/diag_recall_rank.py \
        --collection kb_full_codex_20260830_24c7407e --top-k 30 \
        --case real_yutai_2025_h1_revenue --case real_kelida_2024_net_profit
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

QDRANT = "http://127.0.0.1:16333"
TESTSET = "testsets/rag_real_quality_v2.yaml"


def load_cases(case_ids: list[str]) -> list[dict]:
    import yaml

    data = yaml.safe_load(
        (BACKEND_DIR / TESTSET).read_text(encoding="utf-8")
    )
    tests = data.get("tests") or data.get("cases") or []
    if not case_ids:
        return tests
    return [t for t in tests if t.get("id") in set(case_ids)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--kb-id", default="cninfo_report")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--show", type=int, default=12, help="每题打印前 N 条命中")
    args = ap.parse_args()

    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
    from services.kb.retriever import HybridRetriever  # noqa: E402

    store = QdrantVectorStore(
        url=QDRANT,
        collection_name=args.collection,
        vector_size=1024,
        create_if_missing=False,
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    retriever = HybridRetriever(store, embeddings=emb, top_k=args.top_k)

    cases = load_cases(args.case)
    print(f"collection={args.collection} kb_id={args.kb_id} top_k={args.top_k}")
    print(f"共 {len(cases)} 题\n")

    for case in cases:
        cid = case.get("id")
        question = case.get("question", "")
        evidence = case.get("expected_evidence") or []
        pages = case.get("expected_pages") or []

        hits = retriever.search(question, top_k=args.top_k, kb_id=args.kb_id)
        print("=" * 78)
        print(f"[{cid}] {question}")
        print(f"  期望页 {pages} | 期望证据 {evidence} | 命中 {len(hits)}")

        # 找期望证据首次出现的名次
        ranks: list[tuple[int, str]] = []
        for ev in evidence:
            compact = re.sub(r"\s+", "", ev).casefold()
            for i, h in enumerate(hits, 1):
                text = re.sub(r"\s+", "", (h.get("text") or "")).casefold()
                if compact in text:
                    ranks.append((i, ev))
                    break

        if ranks:
            for rank, ev in ranks:
                verdict = "进 top-5" if rank <= 5 else "在池内但未进 top-5"
                print(f"  ✓ 证据「{ev[:40]}」首次出现于第 {rank} 名 → {verdict}")
        else:
            print(f"  ✗ 期望证据在 top-{args.top_k} 内完全未出现")

        for i, h in enumerate(hits[: args.show], 1):
            md = h.get("metadata") or {}
            text = (h.get("text") or "").replace("\n", " ")
            mark = ""
            for ev in evidence:
                if re.sub(r"\s+", "", ev).casefold() in re.sub(r"\s+", "", (h.get("text") or "")).casefold():
                    mark = "  <<<< 命中期望证据"
                    break
            print(
                f"    #{i:<2} p{str(md.get('page')):<5} "
                f"score={round(float(h.get('score') or 0), 4):<7} "
                f"{str(md.get('source_path') or md.get('doc_title') or '')[-34:]}{mark}"
            )
            print(f"         {text[:120]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
