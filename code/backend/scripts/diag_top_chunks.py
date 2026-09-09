#!/usr/bin/env python3
"""临时诊断：打印指定题目在给定开关下的 top-N 块（文本首 100 字 + 分数）。

用法（KB_BM25_RRF_WEIGHT 由环境变量控制）：
    set KB_BM25_RRF_WEIGHT=0 && python scripts/_tmp_top5.py --case X --case Y --top-k 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

QDRANT = "http://127.0.0.1:16333"
TESTSET = "testsets/rag_real_quality_v2.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--kb-id", default="cninfo_report")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--chars", type=int, default=110)
    args = ap.parse_args()

    import yaml

    data = yaml.safe_load((BACKEND_DIR / TESTSET).read_text(encoding="utf-8"))
    tests = data.get("tests") or data.get("cases") or []
    cases = [t for t in tests if t.get("id") in set(args.case)] if args.case else tests

    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
    from services.kb.retriever import HybridRetriever  # noqa: E402

    store = QdrantVectorStore(
        url=QDRANT, collection_name=args.collection, vector_size=1024, create_if_missing=False
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    retriever = HybridRetriever(store, embeddings=emb, top_k=args.top_k)

    print(f"[env] KB_BM25_RRF_WEIGHT={os.getenv('KB_BM25_RRF_WEIGHT', '(default 0.5)')}")
    for case in cases:
        q = case.get("question") or ""
        print("=" * 78)
        print(f"CASE {case.get('id')}")
        print(f"Q: {q}")
        hits = retriever.search(q, kb_id=args.kb_id, top_k=args.top_k)
        for i, h in enumerate(hits, 1):
            text = (h.get("text") or "").replace("\n", " ")
            meta = h.get("metadata") or {}
            print(
                f"  #{i} score={h.get('rrf_score', h.get('score', 0)):.4f} "
                f"p={meta.get('page', '?')} doc={meta.get('doc_id') or meta.get('source', '')[:18]}"
            )
            print(f"      {text[: args.chars]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
