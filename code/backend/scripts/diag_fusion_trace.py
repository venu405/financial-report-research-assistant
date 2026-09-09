#!/usr/bin/env python3
"""手工重现 HybridRetriever 的向量 → BM25 → RRF 融合全过程，定位目标块在哪一步丢失。

用法：
    .venv/Scripts/python.exe scripts/diag_fusion_trace.py \
        --question "裕太微电子2025年上半年营业收入具体是多少？" \
        --needle "221,828,689.43"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

QDRANT = "http://127.0.0.1:16333"
COLLECTION = "kb_full_codex_20260830_24c7407e"
KB_ID = "cninfo_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--needle", required=True, help="目标证据串，用于定位目标块")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
    from services.kb.retriever import (  # noqa: E402
        RRF_K,
        HybridRetriever,
        _tokenize,
    )

    store = QdrantVectorStore(
        url=QDRANT, collection_name=args.collection,
        vector_size=1024, create_if_missing=False,
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")

    # ---- 1. 向量通道 ----
    qvec = emb.embed_query(args.question)
    vec_hits = store.search(qvec, top_k=args.top_k, kb_id=KB_ID)
    vec_ranks = {h["chunk_id"]: i for i, h in enumerate(vec_hits)}
    tgt_ids = {
        h["chunk_id"]: i
        for i, h in enumerate(vec_hits)
        if args.needle in (h.get("text") or "")
    }
    print(f"[1] 向量通道 top-{args.top_k}: {len(vec_hits)} 条")
    print(f"    目标块名次: {sorted(tgt_ids.values())}  chunk_id: {list(tgt_ids)}")

    # ---- 2. BM25 通道（复用检索器内部的语料缓存）----
    print("[2] 构建 BM25 语料（首次较慢）…", flush=True)
    retriever = HybridRetriever(store, embeddings=emb, top_k=args.top_k)
    retriever._rebuild_if_needed(KB_ID)
    corpus = retriever._corpus.get(KB_ID, [])
    bm25 = retriever._bm25.get(KB_ID)
    print(f"    corpus={len(corpus)} 条, bm25={'已构建' if bm25 else '缺失'}")

    bm25_ranks: dict[str, int] = {}
    if bm25 and corpus:
        tokens = _tokenize(args.question)
        print(f"    查询 tokens({len(tokens)}): {tokens[:20]}")
        scores = bm25.get_scores(tokens)
        pos = int((scores > 0).sum())
        neg = int((scores < 0).sum())
        zero = int((scores == 0).sum())
        print(
            f"    BM25 分数分布: >0 的 {pos} 条, <0 的 {neg} 条, ==0 的 {zero} 条 "
            f"| max={scores.max():.4f} min={scores.min():.4f}"
        )
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank = 0
        for idx in order:
            if rank >= args.top_k:
                break
            if scores[idx] > 0:
                bm25_ranks[corpus[idx]["chunk_id"]] = rank
                rank += 1
        print(f"    BM25 参与融合: {len(bm25_ranks)} 条")
        corpus_ids = {c["chunk_id"] for c in corpus}
        print(f"    目标块在 corpus 中: {bool(set(tgt_ids) & corpus_ids)}")
        for cid in tgt_ids:
            if cid in bm25_ranks:
                print(f"    目标块 BM25 名次: {bm25_ranks[cid]}")
    # chunk_id 空间是否一致
    if bm25_ranks:
        overlap = len(set(vec_ranks) & set(bm25_ranks))
        print(f"    向量/BM25 chunk_id 交集: {overlap}")

    # ---- 3. RRF 融合 ----
    fused: dict[str, float] = {}
    for cid, rank in vec_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for cid, rank in bm25_ranks.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[3] RRF 融合总数: {len(ordered)}（RRF_K={RRF_K}）")
    for cid in tgt_ids:
        if cid in fused:
            pos = [i for i, (c, _s) in enumerate(ordered, 1) if c == cid]
            print(f"    目标块融合分 {fused[cid]:.5f} → 名次 {pos}")
        else:
            print(f"    目标块不在融合结果中！")

    print("\n[4] 融合后前 12 名（手工重现，未含页眉过滤）:")
    for i, (cid, score) in enumerate(ordered[:12], 1):
        mark = "  <<<< 目标块" if cid in tgt_ids else ""
        vec = vec_hits[vec_ranks[cid]] if cid in vec_ranks else None
        page = (vec or {}).get("metadata", {}).get("page") if vec else "?"
        print(f"    #{i:<3} {score:.5f}  p{page}  {cid}{mark}")

    # ---- 5. 真实检索器端到端 ----
    print("\n[5] HybridRetriever.search 端到端（含页眉过滤）top-20:")
    real = retriever.search(args.question, top_k=20, kb_id=KB_ID)
    for i, h in enumerate(real, 1):
        text = (h.get("text") or "").replace("\n", " ")
        hit = args.needle in (h.get("text") or "")
        page = (h.get("metadata") or {}).get("page")
        print(f"    #{i:<3} p{str(page):<5} {'HIT ' if hit else '    '} {text[:88]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
