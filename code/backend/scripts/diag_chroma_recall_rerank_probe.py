#!/usr/bin/env python3
"""P1 诊断：在**纯检索层**（无 LLM rerank）检查目标块的召回情况。

用途：判断 targeted4 里 longyu / shenlian cash 的 `page_hit=False` 到底是
向量+BM25 召回失败，还是 `KB_RERANK_MODE=llm` 的 rerank 抖动。

向量 + BM25 + RRF 是确定性的；LLM rerank 不是。所以：
  * 纯检索 page_hit=True  而评测 page_hit=False  -> 抖动来自 rerank/后续环节
  * 纯检索 page_hit=False 且深层也没有          -> 向量召回本身不足

只读隔离 Chroma 副本，调用本地 Ollama bge-m3（免费），不调用任何云端 LLM。

用法：
    .venv/Scripts/python.exe scripts/diag_chroma_recall_rerank_probe.py --top-k 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

# (case_id, 问句, 期望页, 目标数字)
PROBES: list[tuple[str, str, list[int], str]] = [
    (
        "real_longyu_2024_revenue",
        "上海龙宇数据2024年营业收入是多少？",
        [7],
        "1,404,920,973.02",
    ),
    (
        "real_shenlian_2025_h1_operating_cash",
        "申联生物2025年上半年经营活动现金净流量是多少？",
        [7],
        "39,389,053.48",
    ),
    (
        "real_huawei_2024_revenue",
        "吉林华微电子2024年营业收入是多少？",
        [6],
        "2,057,608,183.78",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chroma-dir",
        default=str(
            BACKEND_DIR / ".rag_eval" / "chroma_repro_26_current_20260831" / "chroma_data"
        ),
    )
    ap.add_argument("--collection", default="enterprise_kb")
    ap.add_argument("--kb-id", default="cninfo_report")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.retriever import HybridRetriever  # noqa: E402
    from services.kb.vector_store import VectorStore  # noqa: E402

    store = VectorStore(
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model="bge-m3",
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    retriever = HybridRetriever(store, embeddings=emb, top_k=args.top_k)

    print(f"chroma_dir={args.chroma_dir}")
    print(f"collection={args.collection} kb_id={args.kb_id} top_k={args.top_k}")
    print("模式：纯检索（HybridRetriever.search，不经任何 rerank）\n")

    for case_id, question, pages, target in PROBES:
        hits = retriever.search(question, top_k=args.top_k, kb_id=args.kb_id)
        print("=" * 78)
        print(f"[{case_id}] {question}")
        print(f"  期望页={pages} 目标数字={target} 命中数={len(hits)}")

        target_rank = None
        page_rank = None
        for idx, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            text = str(hit.get("text") or "")
            page = metadata.get("page")
            has_target = target in text
            if has_target and target_rank is None:
                target_rank = idx
            if page_rank is None and page in pages:
                page_rank = idx
            if idx <= args.show or has_target:
                mark = "  <<< 含目标数字" if has_target else ""
                src = str(metadata.get("doc_title") or "")[-32:]
                print(
                    f"  [{idx:>2}] page={str(page):<5} score={hit.get('score', 0):.4f} "
                    f"...{src}{mark}"
                )

        print(f"  --> 目标数字首次出现排名: {target_rank or '未出现'}")
        print(f"  --> 期望页首次出现排名  : {page_rank or '未出现'}")
        top5_pages = [
            (hit.get("metadata") or {}).get("page") for hit in hits[:5]
        ]
        top5_has_target = any(
            target in str(hit.get("text") or "") for hit in hits[:5]
        )
        print(f"  --> top-5 页码: {top5_pages}")
        print(
            f"  --> top-5 是否含目标数字: {top5_has_target} "
            f"(纯检索 page_hit={'True' if page_rank and page_rank <= 5 else 'False'})"
        )
    print("=" * 78)
    print(
        "判读：纯检索确定性。若某题纯检索 top-5 含目标数字，"
        "而评测报告里 page_hit=False，则抖动来自 rerank 或后续环节。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
