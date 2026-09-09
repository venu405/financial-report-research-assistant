#!/usr/bin/env python3
"""离线检索质量评估：不启 LLM，只测检索，用于快速 A/B 检索配置。

与 36 题端到端评测的区别：后者要起后端 + 逐题调 LLM（约 10 分钟/轮），
本脚本只构建一次 BM25 索引后纯检索，几分钟内可对比多组开关组合。

指标：
  doc@K  期望文档是否出现在 top-K（对应评测里的 hit@K）
  ev@K   期望证据文本是否出现在 top-K（对应 evidence_rate 的命中部分）
  page@K 期望页是否出现在 top-K

用法：
    .venv/Scripts/python.exe scripts/eval_retrieval_offline.py --top-k 20
    KB_BM25_RRF_WEIGHT=1 .venv/Scripts/python.exe scripts/eval_retrieval_offline.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

QDRANT = "http://127.0.0.1:16333"
COLLECTION = "kb_full_codex_20260830_24c7407e"
KB_ID = "cninfo_report"
TESTSET = "testsets/rag_real_quality_v2.yaml"
_STRIP_PIPE = False  # 由 --strip-pipe 打开


def _canon(value: str) -> str:
    """规范化：忽略 PDF 抽取的空白差异；--strip-pipe 时再忽略 markdown 表格分隔符。

    表格抽取把「营业收入 221,828,689.43」写成「| 营业收入 | 221,828,689.43 |」，
    `|` 是表格语法而非内容，不忽略会系统性低估 evidence 命中率。
    """
    text = str(value or "")
    text = re.sub(r"[\s|]+", "", text) if _STRIP_PIPE else re.sub(r"\s+", "", text)
    return text.casefold()


def _name_key(path: str) -> str:
    """取文件名主干，忽略目录前缀差异。"""
    base = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
    return _canon(re.sub(r"\.pdf$", "", base))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--kb-id", default=KB_ID)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--output", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strip-pipe", action="store_true",
                    help="比较时忽略 markdown 表格分隔符 |（更贴近真实证据命中）")
    ap.add_argument("--max-per-doc", type=int, default=0,
                    help="单文档候选上限；0=不限制。生产默认 3（kb_max_hits_per_doc）")
    args = ap.parse_args()
    global _STRIP_PIPE
    _STRIP_PIPE = args.strip_pipe

    import yaml  # noqa: E402
    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
    from services.kb.retriever import HybridRetriever  # noqa: E402

    data = yaml.safe_load((BACKEND_DIR / TESTSET).read_text(encoding="utf-8"))
    cases = data.get("tests") or data.get("cases") or []
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.get("id") in wanted]

    store = QdrantVectorStore(
        url=QDRANT, collection_name=args.collection,
        vector_size=1024, create_if_missing=False,
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    retriever = HybridRetriever(
        store, embeddings=emb, top_k=args.top_k,
        max_candidates_per_doc=(args.max_per_doc or None),
    )

    ks = (5, 10, args.top_k)
    totals = {k: {"doc": 0, "ev": 0, "page": 0} for k in ks}
    scored = {"doc": 0, "ev": 0, "page": 0}
    per_case: list[dict] = []

    print(
        f"collection={args.collection} top_k={args.top_k} "
        f"BM25权重={os.getenv('KB_BM25_RRF_WEIGHT', '0.5')} "
        f"页眉过滤={'关' if os.getenv('KB_FILTER_LOW_INFO_CHUNKS') == '0' else '开'} "
        f"单文档候选上限={args.max_per_doc or '不限'}"
    )
    print(f"共 {len(cases)} 题\n")

    for case in cases:
        cid = case.get("id", "?")
        question = case.get("question", "")
        evidence = [_canon(e) for e in (case.get("expected_evidence") or [])]
        pages = case.get("expected_pages") or []
        sources = [_name_key(s) for s in (case.get("expected_sources") or [])]

        hits = retriever.search(question, top_k=args.top_k, kb_id=args.kb_id)
        texts = [_canon(h.get("text")) for h in hits]
        metas = [h.get("metadata") or {} for h in hits]

        def doc_ok(k: int) -> bool:
            for m in metas[:k]:
                cand = _name_key(m.get("source_path") or m.get("doc_title") or "")
                if cand and any(s in cand or cand in s for s in sources):
                    return True
            return False

        def ev_ok(k: int) -> bool:
            return any(
                ev in t for ev in evidence for t in texts[:k]
            ) if evidence else False

        def page_ok(k: int) -> bool:
            if not pages:
                return False
            for m in metas[:k]:
                if m.get("page") in pages or m.get("page_start") in pages:
                    cand = _name_key(m.get("source_path") or m.get("doc_title") or "")
                    if any(s in cand or cand in s for s in sources):
                        return True
            return False

        row = {"id": cid, "n": len(hits)}
        for k in ks:
            row[f"doc@{k}"] = doc_ok(k)
            row[f"ev@{k}"] = ev_ok(k)
            row[f"page@{k}"] = page_ok(k)
            for metric in ("doc", "ev", "page"):
                if row[f"{metric}@{k}"]:
                    totals[k][metric] += 1
        if sources:
            scored["doc"] += 1
        if evidence:
            scored["ev"] += 1
        if pages:
            scored["page"] += 1
        per_case.append(row)
        if args.verbose:
            print(
                f"  {cid:<48} doc@5={int(row['doc@5'])} "
                f"ev@5={int(row['ev@5'])} ev@{args.top_k}={int(row[f'ev@{args.top_k}'])} "
                f"page@5={int(row['page@5'])}"
            )

    n = len(cases)
    print("=" * 74)
    print("汇总（分母为有该真值的题目数）")
    for metric, denom in (("doc", scored["doc"]), ("ev", scored["ev"]), ("page", scored["page"])):
        if not denom:
            continue
        line = f"  {metric:<5} "
        for k in ks:
            cnt = totals[k][metric]
            line += f"@{k}: {cnt}/{denom} ({cnt / denom:6.1%})   "
        print(line)

    if args.output:
        out = BACKEND_DIR / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"config": vars(args), "totals": totals,
                        "scored": scored, "cases": per_case},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
