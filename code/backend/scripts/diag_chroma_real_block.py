#!/usr/bin/env python3
"""按 source_path 子串（可选页码）从隔离 Chroma 副本中取出**真实**入库块。

用途：核验探针此前用的是手写重建表格，可能与真实入库文本不一致（换行、千分位、
行标签被拆行、单位等）。本脚本只读真实块，用于定位「证据召到了却被拒答」的
真实断链点在检索/切块侧还是核验侧。

用法：
    .venv/Scripts/python.exe scripts/diag_chroma_real_block.py \
        --chroma-dir .rag_eval/chroma_repro_26_current_20260831/chroma_data \
        --collection enterprise_kb \
        --source-keyword "ST富润-2024年年度报告" --page 5 --show-chars 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", required=True)
    ap.add_argument("--collection", default="enterprise_kb")
    ap.add_argument("--source-keyword", required=True)
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--show-chars", type=int, default=2000)
    ap.add_argument("--show-meta", action="store_true")
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(BACKEND_DIR / args.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_collection(args.collection)
    total = coll.count()
    print(f"collection={args.collection} count={total}")

    got = coll.get(include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    print(f"拉取记录数={len(ids)}")

    hits = []
    for i, meta in enumerate(metas):
        meta = meta or {}
        path = str(
            meta.get("source_path")
            or meta.get("source")
            or meta.get("doc_title")
            or ""
        )
        if args.source_keyword not in path:
            continue
        if args.page is not None and meta.get("page") != args.page:
            continue
        hits.append((i, path, docs[i] if i < len(docs) else "", meta))

    print(f"命中块数={len(hits)}")
    # 先按页号排序，便于观察同一页的多个块
    hits.sort(key=lambda x: x[3].get("page") if isinstance(x[3].get("page"), int) else 0)

    for idx, (i, path, text, meta) in enumerate(hits):
        print("=" * 78)
        print(f"[块 {idx}] id={ids[i] if i < len(ids) else '?'} page={meta.get('page')} "
              f"is_table={meta.get('is_table')} unit={meta.get('unit')} "
              f"period={meta.get('report_period')} doc_id={meta.get('doc_id')}")
        if args.show_meta:
            print("metadata:", json.dumps(meta, ensure_ascii=False))
        print("-" * 78)
        print(text[: args.show_chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
