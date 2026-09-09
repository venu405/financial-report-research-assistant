#!/usr/bin/env python3
"""在 Chroma 隔离副本上诊断某题的块级定位（只读查询，不重建、不写入）。

背景：阶段 C 门禁跑在 Chroma（13,351 块），而 Qdrant 是 73,662 块，两者切分
不同，不能在 Qdrant 上推断 Chroma 的失败原因。本脚本直接在 Chroma 副本里
查「装着目标数字的那个块」的 metadata，判断它是否满足结构化路由条件，
以及它在向量召回里的排位。

用法：
    .venv/Scripts/python.exe scripts/diag_chroma_block.py \
        --chroma-dir .rag_eval/chroma_repro_26_current_20260831/chroma_data \
        --collection enterprise_kb \
        --source-keyword 富润 --needle "110,682,912.05"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))


def route_matches(metadata: dict) -> tuple[bool, str]:
    """复算 retriever 结构化财务路由的过滤条件，返回 (是否命中, 未命中原因)。"""
    if not metadata.get("is_table"):
        return False, "is_table 非真"
    if not metadata.get("report_period"):
        return False, "report_period 为空"
    section = str(metadata.get("section_path") or "")
    if not re.search(r"主要会计数据|主要财务指标", section):
        return False, f"section_path 不匹配: {section!r}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", required=True)
    ap.add_argument("--collection", default="enterprise_kb")
    ap.add_argument("--source-keyword", required=True)
    ap.add_argument("--needle", required=True)
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(BACKEND_DIR / args.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:  # noqa: BLE001
        print(f"无法打开集合 {args.collection}: {exc}")
        return 1

    total = collection.count()
    print(f"集合 {args.collection}：{total} 条记录")

    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []
    print(f"已读取 {len(documents)} 条（只读）")

    hits = []
    for doc_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        path = str(meta.get("source_path") or meta.get("source") or "")
        if args.source_keyword not in path:
            continue
        if args.needle in str(text or ""):
            hits.append((doc_id, text, meta, path))

    print(f"\n含目标串 {args.needle!r} 的块：{len(hits)} 个")
    print("=" * 88)
    for doc_id, text, meta, path in hits:
        ok, why = route_matches(meta)
        print(f"id={doc_id}")
        print(f"  page={meta.get('page')} is_table={meta.get('is_table')} "
              f"period={meta.get('report_period')!r} "
              f"unit={meta.get('unit')!r} scope={meta.get('statement_scope')!r}")
        print(f"  section_path={meta.get('section_path')!r}")
        print(f"  table_name={meta.get('table_name')!r}")
        print(f"  doc_id={meta.get('doc_id')}")
        print(f"  路由命中={ok}" + ("" if ok else f"（原因：{why}）"))
        print(f"  src={path.split('/')[-1]}")
        print(f"  text[:220]={str(text)[:220]!r}")
        print("-" * 88)

    # 全公司满足路由条件的块，看正确块是否在其中
    routed = []
    for doc_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        path = str(meta.get("source_path") or meta.get("source") or "")
        if args.source_keyword not in path:
            continue
        if route_matches(meta)[0]:
            routed.append((doc_id, text, meta))
    print(f"\n该公司满足路由条件的块共 {len(routed)} 个：")
    for doc_id, text, meta in routed[:20]:
        mark = "★含目标串★" if args.needle in str(text or "") else ""
        print(f"  page={meta.get('page')} period={meta.get('report_period')!r} "
              f"{mark} section={str(meta.get('section_path'))[:40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
