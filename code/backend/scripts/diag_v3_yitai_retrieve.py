#!/usr/bin/env python3
"""诊断：伊泰现金流题在分离实验档中的检索与 facts 情况。

1. 用 Qdrant REST scroll 按 doc_id + page 取第 6 页 chunks，看 financial_facts_json
2. 用生产 HybridRetriever 检索问题，看 top-5 命中什么
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ["KB_SEPARATE_TABLE_MD"] = "1"

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

COLLECTION = "kb_exp_v3_sep_20260901"
QDRANT = "http://127.0.0.1:16333"
REL = "data_kb_test/cninfo/伊泰Ｂ股-内蒙古伊泰煤炭股份有限公司2025年半年度报告.pdf"
QUERY = "伊泰煤炭2025年上半年经营活动产生的现金流量净额是多少？"
TARGET = "3,898,863,782.71"
TARGET_COMPACT = "3898863782.71"


def stable_doc_id(rel_path: str) -> str:
    return "doc-" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:32]


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{QDRANT}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _facts(payload: dict) -> list[dict]:
    raw = payload.get("financial_facts_json") or "[]"
    try:
        return json.loads(raw)
    except Exception:
        return []


def main() -> int:
    doc_id = stable_doc_id(REL)

    print("=" * 78)
    print(f"[1] 全量拉取该文档 chunks（doc_id={doc_id}）")
    pts: list[dict] = []
    offset = None
    while True:
        body: dict = {
            "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
            "limit": 500,
            "with_payload": True,
            "with_vectors": False,
        }
        if offset is not None:
            body["offset"] = offset
        resp = _post(f"/collections/{COLLECTION}/points/scroll", body)
        res = resp.get("result", {})
        pts.extend(res.get("points", []))
        offset = res.get("next_page_offset")
        if not offset:
            break
    print(f"    共 {len(pts)} 个 chunk")

    pages: dict = {}
    for p in pts:
        pl = p.get("payload") or {}
        pg = pl.get("page")
        pages.setdefault(pg, {"n": 0, "facts": 0, "target": 0})
        pages[pg]["n"] += 1
        fl = _facts(pl)
        pages[pg]["facts"] += len(fl)
        if TARGET in (pl.get("financial_facts_json") or "") or TARGET in (pl.get("text") or ""):
            pages[pg]["target"] += 1
    print("    页分布（page: chunks/facts/含目标值）:")
    for pg in sorted(k for k in pages if k is not None):
        v = pages[pg]
        print(f"      p{pg}: {v['n']} chunks, {v['facts']} facts, 含目标 {v['target']}")

    print("    含目标值的 chunk 明细:")
    for p in pts:
        pl = p.get("payload") or {}
        raw = pl.get("financial_facts_json") or ""
        txt = pl.get("text") or ""
        if TARGET in raw or TARGET in txt:
            print(f"      idx={pl.get('chunk_index')} page={pl.get('page')} "
                  f"in_facts={TARGET in raw} in_text={TARGET in txt}")
            for f in _facts(pl):
                if TARGET in str(f.get("raw_value", "")):
                    print(f"        * {f.get('metric')} = {f.get('raw_value')}")
            print(f"        text: {txt[:200].replace(chr(10), ' ')}")

    print("    第 6 页 chunk 明细:")
    for p in pts:
        pl = p.get("payload") or {}
        if pl.get("page") == 6:
            print(f"      idx={pl.get('chunk_index')} is_table={pl.get('is_table')} "
                  f"facts={len(_facts(pl))}")
            print(f"        text: {(pl.get('text') or '')[:200].replace(chr(10), ' ')}")

    print()
    print("=" * 78)
    print("[2] HybridRetriever 检索 top-5")
    from services.kb.embeddings import EmbeddingClient  # noqa: E402
    from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
    from services.kb.retriever import HybridRetriever  # noqa: E402

    store = QdrantVectorStore(
        url=QDRANT, collection_name=COLLECTION, vector_size=1024, create_if_missing=False
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    retriever = HybridRetriever(store, embeddings=emb, top_k=5)
    hits = retriever.search(QUERY, top_k=5, kb_id="cninfo_report")
    for i, h in enumerate(hits, 1):
        md = h.get("metadata") or {}
        txt = (h.get("text") or "").replace("\n", " ")
        fl = _facts(md)
        hit = TARGET in txt or TARGET in (md.get("financial_facts_json") or "")
        print(f"    #{i} [{'HIT' if hit else 'miss'}] page={md.get('page')} facts={len(fl)} "
              f"src={str(md.get('source_path', ''))[-40:]}")
        print(f"        {txt[:140]}")
        for f in fl[:6]:
            print(f"          * {f.get('metric')} = {f.get('raw_value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
