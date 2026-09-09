#!/usr/bin/env python3
"""跨档对比：伊泰题在「旧档(含 markdown)」与「分离档」的 top-5 上下文差异。

目的：验证假设——核验层依赖「指标名与数字同行」的结构，
旧档靠 find_tables markdown 提供，分离档只留干净正文（指标名被 PDF 换行拆开）导致绑定失败。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["KB_SEPARATE_TABLE_MD"] = "1"

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

from services.kb.embeddings import EmbeddingClient  # noqa: E402
from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402
from services.kb.retriever import HybridRetriever  # noqa: E402

QDRANT = "http://127.0.0.1:16333"
COLLECTIONS = [
    ("旧档(markdown在text)", "kb_full_codex_20260830_24c7407e"),
    ("去find_tables档", "kb_full_v2_noisefix_20260831"),
    ("分离档", "kb_exp_v3_sep_20260901"),
]
QUERY = "伊泰煤炭2025年上半年经营活动产生的现金流量净额是多少？"
TARGET = "3,898,863,782.71"
METRIC_KEY = "经营活动产生的现金流量净额"


def _facts(md: dict) -> list[dict]:
    raw = md.get("financial_facts_json") or "[]"
    try:
        return json.loads(raw)
    except Exception:
        return []


def main() -> int:
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    for label, coll in COLLECTIONS:
        print("=" * 78)
        print(f"### {label}  [{coll}]")
        try:
            store = QdrantVectorStore(
                url=QDRANT, collection_name=coll, vector_size=1024, create_if_missing=False
            )
        except Exception as exc:
            print(f"  跳过：{type(exc).__name__}: {exc}")
            continue
        try:
            r = HybridRetriever(store, embeddings=emb, top_k=5)
            hits = r.search(QUERY, top_k=5, kb_id="cninfo_report")
        except Exception as exc:
            print(f"  检索失败：{type(exc).__name__}: {exc}")
            continue
        for i, h in enumerate(hits, 1):
            md = h.get("metadata") or {}
            txt = h.get("text") or ""
            fl = _facts(md)
            in_text = TARGET in txt
            in_facts = TARGET in (md.get("financial_facts_json") or "")
            # 指标名是否与数字同行（换行/管道分隔视为不同行）
            line_ok = False
            for line in txt.splitlines():
                if TARGET in line and METRIC_KEY in line.replace(" ", ""):
                    line_ok = True
            print(f"  #{i} page={md.get('page')} score={h.get('score', 0):.3f} "
                  f"| 数字在text={in_text} 数字在facts={in_facts} 指标数字同行={line_ok} facts数={len(fl)}")
            if i <= 3:
                body = txt[:420].replace("\n", " ⏎ ")
                print(f"      text: {body}")
                if fl:
                    for f in fl[:5]:
                        print(f"      fact: {f.get('metric')} = {f.get('raw_value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
