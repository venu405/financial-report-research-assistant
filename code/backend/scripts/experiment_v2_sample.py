"""实验档：新分块策略（T1-lite：跳过 find_tables 垃圾）小样本入库 + 检索验证。

- 不碰旧 collection、不改生产代码默认行为（KB_USE_FIND_TABLES 默认 "1"）。
- 写入全新的 Qdrant collection。
- 用 3 道已知失败题验证：正确证据子串是否出现在 top-5 检索结果里。
"""
import os

# 必须在 import ingest 之前设置，切到「新策略」。
os.environ["KB_USE_FIND_TABLES"] = "0"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.kb.embeddings import EmbeddingClient
from services.kb.ingest import build_chunks
from services.kb.qdrant_vector_store import QdrantVectorStore

QDRANT_URL = "http://127.0.0.1:16333"
COLLECTION = "kb_exp_v2_noisefix_20260831"
KB_ID = "default"

SAMPLES = [
    ("gree", "data_kb_test/cninfo_annual/格力电器-2023年年度报告.pdf", "格力电器-2023年年度报告"),
    ("yutai", "data_kb_test/cninfo/裕太微-2025年半年度报告.pdf", "裕太微-2025年半年度报告"),
    ("jinzhou", "data_kb_test/cninfo_annual/ST锦港-锦州港股份有限公司2024年年度报告.pdf", "ST锦港-锦州港股份有限公司2024年年度报告"),
]

QUERIES = {
    "gree": "格力电器2023年年报主要会计数据中，营业收入、归母净利润、经营活动现金流量净额和基本每股收益分别是多少？",
    "yutai": "裕太微电子2025年上半年营业收入具体是多少？",
    "jinzhou": "锦州港2024年年报披露的2025年经营计划中，计划营业收入是多少？",
}

EVIDENCE = {
    "gree": ["203,979,266,387.09", "29,017,387,604.18", "56,398,426,354.17", "5.22"],
    "yutai": ["221,828,689.43", "22,182.87"],
    "jinzhou": ["17.21亿"],
}


def _norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "")


def main() -> int:
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model="bge-m3")
    store = QdrantVectorStore(
        url=QDRANT_URL, collection_name=COLLECTION, vector_size=1024, create_if_missing=True
    )

    for key, path, title in SAMPLES:
        chunks = build_chunks(Path(path), doc_id=f"exp-{key}", document_title=title, kb_id=KB_ID)
        texts = [c.text for c in chunks]
        vecs = emb.embed_texts(texts)
        ids = store.add_chunks(
            embeddings=vecs,
            texts=texts,
            doc_id=f"exp-{key}",
            doc_title=title,
            source_type="pdf",
            chunk_indices=list(range(len(chunks))),
            kb_id=KB_ID,
            extra_metadata=[c.metadata for c in chunks],
        )
        print(f"[{key}] {len(chunks)} chunks 入库（ids={len(ids)}）", flush=True)

    print("\n===== 检索验证（top-5 是否命中证据） =====", flush=True)
    for key, q in QUERIES.items():
        qv = emb.embed_query(q)
        hits = store.search(qv, top_k=5, kb_id=KB_ID)
        ev = EVIDENCE[key]
        print(f"\n--- {key} | 期望证据 {ev} ---", flush=True)
        for i, h in enumerate(hits, 1):
            txt = h.get("text", "")
            hit = any(_norm(e) in _norm(txt) for e in ev)
            meta = h.get("metadata") or {}
            print(f"  #{i} [{'HIT' if hit else 'miss'}] page={meta.get('page')} | {txt[:90].replace(chr(10), ' ')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
