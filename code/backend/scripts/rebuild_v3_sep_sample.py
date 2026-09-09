#!/usr/bin/env python3
"""分离方案实验档：只入退步 5 题涉及的文档，写全新 Qdrant collection。

与所有旧档完全隔离：
- 不碰 kb_full_codex_20260830_24c7407e（旧档 73,662）
- 不碰 kb_full_v2_noisefix_20260831（去 find_tables 档 35,388）
- 不碰 kb_exp_v2_noisefix_20260831

开关：KB_SEPARATE_TABLE_MD=1（启用分离通道），KB_USE_FIND_TABLES 保持默认 1
（分离方案需要 find_tables 产出 table_md 供事实提取使用）。

doc_id / source_path 契约与 rebuild_full_v2.py 完全一致：
  rel = 相对 PROJECT_ROOT 的 posix 路径（code/backend/data_kb_test/...）
  doc_id = "doc-" + sha1(rel)[:32]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import ingest 之前设置，启用「分离通道」。
os.environ["KB_SEPARATE_TABLE_MD"] = "1"
os.environ["KB_USE_FIND_TABLES"] = "1"

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(BACKEND_DIR / "src"))

from services.kb.embeddings import EmbeddingClient  # noqa: E402
from services.kb.ingest import build_chunks  # noqa: E402
from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402

QDRANT_URL = "http://127.0.0.1:16333"
COLLECTION = "kb_exp_v3_sep_20260901"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024

# 退步 5 题涉及的 6 份文档（相对 BACKEND_DIR）
SAMPLES: list[tuple[str, str]] = [
    ("data_kb_test/cninfo/伊泰Ｂ股-内蒙古伊泰煤炭股份有限公司2025年半年度报告.pdf", "cninfo_report"),
    ("data_kb_test/cninfo/申联生物-申联生物医药（上海）股份有限公司2025年半年度报告.pdf", "cninfo_report"),
    ("data_kb_test/cninfo_annual/ST九有-湖北九有投资股份有限公司2024年年度报告.pdf", "cninfo_report"),
    ("data_kb_test/cninfo_annual/ST富润-2024年年度报告.pdf", "cninfo_report"),
    ("data_kb_test/cninfo_annual/帕瓦股份-浙江帕瓦新能源股份有限公司2024年年度报告.pdf", "cninfo_report"),
    ("data_kb_test/cninfo_annual/帕瓦股份-浙江帕瓦新能源股份有限公司2024年年度报告摘要.pdf", "cninfo_report"),
]

CHECKPOINT_PATH = BACKEND_DIR / "migration-record" / "v3-sep-sample-checkpoint.json"


def load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        try:
            return set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_checkpoint(done: set[str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(done), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def stable_doc_id(rel_path: str) -> str:
    return "doc-" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:32]


def main() -> int:
    print(f"开关检查: KB_SEPARATE_TABLE_MD={os.environ.get('KB_SEPARATE_TABLE_MD')} "
          f"KB_USE_FIND_TABLES={os.environ.get('KB_USE_FIND_TABLES')}", flush=True)
    print(f"目标 collection: {COLLECTION}", flush=True)

    store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION,
        vector_size=VECTOR_SIZE,
        create_if_missing=True,
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model=EMBED_MODEL)

    done = load_checkpoint()
    pending = [(rel, kb) for rel, kb in SAMPLES if rel not in done]
    print(f"样本 {len(SAMPLES)} 份，已完成 {len(done)} 份，本轮处理 {len(pending)} 份", flush=True)

    failed = 0
    started = time.time()
    for i, (rel, kb_id) in enumerate(pending, 1):
        pdf = BACKEND_DIR / rel
        if not pdf.is_file():
            failed += 1
            print(f"[{i}/{len(pending)}] MISSING {rel}", flush=True)
            continue
        doc_id = stable_doc_id(rel)
        t0 = time.time()
        try:
            chunks = build_chunks(pdf, doc_id=doc_id, kb_id=kb_id)
            for c in chunks:
                c.metadata["source_path"] = rel
            if not chunks:
                print(f"[{i}/{len(pending)}] EMPTY {rel}", flush=True)
                done.add(rel)
                save_checkpoint(done)
                continue
            texts = [c.text for c in chunks]
            vecs = emb.embed_texts(texts)
            store.add_chunks(
                embeddings=vecs,
                texts=texts,
                doc_id=doc_id,
                doc_title=pdf.stem,
                source_type=pdf.suffix.lstrip(".").lower(),
                chunk_indices=list(range(len(chunks))),
                kb_id=kb_id,
                extra_metadata=[c.metadata for c in chunks],
            )
            noisy = sum(1 for t in texts if "|  |" in t or "|---" in t)
            with_facts = sum(1 for c in chunks if (c.metadata.get("financial_facts_json") or "[]") != "[]")
            done.add(rel)
            save_checkpoint(done)
            print(
                f"[{i}/{len(pending)}] OK {Path(rel).name} "
                f"({len(chunks)} chunks, 噪声块={noisy}, 带facts块={with_facts}, {time.time()-t0:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(pending)}] FAIL {rel}: {type(exc).__name__}: {exc}", flush=True)

    print(f"=== 结束：成功 {len(pending)-failed}，失败 {failed}，耗时 {(time.time()-started)/60:.1f} 分钟 ===", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
