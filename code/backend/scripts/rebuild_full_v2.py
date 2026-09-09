#!/usr/bin/env python3
"""全量重建脚本（新策略 T1-lite：KB_USE_FIND_TABLES=0，去掉 find_tables 垃圾）。

与旧档完全隔离：
- 不碰旧 collection（kb_full_codex_20260830_24c7407e）。
- 写入全新 collection，kb_id 对齐测试集（cninfo_report）。
- 每份 PDF：build_chunks → 补 source_path → embed → add_chunks。
- checkpoint 续跑：记录已完成文件（source_path），中断后跳过。

运行（必须在 code/backend 目录）：
    ./.venv/Scripts/python.exe scripts/rebuild_full_v2.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import ingest 之前设置，切到「新策略」（去掉 find_tables 垃圾）。
os.environ["KB_USE_FIND_TABLES"] = "0"

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(BACKEND_DIR / "src"))

from services.kb.embeddings import EmbeddingClient  # noqa: E402
from services.kb.ingest import build_chunks  # noqa: E402
from services.kb.qdrant_vector_store import QdrantVectorStore  # noqa: E402

QDRANT_URL = "http://127.0.0.1:16333"
COLLECTION = "kb_full_v2_noisefix_20260831"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024

# 目录前缀 → kb_id 映射（与旧档 migrate 的 --kb-map 一致）：
#   cninfo(半年报) + cninfo_annual(年报) → cninfo_report
#   flk(法律 docx) → flk_law
SOURCE_ROOTS = [
    (BACKEND_DIR / "data_kb_test" / "cninfo", "cninfo_report"),
    (BACKEND_DIR / "data_kb_test" / "cninfo_annual", "cninfo_report"),
    (BACKEND_DIR / "data_kb_test" / "flk", "flk_law"),
]
CHECKPOINT_PATH = BACKEND_DIR / "migration-record" / "v2-rebuild-checkpoint.json"


def collect_docs() -> list[tuple[Path, str]]:
    docs: list[tuple[Path, str]] = []
    for root, kb_id in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for ext in ("*.pdf", "*.docx"):
            docs.extend((p, kb_id) for p in sorted(root.rglob(ext)) if p.is_file())
    return docs


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
    parser = argparse.ArgumentParser(description="全量重建（T1-lite 新策略）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 份（冒烟验证用）")
    args = parser.parse_args()

    store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION,
        vector_size=VECTOR_SIZE,
        create_if_missing=True,
    )
    emb = EmbeddingClient(base_url="http://127.0.0.1:11434", model=EMBED_MODEL)

    pdfs = collect_docs()
    done = load_checkpoint()
    pending = [(p, kb) for p, kb in pdfs if p.relative_to(PROJECT_ROOT).as_posix() not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    total = len(pending)
    print(f"全库 {len(pdfs)} 份，已完成 {len(done)} 份，本轮处理 {total} 份", flush=True)

    if not pending:
        print("无待处理文件，退出。", flush=True)
        return 0

    failed = 0
    started = time.time()
    for i, (pdf, kb_id) in enumerate(pending, 1):
        rel = pdf.relative_to(PROJECT_ROOT).as_posix()  # code/backend/data_kb_test/...
        doc_id = stable_doc_id(rel)
        t0 = time.time()
        try:
            chunks = build_chunks(pdf, doc_id=doc_id, kb_id=kb_id)
            # 补 source_path（评测 _source_matches 依赖 data_kb_test/ 前缀对齐）
            for c in chunks:
                c.metadata["source_path"] = rel
            if not chunks:
                print(f"[{i}/{total}] EMPTY {rel}", flush=True)
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
            done.add(rel)
            save_checkpoint(done)
            dt = time.time() - t0
            print(f"[{i}/{total}] OK {rel} ({len(chunks)} chunks, {dt:.1f}s)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] FAIL {rel}: {type(exc).__name__}: {exc}", flush=True)

    elapsed = time.time() - started
    print(f"=== 结束：成功 {total - failed}，失败 {failed}，耗时 {elapsed/60:.1f} 分钟 ===", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
