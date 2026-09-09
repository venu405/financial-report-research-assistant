#!/usr/bin/env python3
"""用**真实入库块**（不是手写重建表格）跑核验，判定拒答发生在核验层还是上游。

与 diag_verify_minrepro.py 的区别：本脚本直连隔离 Chroma 副本取真实文本，
避免"手写表格能通过、真实块却拒答"造成的错误结论（furun 的真实行标签被 PDF
抽取截断，手写重建表格无法复现）。

用法：
    .venv/Scripts/python.exe scripts/diag_verify_real_block.py \
        --chroma-dir .rag_eval/chroma_repro_26_current_20260831/chroma_data \
        --collection enterprise_kb \
        --source-keyword "ST富润-2024年年度报告" --page 5 --block-index 3 \
        --question "ST富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？" \
        --answer "扣除相关收入后的营业收入为110,682,912.05元。[1]" \
        --wrong-answer "扣除相关收入后的营业收入为72,897,668.51元。[1]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))


def load_real_blocks(chroma_dir: str, collection: str, keyword: str, page: int | None):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(BACKEND_DIR / chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_collection(collection)
    got = coll.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    hits = []
    for i, meta in enumerate(metas):
        meta = meta or {}
        path = str(
            meta.get("source_path") or meta.get("source") or meta.get("doc_title") or ""
        )
        if keyword not in path:
            continue
        if page is not None and meta.get("page") != page:
            continue
        hits.append({"text": docs[i] if i < len(docs) else "", "metadata": meta})
    hits.sort(
        key=lambda h: h["metadata"].get("page")
        if isinstance(h["metadata"].get("page"), int)
        else 0
    )
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", required=True)
    ap.add_argument("--collection", default="enterprise_kb")
    ap.add_argument("--source-keyword", required=True)
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--block-index", type=int, default=-1,
                    help="指定块序号；-1 表示把所有命中块拼成证据池")
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--wrong-answer", action="append", default=[])
    ap.add_argument("--show-chars", type=int, default=1500)
    args = ap.parse_args()

    from services.kb.qa_graph import (  # noqa: E402
        _extract_numeric_claims,
        _match_rejection_reason,
        _table_body_evidence,
        _verify_numeric_claims,
    )

    blocks = load_real_blocks(
        args.chroma_dir, args.collection, args.source_keyword, args.page
    )
    print(f"命中块数={len(blocks)}")
    if not blocks:
        return 1

    if args.block_index >= 0:
        chosen = [blocks[args.block_index]]
    else:
        chosen = blocks

    texts: list[str] = []
    for b in chosen:
        texts.extend(_table_body_evidence(b))

    print(f"证据文本条数={len(texts)}")
    for i, t in enumerate(texts):
        print(f"\n--- 证据[{i}] 前 {args.show_chars} 字 ---")
        print(t[: args.show_chars])

    print("\n=== 答案侧 claims ===")
    for c in _extract_numeric_claims(args.answer):
        print(f"  raw={c.raw!r} number={c.number} unit={c.unit!r} "
              f"is_percent={c.is_percent} metric={c.metric!r} line_metric={c.line_metric!r}")

    print("\n=== 证据侧 claims ===")
    ev_claims = []
    for t in texts:
        ev_claims.extend(_extract_numeric_claims(t))
    for c in ev_claims:
        print(f"  raw={c.raw!r} number={c.number} unit={c.unit!r} "
              f"is_percent={c.is_percent} metric={c.metric!r} line_metric={c.line_metric!r}")

    def report(tag: str, answer: str, expected: bool | None) -> bool:
        ok, unsupported = _verify_numeric_claims(answer, texts, question=args.question)
        verdict = "OK  " if (expected is None or ok == expected) else "FAIL"
        print(f"\n[{verdict}] {tag}: all_supported={ok} "
              f"unsupported={len(unsupported)} 期望={expected}")
        for claim in unsupported:
            print(f"   未通过 claim: raw={claim.raw!r} number={claim.number} "
                  f"unit={claim.unit!r} metric={claim.metric!r} "
                  f"line_metric={claim.line_metric!r}")
            for cand in ev_claims:
                reason = _match_rejection_reason(claim, cand)
                if reason is None:
                    print(f"     [PASS] vs {cand.raw!r}")
                    break
                if claim.number == cand.number:
                    print(f"     [同数字被拒] vs raw={cand.raw!r} unit={cand.unit!r} "
                          f"metric={cand.metric!r} line_metric={cand.line_metric!r} -> {reason}")
        return ok

    print("\n=== 核验结果 ===")
    good = report("正解", args.answer, True)
    all_ok = good
    for w in args.wrong_answer:
        r = report("负例(应拒绝)", w, False)
        all_ok = all_ok and not r
    print("\n结论:", "核验层对该真实块判定正确" if all_ok else "核验层存在缺口")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
