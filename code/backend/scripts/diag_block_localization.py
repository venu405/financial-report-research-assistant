#!/usr/bin/env python3
"""诊断：某题的「正确块」在向量召回里的排位，以及结构化路由能否过滤到它。

用途：当某题 hit@5=True（文档级命中）但 evidence_rate=0（证据文本级未命中）时，
说明公司找对了、装着数字的表格块却没进 top-K。本脚本区分这两种失败，并
按 retriever 的路由条件（is_table + section_path + report_period + doc_id）
复算一遍，判断路由是否本可捞到它。

用法：
    .venv/Scripts/python.exe scripts/diag_block_localization.py \
        --collection kb_full_codex_20260830_24c7407e \
        --question "ST富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？" \
        --needle "110,682,912.05" --top 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))

QDRANT_URL = "http://127.0.0.1:16333"
OLLAMA_URL = "http://127.0.0.1:11434"


def embed(text: str, model: str = "bge-m3") -> list[float]:
    payload = {"model": model, "input": text}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embeddings"][0]


def search(collection: str, vector: list[float], top: int) -> list[dict]:
    payload = {
        "vector": vector,
        "limit": top,
        "with_payload": True,
    }
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{collection}/points/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("result") or [])


def route_matches(payload: dict) -> bool:
    """复算 retriever 结构化财务路由的过滤条件。"""
    if not payload.get("is_table"):
        return False
    if not payload.get("report_period"):
        return False
    section = str(payload.get("section_path") or "")
    if not re.search(r"主要会计数据|主要财务指标", section):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--needle", required=True, help="期望证据里的关键数字/短语")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--source-keyword", default="", help="只看该公司的块")
    args = ap.parse_args()

    vector = embed(args.question)
    hits = search(args.collection, vector, args.top)

    print(f"问题：{args.question}")
    print(f"目标串：{args.needle!r}   向量召回 top-{len(hits)}")
    print("=" * 92)
    found_rank = None
    for index, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        text = str(payload.get("text") or "")
        path = str(payload.get("source_path") or "")
        if args.source_keyword and args.source_keyword not in path:
            continue
        has_needle = args.needle in text
        if has_needle and found_rank is None:
            found_rank = index
        print(
            f"#{index:>2} score={hit.get('score', 0):.4f} "
            f"page={payload.get('page')} table={str(payload.get('is_table')):5s} "
            f"route={str(route_matches(payload)):5s} "
            f"needle={'★YES★' if has_needle else '  .  '}"
        )
        print(f"      section={payload.get('section_path')}")
        print(f"      src={path.split('/')[-1][:46]}")

    print("=" * 92)
    if found_rank is None:
        print(f"结论：top-{len(hits)} 向量召回里【没有】含目标串的块 —— 块级定位失效。")
    else:
        print(f"结论：含目标串的块在向量召回 #{found_rank}。")

    # 路由过滤复算：目标块是否满足路由条件（与排位无关）
    print("\n路由条件复算（is_table + report_period + 主要会计数据/主要财务指标）：")
    for index, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        text = str(payload.get("text") or "")
        if args.needle not in text:
            continue
        print(f"  #{index} 命中目标串：route_matches={route_matches(payload)} "
              f"is_table={payload.get('is_table')} "
              f"period={payload.get('report_period')!r} "
              f"section={payload.get('section_path')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
