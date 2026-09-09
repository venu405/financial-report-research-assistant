#!/usr/bin/env python3
"""全库扫描：验证评测期望的证据文本是否真的存在于索引中。

用途：区分两类根因——
  * 索引里根本没有 → 分块/解析丢失，检索再怎么调都救不回来；
  * 索引里有但没召回 → 检索层问题（召回池太小 / 排序靠后）。

用法：
    .venv/Scripts/python.exe scripts/diag_index_presence.py \
        --collection kb_full_codex_20260830_24c7407e
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from typing import Any

QDRANT_URL = "http://127.0.0.1:16333"

TARGETS: dict[str, list[str]] = {
    "real_yutai_2025_h1_revenue": ["221,828,689.43", "221828689.43"],
    "real_zhongcheng_2024_net_profit": ["310,302,902.32", "310302902.32"],
    "real_kelida_2024_net_profit": ["8,583,076.40", "8583076.40"],
    "real_gree_2023_core_table_fields": ["203,979,266,387.09", "29,017,387,604.18"],
    "real_shenlian_2025_h1_rnd_ratio_synonym": ["22.32"],
    "real_huluwa_2025_h1_revenue_reason": ["呼吸系统用药销售额减少", "42.89"],
}


def scroll_all(collection: str, batch: int = 512) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    offset = None
    url = f"{QDRANT_URL}/collections/{collection}/points/scroll"
    while True:
        payload: dict[str, Any] = {
            "limit": batch,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            payload["offset"] = offset
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result = data.get("result") or {}
        chunk = result.get("points") or []
        points.extend(chunk)
        offset = result.get("next_page_offset")
        if not chunk or offset is None:
            break
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    args = ap.parse_args()

    print(f"扫描 {args.collection} …", flush=True)
    points = scroll_all(args.collection)
    print(f"共 {len(points)} points\n", flush=True)

    for case, needles in TARGETS.items():
        hits: list[tuple[str, dict[str, Any], str]] = []
        for point in points:
            payload = point.get("payload") or {}
            text = str(payload.get("text") or "")
            if not text:
                continue
            for needle in needles:
                if needle in text:
                    meta = payload.get("metadata") or payload
                    hits.append((needle, meta, text))
                    break
        print(f"===== {case}")
        if not hits:
            print("  未命中：索引中不存在任何目标文本")
            continue
        print(f"  命中 {len(hits)} 块")
        for needle, meta, text in hits[:3]:
            title = meta.get("doc_title") or meta.get("source_path") or "?"
            page = meta.get("page", meta.get("page_start", "?"))
            idx = text.find(needle)
            snippet = text[max(0, idx - 90) : idx + 60].replace("\n", " ⏎ ")
            print(f"    [{needle}] {title} p{page}")
            print(f"      …{snippet}…")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
