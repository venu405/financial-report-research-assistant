#!/usr/bin/env python3
"""诊断：评测期望的证据数字在索引里是否“完整连续”。

背景（重要）：核验层要求答案里的数字能在证据文本中原样找到。如果 PDF 表格解析把
数字切碎（例如 "507,670,9" + 换行 + "12.42"），那么无论检索多准，核验都必然拒答。
本脚本直接到向量库里逐块检查期望数字的完整性，把「解析失真」与「检索/核验问题」分开。

判定分三档：
  INTACT     —— 期望短语（指标名+数字）去空白后完整连续出现
  NUMBER_ONLY—— 只有数字本身连续出现，指标名与数字之间隔了别的列（表格中间列）
  BROKEN     —— 连数字本身都被截断（PDF 抽取把数字切碎）

用法：
    .venv/Scripts/python.exe scripts/diag_evidence_integrity.py \
        --collection kb_full_codex_20260830_24c7407e
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

QDRANT_URL = "http://127.0.0.1:16333"
BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTSET = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d+")


def canonical(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def scroll_all(collection: str, batch: int = 512) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    offset = None
    url = f"{QDRANT_URL}/collections/{collection}/points/scroll"
    while True:
        payload: dict[str, Any] = {"limit": batch, "with_payload": True, "with_vector": False}
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
        points.extend(result.get("points") or [])
        offset = result.get("next_page_offset")
        if not offset:
            break
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--context", type=int, default=70)
    args = ap.parse_args()

    import yaml

    tests = (yaml.safe_load(TESTSET.read_text(encoding="utf-8")) or {}).get("tests") or []
    if args.case:
        tests = [t for t in tests if t.get("id") in set(args.case)]

    print(f"扫描 {args.collection} ...", flush=True)
    points = scroll_all(args.collection)
    print(f"共 {len(points)} points\n", flush=True)

    # 预计算每块的规范化文本，并保留原文用于展示上下文
    corpus: list[tuple[dict, str, str]] = []
    for p in points:
        payload = p.get("payload") or {}
        raw = str(payload.get("text") or "")
        corpus.append((payload, canonical(raw), raw))

    stats = {"INTACT": 0, "NUMBER_ONLY": 0, "BROKEN": 0, "MISSING": 0}
    for t in tests:
        evidences = t.get("expected_evidence") or []
        if not evidences:
            continue
        print("=" * 78)
        print(f"## {t.get('id')}")
        print(f"   Q: {t.get('question')}")
        for ev in evidences:
            can_ev = canonical(ev)
            nums = NUM_RE.findall(ev)
            hits = [c for c in corpus if can_ev in c[1]]
            if hits:
                stats["INTACT"] += 1
                payload = hits[0][0]
                print(f"   [INTACT]      {ev!r}  ({len(hits)} 块, page={payload.get('page')})")
                continue
            # 短语不成立，退而检查数字本身
            if not nums:
                stats["MISSING"] += 1
                print(f"   [MISSING]     {ev!r}  (无数字，且短语未命中)")
                continue
            num = nums[0]
            can_num = canonical(num)
            bare = can_num.lstrip("-").replace(",", "")
            num_hits = [c for c in corpus if can_num in c[1]]
            if num_hits:
                stats["NUMBER_ONLY"] += 1
                payload = num_hits[0][0]
                idx = num_hits[0][1].find(can_num)
                print(f"   [NUMBER_ONLY] {ev!r}  数字 {num} 连续存在({len(num_hits)}块, page={payload.get('page')})，但指标名与数字之间有间隔")
                print(f"                 上下文: ...{num_hits[0][2][max(0, idx - args.context): idx + len(can_num) + 20]}...")
                continue
            # 数字本身也不连续 → 找被切碎的近邻
            probe = bare[:6]
            broken = [c for c in corpus if probe in c[1]]
            if broken:
                stats["BROKEN"] += 1
                payload, can_text, raw = broken[0]
                idx = can_text.find(probe)
                print(f"   [BROKEN]      {ev!r}  数字 {num} 不连续！疑似切碎({len(broken)}块含前6位, page={payload.get('page')})")
                print(f"                 上下文: ...{raw[max(0, idx - args.context): idx + 60]}...")
            else:
                stats["MISSING"] += 1
                print(f"   [MISSING]     {ev!r}  数字 {num} 全库未出现")

    print("\n" + "=" * 78)
    print("汇总（按期望证据条数计）：", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
