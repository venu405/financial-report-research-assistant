#!/usr/bin/env python3
"""最小复现：用真实证据块跑 _verify_numeric_claims，定位核验拒答的断链点。

用途：当某题「证据召到了(ev=1.0)却被核验拒答」时，不必重跑整个后端，
直接把真实块喂给核验函数，逐条打印 claim vs candidate 的拒绝原因。

用法：
    .venv/Scripts/python.exe scripts/diag_verify_minrepro.py \
        --collection kb_full_codex_20260830_24c7407e \
        --source-keyword "ST中程-2024年年度报告" --page 11 \
        --question "青岛中资中程2024年归属于上市公司股东的净利润是多少？" \
        --answer "ST中程2024年归属于上市公司股东的净利润为-310,302,902.32元。[1][2]"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))

QDRANT_URL = "http://127.0.0.1:16333"


def scroll_all(collection: str, batch: int = 512) -> list[dict]:
    """全量拉取；本机 7.3 万点约 10 秒，比猜 payload 索引类型更稳。"""
    points: list[dict] = []
    offset = None
    url = f"{QDRANT_URL}/collections/{collection}/points/scroll"
    while True:
        payload: dict = {"limit": batch, "with_payload": True, "with_vector": False}
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


def scroll_by_filter(collection: str, source_keyword: str, page: int | None) -> list[dict]:
    """按 source_path 子串（可选页码）筛选真实块。"""
    hits = []
    for point in scroll_all(collection):
        payload = point.get("payload") or {}
        path = str(payload.get("source_path") or "")
        if source_keyword not in path:
            continue
        if page is not None and payload.get("page") != page:
            continue
        hits.append(point)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--source-keyword", required=True)
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--show-chars", type=int, default=1200)
    args = ap.parse_args()

    from services.kb.qa_graph import (  # noqa: E402
        _extract_numeric_claims,
        _match_rejection_reason,
        _table_body_evidence,
        _verify_numeric_claims,
    )

    points = scroll_by_filter(args.collection, args.source_keyword, args.page)
    print(f"命中块数: {len(points)}")
    if not points:
        return 1

    texts: list[str] = []
    for p in points:
        payload = p.get("payload") or {}
        hit = {"text": payload.get("text", ""), "metadata": payload}
        texts.extend(_table_body_evidence(hit))
    print(f"证据文本条数: {len(texts)}")

    for i, t in enumerate(texts[:2]):
        print(f"\n--- 证据[{i}] 前 {args.show_chars} 字 ---")
        print(t[: args.show_chars])

    print("\n=== 答案侧 claims ===")
    for c in _extract_numeric_claims(args.answer):
        print(f"  raw={c.raw!r} number={c.number} unit={c.unit!r} "
              f"is_percent={c.is_percent} metric={c.metric!r} line_metric={c.line_metric!r}")

    print("\n=== 证据侧 claims（前 30） ===")
    ev_claims = []
    for t in texts:
        ev_claims.extend(_extract_numeric_claims(t))
    for c in ev_claims[:30]:
        print(f"  raw={c.raw!r} number={c.number} unit={c.unit!r} "
              f"is_percent={c.is_percent} metric={c.metric!r} line_metric={c.line_metric!r}")

    print("\n=== 核验结果 ===")
    # 返回 (是否全部有证据, 未被支持的声明)，第一个值是 bool
    all_supported, unsupported = _verify_numeric_claims(
        args.answer, texts, question=args.question
    )
    print(f"all_supported={all_supported} unsupported={len(unsupported)}")
    for claim in unsupported:
        print(f"\n  未通过 claim: raw={claim.raw!r} number={claim.number} "
              f"unit={claim.unit!r} is_percent={claim.is_percent} metric={claim.metric!r}")
        for cand in ev_claims[:40]:
            reason = _match_rejection_reason(claim, cand)
            if reason is None:
                print(f"    [PASS] vs {cand.raw!r}")
                break
            if claim.number == cand.number:
                print(f"    [同数字被拒] vs raw={cand.raw!r} unit={cand.unit!r} "
                      f"is_percent={cand.is_percent} metric={cand.metric!r} "
                      f"line_metric={cand.line_metric!r} -> {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
