#!/usr/bin/env python3
"""单题全链路追踪：跑 qa_graph，打印 verification_reasons / contexts / answer。

用于定位「检索命中但核验拒答」的具体原因。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env", override=False)

# 指向分离实验档
os.environ["KB_VECTOR_BACKEND"] = "qdrant"
os.environ["KB_QDRANT_URL"] = "http://127.0.0.1:16333"
os.environ["KB_QDRANT_COLLECTION"] = os.environ.get("DIAG_COLLECTION", "kb_exp_v3_sep_20260901")
os.environ["KB_QDRANT_VECTOR_SIZE"] = "1024"
os.environ["KB_QDRANT_CREATE_IF_MISSING"] = "false"

QUESTION = "伊泰煤炭2025年上半年经营活动产生的现金流量净额是多少？"
TARGET = "3,898,863,782.71"


def main() -> int:
    from services.kb.qa_graph import build_qa_graph, run_qa  # noqa: E402

    graph = build_qa_graph()
    print(f"collection = {os.environ['KB_QDRANT_COLLECTION']}", flush=True)
    print(f"question   = {QUESTION}", flush=True)
    result = run_qa(graph, question=QUESTION, kb_id="cninfo_report")

    print()
    print("=" * 78)
    print("ANSWER:", result.get("answer"))
    print("score:", result.get("score"), "| degraded:", result.get("answer_degraded"))

    ctxs = result.get("contexts") or []
    print(f"\n--- contexts ({len(ctxs)}) ---")
    for i, c in enumerate(ctxs, 1):
        md = c.get("metadata") or {}
        txt = c.get("text") or ""
        facts = md.get("financial_facts_json") or "[]"
        try:
            nf = len(json.loads(facts))
        except Exception:
            nf = 0
        print(f"  #{i} page={md.get('page')} score={c.get('score', 0):.3f} facts={nf} "
              f"含目标={TARGET in txt or TARGET in facts}")
        print(f"      {txt[:260].replace(chr(10), ' ⏎ ')}")
        if nf:
            for f in json.loads(facts)[:6]:
                print(f"      fact: {f.get('metric')} = {f.get('raw_value')}")

    # verification_reasons 不在 run_qa 返回值里，改从 graph state 取
    print()
    print("--- 从图状态取 verification_reasons ---")
    cfg = {"configurable": {"thread_id": "diag-trace"}}
    state = graph.get_state(cfg) if hasattr(graph, "get_state") else None
    if state is not None:
        vals = getattr(state, "values", {}) or {}
        reasons = vals.get("verification_reasons") or []
        print("  verification_reasons:", json.dumps(reasons, ensure_ascii=False, indent=2)[:2000])
        print("  gate_action:", vals.get("gate_action"))
        print("  retries:", vals.get("retries"))
    else:
        print("  (graph 无 get_state，跳过)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
