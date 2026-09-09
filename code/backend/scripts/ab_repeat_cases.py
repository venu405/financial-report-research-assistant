#!/usr/bin/env python3
"""重复跑若干题目 R 次，输出 pass/fail 矩阵，用于区分「真实退步」与「随机波动」。

端到端评测里 LLM 生成与核验都存在天然波动，单跑一次得出的 +1/-1 不能直接归因到代码改动。
本脚本只起一次后端，把每道题重复跑 R 遍，输出每题的通过率，让噪声显形。

用法：
    # 当前配置下把两道退步题各跑 3 遍
    .venv/Scripts/python.exe scripts/ab_repeat_cases.py --repeat 3 \
        --case real_huluwa_2025_h1_revenue --case real_jiuyou_2024_net_profit

    # 带开关对比（环境变量直接透传给后端）
    KB_BM25_RRF_WEIGHT=0 .venv/Scripts/python.exe scripts/ab_repeat_cases.py \
        --repeat 3 --case real_huluwa_2025_h1_revenue \
        --output reports/ab-bm25w0.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SCRIPT_PATH))

import run_qdrant_full36 as full36  # noqa: E402

PORT = 18084
QDRANT_URL = "http://127.0.0.1:16333"
VECTOR_SIZE = 1024
TESTSET = "testsets/rag_real_quality_v2.yaml"
# 这些开关决定检索/核验行为，写进报告便于事后对照
TRACKED_ENVS = (
    "KB_BM25_RRF_WEIGHT",
    "KB_METRIC_LOOSE_BINDING",
    "KB_FILTER_LOW_INFO_CHUNKS",
    "KB_LENIENT_PERIOD_MATCH",
    "KB_ANSWERABILITY_SCORE",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = (
        Path(args.output)
        if args.output
        else BACKEND_DIR / f"reports/ab-repeat-{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = full36.build_backend_environment(
        BACKEND_DIR,
        collection=args.collection,
        qdrant_url=QDRANT_URL,
        vector_size=VECTOR_SIZE,
    )
    env["KB_QDRANT_CREATE_IF_MISSING"] = "false"

    config = {k: os.getenv(k, "") for k in TRACKED_ENVS}
    print("配置:", json.dumps(config, ensure_ascii=False), flush=True)
    print(f"每 {args.repeat} 遍 × {len(args.case)} 题\n", flush=True)

    runs: dict[str, list[dict]] = {c: [] for c in args.case}
    cmd = full36.build_backend_command(args.port)
    log_path = BACKEND_DIR / f"logs-ab-repeat-{int(time.time())}.txt"
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            full36.wait_for_ready(proc, f"http://127.0.0.1:{args.port}", 120)
            total = args.repeat * len(args.case)
            done = 0
            for rep in range(1, args.repeat + 1):
                for case in args.case:
                    done += 1
                    tmp = BACKEND_DIR / f"reports/_ab-{case}-{int(time.time())}.json"
                    print(f"[{done}/{total}] rep{rep} {case}", flush=True)
                    subprocess.run(
                        [sys.executable, "scripts/evaluate_quality.py", "--rag", TESTSET,
                         "--base-url", f"http://127.0.0.1:{args.port}",
                         "--label", f"ab-r{rep}", "--case-id", case, "--output", str(tmp)],
                        cwd=BACKEND_DIR, env=env, check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    rec: dict = {"rep": rep, "passed": None}
                    if tmp.exists():
                        try:
                            data = json.loads(tmp.read_text(encoding="utf-8"))
                            res = data.get("reports", [{}])[0].get("results", [])
                            if res:
                                r = res[0]
                                rec.update(
                                    passed=bool(r.get("passed")),
                                    fail_checks=[c for c, v in (r.get("checks") or {}).items() if not v],
                                    escalate=bool(r.get("escalate")),
                                    citations=r.get("citations"),
                                    faithfulness=r.get("faithfulness"),
                                    answer=(r.get("answer") or "")[:200],
                                )
                        except Exception as exc:  # noqa: BLE001
                            rec["error"] = str(exc)
                        # 不删除临时报告：批量删除会触发环境的 bulk-delete 保护，
                        # 反而打断整轮评测；留档也便于事后逐题复查。
                    runs[case].append(rec)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    print("\n" + "=" * 78)
    summary: dict[str, dict] = {}
    for case in args.case:
        items = runs[case]
        ok = sum(1 for i in items if i.get("passed"))
        summary[case] = {"pass": ok, "of": len(items), "runs": items}
        mark = "稳定通过" if ok == len(items) else ("稳定失败" if ok == 0 else "★波动★")
        print(f"{case:<52} {ok}/{len(items)}  {mark}")
        for i in items:
            print(f"    rep{i['rep']}: passed={i.get('passed')} fail={i.get('fail_checks')}")
            if i.get("answer"):
                print(f"           ans={i['answer'][:120]}")

    out_path.write_text(
        json.dumps({"config": config, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
