#!/usr/bin/env python3
"""一次起后端 → 依次跑多道指定题 → 关后端 → 抽取核验相关日志行。

与 diag_v3_backend_trace.py 的区别：后者每题都要重启后端，浪费大量时间。
本脚本只起一次后端，每题评测前往日志里写分隔标记，便于事后按题切分。

用法：
    .venv/Scripts/python.exe scripts/diag_verify_batch.py \
        --collection kb_full_codex_20260830_24c7407e \
        --case real_yutai_2025_h1_revenue --case real_kelida_2024_net_profit
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SCRIPT_PATH.parent))

import run_qdrant_full36 as full36  # noqa: E402

PORT = 18083
QDRANT_URL = "http://127.0.0.1:16333"
VECTOR_SIZE = 1024
TESTSET = "testsets/rag_real_quality_v2.yaml"
PATTERNS = [
    r"\[verify-miss\]", r"核验", r"verif", r"answerable", r"可答性", r"gate",
    r"数字", r"numeric", r"遮蔽", r"低质量", r"low_quality",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--collection", default="kb_full_codex_20260830_24c7407e")
    ap.add_argument("--log", default=None)
    ap.add_argument("--per-case-lines", type=int, default=40,
                    help="每题最多输出的匹配日志行数（默认 40）")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else BACKEND_DIR / f"logs-diag-batch-{int(time.time())}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = full36.build_backend_environment(
        BACKEND_DIR, collection=args.collection,
        qdrant_url=QDRANT_URL, vector_size=VECTOR_SIZE,
    )
    env["KB_QDRANT_CREATE_IF_MISSING"] = "false"
    env["KB_LOG_LEVEL"] = os.getenv("KB_LOG_LEVEL", "INFO")
    env["KB_DEBUG_VERIFY"] = os.getenv("KB_DEBUG_VERIFY", "1")
    if os.getenv("KB_LENIENT_PERIOD_MATCH"):
        env["KB_LENIENT_PERIOD_MATCH"] = os.environ["KB_LENIENT_PERIOD_MATCH"]
    print(f"collection = {args.collection}", flush=True)
    print(f"KB_DEBUG_VERIFY = {env['KB_DEBUG_VERIFY']}", flush=True)
    print(f"KB_LENIENT_PERIOD_MATCH = {env.get('KB_LENIENT_PERIOD_MATCH', '0')}", flush=True)

    cmd = full36.build_backend_command(PORT)
    print(f"后端启动中（日志 -> {log_path}）", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            full36.wait_for_ready(proc, f"http://127.0.0.1:{PORT}", 120)
            for i, case in enumerate(args.case, 1):
                lf.write(f"\n===== CASE-BEGIN {i}/{len(args.case)} {case} =====\n")
                lf.flush()
                print(f"[{i}/{len(args.case)}] 跑 {case}", flush=True)
                out = BACKEND_DIR / f"reports/diag-{case}-{int(time.time())}.json"
                res = subprocess.run(
                    [sys.executable, "scripts/evaluate_quality.py", "--rag", TESTSET,
                     "--base-url", f"http://127.0.0.1:{PORT}", "--label", f"diag-{case}",
                     "--case-id", case, "--output", str(out)],
                    cwd=BACKEND_DIR, env=env, check=False,
                )
                lf.write(f"===== CASE-END {case} rc={res.returncode} =====\n\n")
                lf.flush()
                print(f"    退出码 {res.returncode} | 报告 {out}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    print()
    print("=" * 78)
    print("核验相关日志行（按题分组）：")
    rx = re.compile("|".join(PATTERNS), re.I)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    current = None
    shown = 0
    for ln in lines:
        m = re.match(r"===== CASE-BEGIN \d+/\d+ (\S+) =====", ln.strip())
        if m:
            current = m.group(1)
            print(f"\n########## {current} ##########")
            shown = 0
            continue
        if re.match(r"===== CASE-END", ln.strip()):
            current = None
            continue
        if rx.search(ln):
            print("  ", ln.strip()[:300])
            shown += 1
            if shown >= args.per_case_lines:
                print("  ...(截断)")
                shown = 10 ** 9  # 本题后续不再输出
    return 0


if __name__ == "__main__":
    sys.exit(main())
