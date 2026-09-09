#!/usr/bin/env python3
"""起后端（捕获日志到文件）→ 跑单题评测 → 关后端 → 抽取核验相关日志行。

用法：
    .venv/Scripts/python.exe scripts/diag_v3_backend_trace.py --case real_yitai_2025_h1_operating_cash
可选 --collection 指定 collection（默认分离实验档）。
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

PORT = 18082
QDRANT_URL = "http://127.0.0.1:16333"
VECTOR_SIZE = 1024
TESTSET = "testsets/rag_real_quality_v2.yaml"
PATTERNS = [
    r"核验", r"verif", r"answerable", r"可答性", r"gate",
    r"数字", r"numeric", r"遮蔽", r"fallback", r"低质量", r"low_quality",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--collection", default="kb_exp_v3_sep_20260901")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else BACKEND_DIR / f"logs-diag-{args.case}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = full36.build_backend_environment(
        BACKEND_DIR, collection=args.collection,
        qdrant_url=QDRANT_URL, vector_size=VECTOR_SIZE,
    )
    env["KB_QDRANT_CREATE_IF_MISSING"] = "false"
    env["KB_LOG_LEVEL"] = "DEBUG"
    if os.getenv("KB_LENIENT_PERIOD_MATCH"):
        env["KB_LENIENT_PERIOD_MATCH"] = os.environ["KB_LENIENT_PERIOD_MATCH"]
    print(f"KB_LENIENT_PERIOD_MATCH = {env.get('KB_LENIENT_PERIOD_MATCH', '0')}", flush=True)

    cmd = full36.build_backend_command(PORT)
    print(f"后端启动中（日志 -> {log_path}）", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            full36.wait_for_ready(proc, f"http://127.0.0.1:{PORT}", 120)
            print(f"后端就绪，跑单题 {args.case}", flush=True)
            out = BACKEND_DIR / f"reports/diag-{args.case}-{int(time.time())}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(
                [sys.executable, "scripts/evaluate_quality.py", "--rag", TESTSET,
                 "--base-url", f"http://127.0.0.1:{PORT}", "--label", f"diag-{args.case}",
                 "--case-id", args.case, "--output", str(out)],
                cwd=BACKEND_DIR, env=env, check=False,
            )
            print("评测退出码:", res.returncode, "| 报告:", out, flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    print()
    print("=" * 78)
    print("核验相关日志行：")
    rx = re.compile("|".join(PATTERNS), re.I)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    shown = 0
    for ln in lines:
        if rx.search(ln):
            print("  ", ln.strip()[:260])
            shown += 1
            if shown >= 60:
                print("  ...(截断)")
                break
    if shown == 0:
        print("  （无匹配，输出最后 40 行）")
        for ln in lines[-40:]:
            print("  ", ln.strip()[:260])
    return 0


if __name__ == "__main__":
    sys.exit(main())
