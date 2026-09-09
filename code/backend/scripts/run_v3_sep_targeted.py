#!/usr/bin/env python3
"""分离方案定向评测：起后端（指向实验 collection）→ 跑指定 case-id → 关后端。

用法：
    .venv/Scripts/python.exe scripts/run_v3_sep_targeted.py \
        --output reports/v3-sep-targeted5-20260901.json \
        --case real_yitai_2025_h1_operating_cash \
        --case real_shenlian_2025_h1_revenue \
        --case real_pawa_full_summary_consistency

不指定 --case 时跑全部 36 题。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SCRIPT_PATH.parent))

import run_qdrant_full36 as full36  # noqa: E402

COLLECTION = "kb_exp_v3_sep_20260901"
PORT = 18081
QDRANT_URL = "http://127.0.0.1:16333"
VECTOR_SIZE = 1024
TESTSET = "testsets/rag_real_quality_v2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="分离方案定向评测")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[], dest="cases",
                        help="指定 case-id，可重复；不指定则跑全部")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"目标 Qdrant collection（默认 {COLLECTION}）",
    )
    parser.add_argument(
        "--lenient-period",
        action="store_true",
        help="打开 KB_LENIENT_PERIOD_MATCH=1（页眉年份污染导致的期间失配放宽）",
    )
    args = parser.parse_args()

    n = len(args.cases) if args.cases else 36
    label = args.label or f"v3-sep-targeted{n}-{COLLECTION}"

    out = args.output if args.output.is_absolute() else BACKEND_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    env = full36.build_backend_environment(
        BACKEND_DIR,
        collection=args.collection,
        qdrant_url=QDRANT_URL,
        vector_size=VECTOR_SIZE,
    )
    env["KB_QDRANT_CREATE_IF_MISSING"] = "false"
    if args.lenient_period:
        env["KB_LENIENT_PERIOD_MATCH"] = "1"
    print(f"KB_LENIENT_PERIOD_MATCH = {env.get('KB_LENIENT_PERIOD_MATCH', '0')}", flush=True)

    cmd = full36.build_backend_command(PORT)
    print(f"启动后端 (port={PORT}, collection={args.collection})", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=BACKEND_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        full36.wait_for_ready(proc, f"http://127.0.0.1:{PORT}", 90)
        print(f"后端就绪，评测 {n} 题: {args.cases or 'ALL'}", flush=True)
        eval_cmd = [
            sys.executable, "scripts/evaluate_quality.py",
            "--rag", TESTSET,
            "--base-url", f"http://127.0.0.1:{PORT}",
            "--label", label,
            "--output", str(out),
        ]
        for cid in args.cases:
            eval_cmd.extend(["--case-id", cid])
        result = subprocess.run(eval_cmd, cwd=BACKEND_DIR, env=env, check=False)
        print("评测退出码:", result.returncode, flush=True)
        print("报告:", out, flush=True)
        return int(result.returncode)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
