#!/usr/bin/env python3
"""轻量 36 题全量验收：起后端(指向新 collection) → 跑 evaluate_quality.py → 关后端。

不碰 run_qdrant_full36.py 的 preflight / lineage 校验（那些绑定旧 run 状态），
只复用它的环境变量与命令构造，直接对新 collection 验收。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SCRIPT_PATH.parent))

import run_qdrant_full36 as full36  # noqa: E402

COLLECTION = "kb_full_v2_noisefix_20260831"
PORT = 18080
QDRANT_URL = "http://127.0.0.1:16333"
VECTOR_SIZE = 1024
TESTSET = "testsets/rag_real_quality_v2.yaml"
LABEL = f"qdrant-full-185-{COLLECTION}"


def main() -> int:
    out = BACKEND_DIR / "reports" / "qdrant-full36-v2-noisefix-20260831.json"
    env = full36.build_backend_environment(
        BACKEND_DIR,
        collection=COLLECTION,
        qdrant_url=QDRANT_URL,
        vector_size=VECTOR_SIZE,
    )
    env["KB_QDRANT_CREATE_IF_MISSING"] = "false"

    cmd = full36.build_backend_command(PORT)
    print("启动后端:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=BACKEND_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        full36.wait_for_ready(proc, f"http://127.0.0.1:{PORT}", 90)
        print("后端就绪，开始 36 题全量评测...", flush=True)
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_quality.py",
            "--rag",
            TESTSET,
            "--base-url",
            f"http://127.0.0.1:{PORT}",
            "--label",
            LABEL,
            "--output",
            str(out),
        ]
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
