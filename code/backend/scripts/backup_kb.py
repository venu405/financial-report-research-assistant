#!/usr/bin/env python3
"""企业知识库数据备份：SQLite 一致性快照 + Chroma 目录复制 + manifest。

备份范围（与 chroma_data 同目录）：
  - kb_users.db      用户与权限（RBAC）
  - kb_checkpoints.db 对话状态（LangGraph checkpointer）
  - kb_audit.db      审计日志
  - chroma_data/     向量库（Chroma）

用法（在 code/backend 目录下）：
  ./.venv/Scripts/python.exe scripts/backup_kb.py                # 默认 data-dir=src，输出 backups/kb_<时间戳>
  ./.venv/Scripts/python.exe scripts/backup_kb.py --data-dir D:/prod/kb_data --backup-dir D:/backups/kb

设计要点：
  - SQLite 用 VACUUM INTO 做一致性快照（WAL 模式下安全，备份时业务写不中断）。
  - Chroma 目录整体 copytree（含 chroma.sqlite3 与段落文件）。
  - manifest.json 记录时间、Chroma 版本、文件清单，供恢复/升级核对。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# 与 chroma_data 同目录、需要备份的 SQLite 文件
SQLITE_FILES = ["kb_users.db", "kb_checkpoints.db", "kb_audit.db"]
CHROMA_DIR = "chroma_data"


def _vacuum_snapshot(src: Path, dst: Path) -> None:
    """SQLite 一致性快照（VACUUM INTO，不阻塞业务写）。"""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("VACUUM INTO ?", (str(dst),))
    finally:
        conn.close()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="备份企业知识库数据")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="数据目录（默认 backend/src，含 chroma_data + kb_*.db）",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="备份输出目录（默认 backend/backups/kb_<时间戳>）",
    )
    args = parser.parse_args(argv)

    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else backend_dir / "src"
    backup_dir = (
        Path(args.backup_dir)
        if args.backup_dir
        else backend_dir / "backups" / f"kb_{_ts()}"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "files": [],
    }

    # 1. SQLite 一致性快照
    for name in SQLITE_FILES:
        src = data_dir / name
        if not src.exists():
            continue
        dst = backup_dir / name
        _vacuum_snapshot(src, dst)
        manifest["files"].append(
            {"name": name, "size": dst.stat().st_size, "method": "vacuum_into"}
        )

    # 2. Chroma 目录整体复制
    chroma_src = data_dir / CHROMA_DIR
    if chroma_src.exists():
        chroma_dst = backup_dir / CHROMA_DIR
        shutil.copytree(chroma_src, chroma_dst)
        manifest["files"].append({"name": CHROMA_DIR, "method": "copytree"})

    # 3. Chroma 版本（升级时核对兼容性）
    try:
        import chromadb  # noqa: PLC0415

        manifest["chroma_version"] = chromadb.__version__
    except Exception:  # pragma: no cover - chromadb 未装时降级
        manifest["chroma_version"] = "unknown"

    # 4. 写 manifest
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not manifest["files"]:
        print(f"⚠️ 警告：{data_dir} 下没有找到可备份的数据文件")
        return 1

    print(f"✅ 备份完成：{backup_dir}")
    for f in manifest["files"]:
        print(f"   - {f['name']}（{f['method']}）")
    print(f"   - chroma_version={manifest['chroma_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
