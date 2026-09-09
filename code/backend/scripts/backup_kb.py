#!/usr/bin/env python3
"""企业知识库数据备份：SQLite 一致性快照 + Chroma 目录复制 + manifest。

备份范围（data-dir 下）：
  - 所有 kb_*.db     SQLite 业务数据（动态发现，未来新增文件也会自动纳入）
  - chroma_data/     向量库（Chroma）

用法（在 code/backend 目录下）：
  ./.venv/Scripts/python.exe scripts/backup_kb.py                # 默认 data-dir=backend，输出 backups/kb_<时间戳>
  ./.venv/Scripts/python.exe scripts/backup_kb.py --data-dir D:/prod/kb_data --backup-dir D:/backups/kb

设计要点：
  - SQLite 用 VACUUM INTO 做一致性快照（WAL 模式下安全，备份时业务写不中断）。
  - Chroma 目录整体 copytree（含 chroma.sqlite3 与段落文件）。
  - manifest.json 记录时间、Chroma 版本、文件清单，供恢复/升级核对。
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import platform
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIRECTORIES = ("chroma_data", "document_originals")
MANIFEST_SCHEMA_VERSION = 2


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_members(path: Path) -> list[dict[str, str | int]]:
    """Describe every regular file in a directory with stable relative paths."""
    members: list[dict[str, str | int]] = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            members.append(
                {
                    "name": child.relative_to(path).as_posix(),
                    "size": child.stat().st_size,
                    "sha256": _sha256_file(child),
                }
            )
    return members


def _directory_sha256(members: list[dict[str, str | int]]) -> str:
    """Hash directory member metadata so additions and deletions are detected."""
    digest = hashlib.sha256()
    for member in members:
        digest.update(
            f"{member['name']}\0{member['size']}\0{member['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _vacuum_snapshot(src: Path, dst: Path) -> None:
    """SQLite 一致性快照（VACUUM INTO，不阻塞业务写）。"""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("VACUUM INTO ?", (str(dst),))
    finally:
        conn.close()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_data_dir(backend_dir: Path, data_dir_arg: str | None) -> Path:
    """解析数据目录；未指定时使用 code/backend（启动默认工作目录）。"""
    return Path(data_dir_arg) if data_dir_arg else backend_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="备份企业知识库数据")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="数据目录（默认 code/backend，含 chroma_data + kb_*.db）",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="备份输出目录（默认 backend/backups/kb_<时间戳>）",
    )
    args = parser.parse_args(argv)

    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = _resolve_data_dir(backend_dir, args.data_dir)
    signing_key = os.getenv("KB_BACKUP_SIGNING_KEY", "").encode("utf-8")
    require_signature = os.getenv("KB_REQUIRE_BACKUP_SIGNATURE", "").lower() in {
        "1", "true", "yes"
    }
    if require_signature and len(signing_key) < 32:
        print("Backup signing is required but KB_BACKUP_SIGNING_KEY is shorter than 32 bytes")
        return 2
    backup_dir = (
        Path(args.backup_dir)
        if args.backup_dir
        else backend_dir / "backups" / f"kb_{_ts()}"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "backup_format": "enterprise-kb",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "backup_name": backup_dir.name,
        "python_version": platform.python_version(),
        "files": [],
        "total_bytes": 0,
    }

    # 1. 动态发现并制作所有 SQLite 一致性快照
    for src in sorted(data_dir.glob("kb_*.db")):
        if not src.is_file():
            continue
        name = src.name
        dst = backup_dir / name
        _vacuum_snapshot(src, dst)
        manifest["files"].append(
            {
                "name": name,
                "type": "sqlite",
                "size": dst.stat().st_size,
                "sha256": _sha256_file(dst),
                "method": "vacuum_into",
            }
        )

    # 2. Chroma 目录整体复制
    for directory_name in DATA_DIRECTORIES:
        source_dir = data_dir / directory_name
        if not source_dir.exists():
            continue
        destination_dir = backup_dir / directory_name
        shutil.copytree(source_dir, destination_dir)
        members = _directory_members(destination_dir)
        manifest["files"].append(
            {
                "name": directory_name,
                "type": "directory",
                "method": "copytree",
                "size": sum(int(member["size"]) for member in members),
                "sha256": _directory_sha256(members),
                "members": members,
            }
        )
    manifest["total_bytes"] = sum(int(entry.get("size", 0)) for entry in manifest["files"])

    # 3. Chroma 版本（升级时核对兼容性）
    try:
        import chromadb  # noqa: PLC0415

        manifest["chroma_version"] = chromadb.__version__
    except Exception:  # pragma: no cover - chromadb 未装时降级
        manifest["chroma_version"] = "unknown"

    # 4. Write manifest and an optional HMAC signature kept outside the manifest.
    manifest["signed"] = bool(signing_key)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    (backup_dir / "manifest.json").write_bytes(manifest_bytes)
    if signing_key:
        signature = hmac.new(signing_key, manifest_bytes, hashlib.sha256).hexdigest()
        (backup_dir / "manifest.sig").write_text(signature, encoding="ascii")

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
