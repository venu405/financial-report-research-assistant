#!/usr/bin/env python3
"""Restore a verified backup only into an explicitly supplied empty drill directory."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

try:  # Supports both `python scripts/restore_drill.py` and package imports in tests.
    from scripts.verify_backup import verify_backup
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI invocation
    from verify_backup import verify_backup


def _load_manifest(backup_dir: Path) -> dict[str, Any]:
    return json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))


def _require_empty_drill_target(backup_dir: Path, target_dir: Path, manifest: dict[str, Any]) -> None:
    """Reject every target that could overwrite data or an existing backup."""
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError("--target-dir must be an existing empty directory")
    if any(target_dir.iterdir()):
        raise ValueError("--target-dir must be empty; refusing to overwrite data")
    backup_resolved = backup_dir.resolve()
    target_resolved = target_dir.resolve()
    if target_resolved == backup_resolved or backup_resolved in target_resolved.parents:
        raise ValueError("--target-dir must not be the backup directory or its child")
    source_data_dir = manifest.get("data_dir")
    if isinstance(source_data_dir, str):
        try:
            if target_resolved == Path(source_data_dir).resolve():
                raise ValueError("--target-dir matches manifest data_dir; refusing production restore")
        except OSError:
            pass


def _verify_chroma_load(target_dir: Path) -> None:
    """Open a restored Chroma database and enumerate collections when one exists."""
    chroma_dir = target_dir / "chroma_data"
    if not (chroma_dir / "chroma.sqlite3").is_file():
        return
    try:
        import chromadb  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="kb-chroma-verify-") as temp_dir:
            verification_dir = Path(temp_dir) / "chroma_data"
            shutil.copytree(chroma_dir, verification_dir)
            client = chromadb.PersistentClient(path=str(verification_dir))
            try:
                for collection in client.list_collections():
                    client.get_collection(collection.name).count()
            finally:
                client._system.stop()  # noqa: SLF001 - release Windows file handles
    except Exception as exc:
        raise RuntimeError(f"restored Chroma cannot be opened: {exc}") from exc


def restore_drill(backup_dir: str | Path, target_dir: str | Path) -> None:
    """Restore a verified backup into an empty drill directory and verify the copy."""
    backup = Path(backup_dir)
    target = Path(target_dir)
    errors = verify_backup(backup)
    if errors:
        raise ValueError("backup verification failed: " + "; ".join(errors))
    manifest = _load_manifest(backup)
    _require_empty_drill_target(backup, target, manifest)

    shutil.copy2(backup / "manifest.json", target / "manifest.json")
    if (backup / "manifest.sig").is_file():
        shutil.copy2(backup / "manifest.sig", target / "manifest.sig")
    for entry in manifest["files"]:
        name = Path(entry["name"])
        source = backup / name
        destination = target / name
        if entry["type"] == "sqlite":
            shutil.copy2(source, destination)
        elif entry["type"] == "directory":
            shutil.copytree(source, destination)
        else:  # verify_backup rejects this before this point; keep the guard explicit.
            raise ValueError(f"unsupported backup entry type: {entry['type']!r}")

    restored_errors = verify_backup(target)
    if restored_errors:
        raise RuntimeError("restored drill verification failed: " + "; ".join(restored_errors))
    _verify_chroma_load(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore a KB backup only for a safe recovery drill")
    parser.add_argument("--backup-dir", required=True, help="verified backup directory")
    parser.add_argument("--target-dir", required=True, help="existing empty recovery-drill directory")
    args = parser.parse_args(argv)
    try:
        restore_drill(args.backup_dir, args.target_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Restore drill failed: {exc}", file=sys.stderr)
        return 1
    print(f"Restore drill verified: {Path(args.target_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
