#!/usr/bin/env python3
"""Read-only integrity verification for an enterprise knowledge-base backup."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(members: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for member in members:
        digest.update(
            f"{member['name']}\0{member['size']}\0{member['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _safe_relative_path(name: Any) -> Path | None:
    if not isinstance(name, str) or not name or "\\" in name:
        return None
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _verify_sqlite(path: Path) -> str | None:
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        return f"SQLite integrity check failed for {path.name}: {exc}"
    if not result or result[0] != "ok":
        return f"SQLite integrity check failed for {path.name}: {result[0] if result else 'no result'}"
    return None


def _load_manifest(backup_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, ["manifest.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"manifest.json is invalid: {exc}"]
    if not isinstance(manifest, dict):
        return None, ["manifest.json must contain an object"]
    return manifest, []


def verify_backup(backup_dir: str | Path) -> list[str]:
    """Return all integrity errors without modifying the backup or its contents."""
    root = Path(backup_dir)
    manifest, errors = _load_manifest(root)
    if manifest is None:
        return errors
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported or missing manifest schema_version")
    signing_key = os.getenv("KB_BACKUP_SIGNING_KEY", "").encode("utf-8")
    require_signature = os.getenv("KB_REQUIRE_BACKUP_SIGNATURE", "").lower() in {
        "1", "true", "yes"
    }
    if manifest.get("signed"):
        signature_path = root / "manifest.sig"
        if not signing_key:
            errors.append("backup is signed but KB_BACKUP_SIGNING_KEY is unavailable")
        elif not signature_path.is_file():
            errors.append("manifest.sig is missing")
        else:
            expected = hmac.new(
                signing_key, (root / "manifest.json").read_bytes(), hashlib.sha256
            ).hexdigest()
            supplied = signature_path.read_text(encoding="ascii").strip()
            if not hmac.compare_digest(expected, supplied):
                errors.append("manifest signature mismatch")
    elif require_signature:
        errors.append("backup signature is required but this backup is unsigned")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("manifest files must be a non-empty list")
        return errors

    seen_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object file entry")
            continue
        name = entry.get("name")
        relative_path = _safe_relative_path(name)
        if relative_path is None:
            errors.append(f"unsafe manifest path: {name!r}")
            continue
        if name in seen_names:
            errors.append(f"duplicate manifest entry: {name}")
            continue
        seen_names.add(name)
        item_path = root / relative_path
        item_type = entry.get("type")
        if item_type == "sqlite":
            if not item_path.is_file() or item_path.is_symlink():
                errors.append(f"SQLite backup is missing or not a regular file: {name}")
                continue
            if entry.get("size") != item_path.stat().st_size:
                errors.append(f"size mismatch: {name}")
            if entry.get("sha256") != _sha256_file(item_path):
                errors.append(f"SHA-256 mismatch: {name}")
            sqlite_error = _verify_sqlite(item_path)
            if sqlite_error:
                errors.append(sqlite_error)
        elif item_type == "directory":
            if not item_path.is_dir() or item_path.is_symlink():
                errors.append(f"backup directory is missing or invalid: {name}")
                continue
            members = entry.get("members")
            if not isinstance(members, list):
                errors.append(f"directory members are missing: {name}")
                continue
            actual_members: list[dict[str, Any]] = []
            member_names: set[str] = set()
            for member in members:
                if not isinstance(member, dict):
                    errors.append(f"invalid directory member in {name}")
                    continue
                member_path = _safe_relative_path(member.get("name"))
                if member_path is None:
                    errors.append(f"unsafe directory member in {name}: {member.get('name')!r}")
                    continue
                member_name = member_path.as_posix()
                if member_name in member_names:
                    errors.append(f"duplicate directory member in {name}: {member_name}")
                    continue
                member_names.add(member_name)
                source = item_path / member_path
                if not source.is_file() or source.is_symlink():
                    errors.append(f"directory member is missing or invalid: {name}/{member_name}")
                    continue
                actual = {
                    "name": member_name,
                    "size": source.stat().st_size,
                    "sha256": _sha256_file(source),
                }
                actual_members.append(actual)
                if member.get("size") != actual["size"]:
                    errors.append(f"size mismatch: {name}/{member_name}")
                if member.get("sha256") != actual["sha256"]:
                    errors.append(f"SHA-256 mismatch: {name}/{member_name}")
                if name == "chroma_data" and member_name == "chroma.sqlite3":
                    sqlite_error = _verify_sqlite(source)
                    if sqlite_error:
                        errors.append(sqlite_error)
            found_names = {
                child.relative_to(item_path).as_posix()
                for child in item_path.rglob("*")
                if child.is_file() and not child.is_symlink()
            }
            if found_names != member_names:
                errors.append(f"directory contents mismatch: {name}")
            if entry.get("size") != sum(member["size"] for member in actual_members):
                errors.append(f"directory size mismatch: {name}")
            if entry.get("sha256") != _directory_sha256(actual_members):
                errors.append(f"directory SHA-256 mismatch: {name}")
        else:
            errors.append(f"unsupported manifest entry type for {name}: {item_type!r}")

    if not any(
        entry.get("type") == "directory" and entry.get("name") == "chroma_data"
        for entry in entries
        if isinstance(entry, dict)
    ):
        errors.append("manifest does not contain a Chroma directory entry")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only verification for a KB backup")
    parser.add_argument("--backup-dir", required=True, help="backup directory containing manifest.json")
    args = parser.parse_args(argv)
    errors = verify_backup(args.backup_dir)
    if errors:
        print("Backup verification failed:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1
    print(f"Backup verified: {Path(args.backup_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
