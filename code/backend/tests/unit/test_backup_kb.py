"""备份脚本测试：默认目录、动态 SQLite 发现和一致性快照。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import backup_kb, verify_backup

CURRENT_DATABASES = {
    "kb_users.db",
    "kb_checkpoints.db",
    "kb_audit.db",
    "kb_faq.db",
    "kb_persona.db",
    "kb_feedback.db",
    "kb_conversations.db",
    "kb_tickets.db",
    "kb_quick_replies.db",
    "kb_meta.db",
    "kb_retrieval_log.db",
    "kb_governance.db",
    "kb_privacy.db",
    "kb_sources.db",
    "kb_alerts.db",
}


def _make_database(path: Path, value: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sample(value) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def test_default_data_dir_is_backend(tmp_path: Path) -> None:
    backend_dir = tmp_path / "code" / "backend"

    assert backup_kb._resolve_data_dir(backend_dir, None) == backend_dir
    assert backup_kb._resolve_data_dir(backend_dir, "D:/custom/kb") == Path("D:/custom/kb")


def test_main_uses_backend_as_default_data_dir(tmp_path: Path, monkeypatch) -> None:
    backend_dir = tmp_path / "backend"
    scripts_dir = backend_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    _make_database(backend_dir / "kb_default.db", "default")
    (backend_dir / "chroma_data").mkdir()
    (backend_dir / "chroma_data" / "marker.txt").write_text("chroma", encoding="utf-8")
    monkeypatch.setattr(backup_kb, "__file__", str(scripts_dir / "backup_kb.py"))
    backup_dir = tmp_path / "backup"

    assert backup_kb.main(["--backup-dir", str(backup_dir)]) == 0

    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_dir"] == str(backend_dir)
    assert (backup_dir / "kb_default.db").is_file()
    assert (backup_dir / "chroma_data" / "marker.txt").read_text(encoding="utf-8") == "chroma"


def test_backup_discovers_all_sqlite_files_and_snapshots_each(tmp_path: Path) -> None:
    data_dir = tmp_path / "backend"
    backup_dir = tmp_path / "backup"
    data_dir.mkdir()
    for index, name in enumerate(sorted(CURRENT_DATABASES | {"kb_future.db"})):
        _make_database(data_dir / name, f"value-{index}")
    _make_database(data_dir / "not_a_kb_database.db", "excluded")
    (data_dir / "chroma_data").mkdir()
    originals_dir = data_dir / "document_originals" / "doc-1"
    originals_dir.mkdir(parents=True)
    (originals_dir / "source.pdf").write_bytes(b"original document")

    assert backup_kb.main(
        ["--data-dir", str(data_dir), "--backup-dir", str(backup_dir)]
    ) == 0

    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["files"]}
    expected = CURRENT_DATABASES | {"kb_future.db"}
    assert expected <= entries.keys()
    assert "not_a_kb_database.db" not in entries
    assert entries["chroma_data"]["type"] == "directory"
    assert entries["document_originals"]["type"] == "directory"
    assert (backup_dir / "document_originals" / "doc-1" / "source.pdf").read_bytes() == b"original document"

    for name in expected:
        entry = entries[name]
        assert entry["type"] == "sqlite"
        assert entry["method"] == "vacuum_into"
        assert entry["size"] > 0
        assert len(entry["sha256"]) == 64
        with sqlite3.connect(backup_dir / name) as conn:
            assert conn.execute("SELECT value FROM sample").fetchone() is not None
    assert entries["chroma_data"]["members"] == []
    assert len(entries["chroma_data"]["sha256"]) == 64
    assert verify_backup.verify_backup(backup_dir) == []


def test_verify_backup_detects_tampered_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    backup_dir = tmp_path / "backup"
    data_dir.mkdir()
    _make_database(data_dir / "kb_users.db", "original")
    (data_dir / "chroma_data").mkdir()

    assert backup_kb.main(["--data-dir", str(data_dir), "--backup-dir", str(backup_dir)]) == 0

    database = backup_dir / "kb_users.db"
    database.write_bytes(database.read_bytes() + b"tampered")

    errors = verify_backup.verify_backup(backup_dir)
    assert any(error == "SHA-256 mismatch: kb_users.db" for error in errors)
    assert verify_backup.main(["--backup-dir", str(backup_dir)]) == 1


def test_signed_manifest_detects_manifest_tampering(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    backup_dir = tmp_path / "backup"
    data_dir.mkdir()
    _make_database(data_dir / "kb_users.db", "original")
    (data_dir / "chroma_data").mkdir()
    monkeypatch.setenv("KB_BACKUP_SIGNING_KEY", "test-signing-secret")

    assert backup_kb.main(["--data-dir", str(data_dir), "--backup-dir", str(backup_dir)]) == 0
    assert verify_backup.verify_backup(backup_dir) == []
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["backup_name"] = "tampered"
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert "manifest signature mismatch" in verify_backup.verify_backup(backup_dir)


def test_backup_refuses_missing_required_signing_key(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("KB_REQUIRE_BACKUP_SIGNATURE", "1")
    monkeypatch.delenv("KB_BACKUP_SIGNING_KEY", raising=False)

    assert backup_kb.main([
        "--data-dir", str(data_dir), "--backup-dir", str(tmp_path / "backup")
    ]) == 2
    assert not (tmp_path / "backup").exists()
