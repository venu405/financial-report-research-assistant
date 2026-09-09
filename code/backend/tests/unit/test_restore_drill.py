"""Recovery-drill tests: restore only into a safe empty temporary directory."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import chromadb

from scripts import backup_kb, restore_drill, verify_backup


def _make_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample(value) VALUES ('restorable')")


def test_restore_drill_copies_and_reverifies_backup(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    backup_dir = tmp_path / "backup"
    target_dir = tmp_path / "drill-target"
    data_dir.mkdir()
    target_dir.mkdir()
    _make_database(data_dir / "kb_users.db")
    chroma_dir = data_dir / "chroma_data"
    chroma_dir.mkdir()
    (chroma_dir / "marker.txt").write_text("chroma", encoding="utf-8")
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_or_create_collection("restore_test")
    collection.add(ids=["one"], embeddings=[[0.1, 0.2]], documents=["restorable"])
    originals_dir = data_dir / "document_originals" / "doc-1"
    originals_dir.mkdir(parents=True)
    (originals_dir / "source.pdf").write_bytes(b"original document")

    assert backup_kb.main(["--data-dir", str(data_dir), "--backup-dir", str(backup_dir)]) == 0
    assert restore_drill.main(
        ["--backup-dir", str(backup_dir), "--target-dir", str(target_dir)]
    ) == 0
    assert verify_backup.verify_backup(target_dir) == []
    with sqlite3.connect(target_dir / "kb_users.db") as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == ("restorable",)
    assert (target_dir / "chroma_data" / "marker.txt").read_text(encoding="utf-8") == "chroma"
    restored_chroma = chromadb.PersistentClient(path=str(target_dir / "chroma_data"))
    assert restored_chroma.get_collection("restore_test").count() == 1
    assert (target_dir / "document_originals" / "doc-1" / "source.pdf").read_bytes() == b"original document"


def test_restore_drill_refuses_nonempty_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    backup_dir = tmp_path / "backup"
    target_dir = tmp_path / "nonempty-target"
    data_dir.mkdir()
    target_dir.mkdir()
    _make_database(data_dir / "kb_users.db")
    (data_dir / "chroma_data").mkdir()
    (target_dir / "do-not-overwrite.txt").write_text("keep", encoding="utf-8")

    assert backup_kb.main(["--data-dir", str(data_dir), "--backup-dir", str(backup_dir)]) == 0
    assert restore_drill.main(
        ["--backup-dir", str(backup_dir), "--target-dir", str(target_dir)]
    ) == 1
    assert (target_dir / "do-not-overwrite.txt").read_text(encoding="utf-8") == "keep"
