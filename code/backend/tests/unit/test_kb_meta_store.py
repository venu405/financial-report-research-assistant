from services.kb.kb_meta_store import KbMetaStore


def test_visibility_defaults_internal_and_can_be_published(tmp_path):
    store = KbMetaStore(tmp_path / "meta.db")
    created = store.create("legal", "法律制度")
    assert created["visibility"] == "internal"
    assert store.is_public("legal") is False

    assert store.update("legal", visibility="public") is True
    assert store.get("legal")["visibility"] == "public"
    assert store.public_ids() == {"legal"}


def test_old_schema_migrates_to_internal(tmp_path):
    import sqlite3

    path = tmp_path / "meta.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE kb_meta (kb_id TEXT PRIMARY KEY, name TEXT, "
        "description TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO kb_meta(kb_id, name, description, created_at) VALUES(?,?,?,?)",
        ("old", "旧库", "", "2026-01-01"),
    )
    conn.commit()
    conn.close()

    store = KbMetaStore(path)
    assert store.get("old")["visibility"] == "internal"
