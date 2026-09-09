"""Luna-1 implementation-side tests for the isolated corpus migration tool."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_corpus_to_qdrant.py"
SPEC = importlib.util.spec_from_file_location("migrate_corpus_to_qdrant_impl", SCRIPT)
assert SPEC and SPEC.loader
migrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migrate
SPEC.loader.exec_module(migrate)


def _args(**overrides):
    values = {
        "kb_id": None,
        "kb_map": ["cninfo=cninfo_report"],
        "kb_map_file": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_manifest(root: Path, files: list[tuple[str, bytes]]) -> Path:
    source = root / "code" / "backend" / "data_kb_test"
    source.mkdir(parents=True)
    rows: list[str] = []
    for relative, content in files:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = migrate._hash_file(path)[1]
        rows.append(
            "code/backend/data_kb_test/"
            + relative.replace("\\", "/")
            + f"\t{len(content)}\t{digest}"
        )
    manifest = root / "manifest.sha256"
    manifest.write_text(
        "# source_root=code/backend/data_kb_test; scan_count="
        + str(len(rows))
        + "; columns=relative_path<TAB>size_bytes<TAB>sha256\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_manifest_validates_paths_sizes_hashes_and_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate, "PROJECT_ROOT", tmp_path)
    manifest = _write_manifest(tmp_path, [("cninfo/a.pdf", b"pdf"), ("flk/a.docx", b"docx")])
    result = migrate.validate_source_manifest(
        tmp_path / "code" / "backend" / "data_kb_test",
        manifest,
        expected_count=2,
        expected_extensions={".pdf": 1, ".docx": 1},
    )
    assert len(result.files) == 2
    assert result.extension_counts == {".pdf": 1, ".docx": 1}


@pytest.mark.parametrize(
    "bad_row",
    [
        "../escape.pdf\t1\t" + "0" * 64,
        "code/backend/data_kb_test/a.pdf\tbad\t" + "0" * 64,
        "code/backend/data_kb_test/a.pdf\t1\t" + "0" * 63,
    ],
)
def test_manifest_rejects_path_traversal_or_bad_fields(bad_row):
    with pytest.raises(migrate.MigrationError):
        migrate._parse_manifest_rows(
            "# source_root=code/backend/data_kb_test; scan_count=1; columns=x\n" + bad_row
        )


def test_manifest_rejects_extra_file(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate, "PROJECT_ROOT", tmp_path)
    manifest = _write_manifest(tmp_path, [("cninfo/a.pdf", b"pdf")])
    (tmp_path / "code" / "backend" / "data_kb_test" / "cninfo" / "extra.pdf").write_bytes(b"extra")
    with pytest.raises(migrate.MigrationError, match="文件集合不一致"):
        migrate.validate_source_manifest(
            tmp_path / "code" / "backend" / "data_kb_test",
            manifest,
            expected_count=1,
            expected_extensions={".pdf": 1},
        )


def test_mapping_is_explicit_and_unknown_prefix_fails():
    source = migrate.SourceFile("root/unknown/a.pdf", "unknown/a.pdf", Path("a.pdf"), 1, "a" * 64, ".pdf")
    with pytest.raises(migrate.MigrationError, match="没有为"):
        migrate.kb_id_for_file(source, {"cninfo": "cninfo_report"})
    assert migrate.kb_id_for_file(source, {"*": "explicit"}) == "explicit"


@pytest.mark.parametrize("url", ["http://example.com:16333", "http://localhost:16333", "http://127.0.0.1:16333/path", "http://user:pass@127.0.0.1:16333"])
def test_only_explicit_loopback_url_is_accepted(url):
    with pytest.raises(migrate.MigrationError):
        migrate.validate_loopback_url(url, field="qdrant-url")


def test_staging_name_is_date_random_and_protected_names_fail():
    name = migrate.staging_collection_name("kb_sample", token="deadbeef")
    assert name.startswith("kb_sample_")
    assert name.endswith("_deadbeef")
    with pytest.raises(migrate.MigrationError):
        migrate.validate_collection_request("enterprise_kb_current")
    with pytest.raises(migrate.MigrationError):
        migrate.validate_collection_request("latest")


def test_lock_conflict_and_atomic_json(tmp_path):
    path = tmp_path / "state.json"
    migrate._atomic_write_json(path, {"中文": "ok"})
    assert json.loads(path.read_text(encoding="utf-8"))["中文"] == "ok"
    first = migrate.RunLock(tmp_path / "run.lock", run_id="one")
    second = migrate.RunLock(tmp_path / "run.lock", run_id="two")
    first.acquire()
    try:
        with pytest.raises(migrate.MigrationError, match="lock"):
            second.acquire()
    finally:
        first.release()


def test_default_mode_is_non_executing_and_nonzero(tmp_path, capsys):
    result = migrate.main(
        [
            "--source-dir", str(tmp_path),
            "--source-manifest", str(tmp_path / "manifest"),
            "--qdrant-url", "http://127.0.0.1:16333",
            "--collection", "kb_sample",
            "--kb-id", "kb-a",
            "--run-dir", str(tmp_path / "run"),
        ]
    )
    assert result == 2
    assert "必须显式指定" in capsys.readouterr().err


def test_scoped_point_id_and_error_redaction():
    assert migrate.point_id_for_chunk("same-0", "kb-a") != migrate.point_id_for_chunk("same-0", "kb-b")
    error = migrate._safe_error(RuntimeError("api-key=secret-token\nfull text should not be printed"))
    assert "secret-token" not in error
    assert "full text" in error


def _gate_args(*, collection: str = "kb_full_20260829_deadbeef"):
    return SimpleNamespace(collection=collection, qdrant_url="http://127.0.0.1:16333")


def _gate_inputs(tmp_path: Path):
    validation = migrate.ManifestValidation(
        manifest_path=tmp_path / "manifest.sha256",
        source_dir=tmp_path,
        source_root="code/backend/data_kb_test",
        manifest_sha256="M" * 64,
        files=(),
        extension_counts={},
    )
    identity = migrate.PipelineIdentity(
        script_sha256="S" * 64,
        code_sha256="C" * 64,
        config_sha256="F" * 64,
        config={},
    )
    return validation, identity


def test_full_gate_requires_bound_staging_and_t1_and_normalizes_hashes(tmp_path, monkeypatch):
    gate_path = tmp_path / "full-run-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "decision": "READY",
                "script_sha256": "s" * 64,
                "manifest_sha256": "m" * 64,
                "qdrant_url": "http://127.0.0.1:16333/",
                "embedding_model": "bge-m3",
                "embedding_dimension": 1024,
                "vector_distance": "Cosine",
                "staging_collection": "kb_full_20260829_deadbeef",
                "t1_snapshot": {
                    "collection": migrate.T1_COLLECTION,
                    "points": 100,
                    "vector_size": 1024,
                    "distance": "cosine",
                    "status": "green",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(migrate, "FULL_RUN_GATE", gate_path)
    validation, identity = _gate_inputs(tmp_path)
    result = migrate._full_gate_check(_gate_args(), validation, identity)
    assert result["decision"] == "READY"


def test_full_gate_rejects_missing_t1_binding(tmp_path, monkeypatch):
    gate_path = tmp_path / "full-run-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "decision": "READY",
                "script_sha256": "S" * 64,
                "manifest_sha256": "M" * 64,
                "qdrant_url": "http://127.0.0.1:16333",
                "embedding_model": "bge-m3",
                "embedding_dimension": 1024,
                "vector_distance": "Cosine",
                "staging_collection": "kb_full_20260829_deadbeef",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(migrate, "FULL_RUN_GATE", gate_path)
    validation, identity = _gate_inputs(tmp_path)
    with pytest.raises(migrate.MigrationError, match="T1"):
        migrate._full_gate_check(_gate_args(), validation, identity)


def test_qdrant_snapshot_requires_named_result():
    api = object.__new__(migrate.QdrantApi)
    api._request = lambda method, path: {"result": {"name": "kb_full.snapshot"}}
    assert api.create_snapshot("kb_full_20260829_deadbeef") == {"name": "kb_full.snapshot"}

    api._request = lambda method, path: {"result": {}}
    with pytest.raises(migrate.MigrationError, match="snapshot"):
        api.create_snapshot("kb_full_20260829_deadbeef")


class _VerifyApi:
    def __init__(self, points):
        self.points = points

    def collection_info(self, collection):
        return {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}

    def scroll_all(self, collection, *, page_size, with_vector, filter_body=None):
        if filter_body:
            values = {
                clause["key"]: clause["match"]["value"]
                for clause in filter_body["must"]
            }
            wanted_doc = values.get("doc_id")
            wanted_kb = values.get("kb_id")
            return [
                point for point in self.points
                if (wanted_doc is None or point["payload"]["doc_id"] == wanted_doc)
                and (wanted_kb is None or point["payload"]["kb_id"] == wanted_kb)
            ]
        return list(self.points)

    def count(self, collection, *, filter_body=None):
        return len(self.scroll_all(collection, page_size=64, with_vector=False, filter_body=filter_body))


class _VerifyStore:
    def get_chunk_by_id(self, chunk_id, *, kb_id):
        return {"payload": {"source_hash": "h" * 64}}

    def snapshot_doc(self, doc_id):
        return {"ids": [f"{doc_id}-0"], "embeddings": [[1.0] + [0.0] * 1023]}

    def search(self, vector, *, top_k, where, kb_id):
        return [{"payload": {"source_hash": "h" * 64}}]


def test_verify_checks_collection_file_kb_pagination_and_payload(tmp_path):
    doc_id = "doc-a"
    kb_id = "kb-a"
    logical_id = f"{doc_id}-0"
    physical_id = migrate.point_id_for_chunk(logical_id, kb_id)
    payload = {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "chunk_id": logical_id,
        "chunk_index": 0,
        "source_path": "cninfo/a.pdf",
        "source_hash": "h" * 64,
        "content_hash": "h" * 64,
        "migration_run_id": "run",
        "payload_version": 1,
        "embedding_model": "bge-m3",
        "embedding_dimension": 1024,
        "ingest_config_hash": "config",
        "source_manifest_hash": "manifest",
    }
    points = [{"id": physical_id, "payload": payload}]
    api = _VerifyApi(points)
    manifest = {
        "run_id": "run",
        "collection_name": "kb_sample_20260829_deadbeef",
        "selected_files": [{"manifest_path": "code/backend/data_kb_test/cninfo/a.pdf", "kb_id": kb_id, "chunk_count": 1}],
    }
    checkpoint = {
        "files": {
            "code/backend/data_kb_test/cninfo/a.pdf": {
                "status": "completed",
                "manifest_path": "code/backend/data_kb_test/cninfo/a.pdf",
                "source_hash": "h" * 64,
                "config_sha256": "config",
                "kb_id": kb_id,
                "doc_id": doc_id,
                "point_ids": [physical_id],
            }
        }
    }
    result = migrate.verify_run(api, _VerifyStore(), run_dir=tmp_path, run_manifest=manifest, checkpoint=checkpoint, page_size=1)
    assert result["collection_count"] == 1
    assert json.loads((tmp_path / "verify-report.json").read_text(encoding="utf-8"))["status"] == "verified"


def test_verify_rejects_duplicate_physical_ids(tmp_path):
    point = {"id": "same", "payload": {"kb_id": "kb", "chunk_id": "chunk"}}
    api = _VerifyApi([point, point])
    with pytest.raises(migrate.MigrationError, match="重复"):
        migrate.verify_run(api, _VerifyStore(), run_dir=tmp_path, run_manifest={"run_id": "r", "collection_name": "kb_sample_20260829_deadbeef", "selected_files": []}, checkpoint={"files": {}}, page_size=1)
