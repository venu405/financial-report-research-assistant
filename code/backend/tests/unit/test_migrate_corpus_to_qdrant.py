"""Luna-2 independent safety tests for the Qdrant corpus migration tool.

This file intentionally does not patch the Luna-1 implementation.  It checks
the public gate helpers and the failure boundaries from a separate test
module, including cases not covered by the implementation-side tests.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_corpus_to_qdrant.py"
SPEC = importlib.util.spec_from_file_location("migrate_corpus_to_qdrant_luna2", SCRIPT)
assert SPEC and SPEC.loader
migrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migrate
SPEC.loader.exec_module(migrate)


def _write_manifest(
    root: Path,
    rows: list[tuple[str, bytes]],
    *,
    manifest_rows: list[tuple[str, int, str]] | None = None,
) -> Path:
    source = root / "code" / "backend" / "data_kb_test"
    source.mkdir(parents=True, exist_ok=True)
    computed: list[tuple[str, int, str]] = []
    for relative, content in rows:
        path = source / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        computed.append((relative.replace("\\", "/"), len(content), migrate._hash_file(path)[1]))
    manifest_rows = computed if manifest_rows is None else manifest_rows
    manifest = root / "source-manifest.sha256"
    lines = [
        "# source_root=code/backend/data_kb_test; scan_count="
        f"{len(manifest_rows)}; columns=relative_path<TAB>size_bytes<TAB>sha256"
    ]
    lines.extend(
        "code/backend/data_kb_test/"
        f"{relative}\t{size}\t{digest}"
        for relative, size, digest in manifest_rows
    )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _validate_small(
    root: Path,
    rows: list[tuple[str, bytes]],
    manifest: Path | None = None,
) -> migrate.ManifestValidation:
    return migrate.validate_source_manifest(
        root / "code" / "backend" / "data_kb_test",
        manifest or _write_manifest(root, rows),
        expected_count=len(rows),
        expected_extensions={
            extension: sum(Path(relative).suffix.lower() == extension for relative, _ in rows)
            for extension in {Path(relative).suffix.lower() for relative, _ in rows}
        },
    )


def test_modes_are_mutually_exclusive_and_default_is_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        migrate.parse_args(
            [
                "--source-dir",
                str(tmp_path),
                "--source-manifest",
                str(tmp_path / "manifest"),
                "--collection",
                "kb_sample",
                "--run-dir",
                str(tmp_path / "run"),
                "--kb-id",
                "kb-a",
                "--dry-run",
                "--execute",
            ]
        )
    result = migrate.main(
        [
            "--source-dir",
            str(tmp_path),
            "--source-manifest",
            str(tmp_path / "manifest"),
            "--collection",
            "kb_sample",
            "--run-dir",
            str(tmp_path / "run"),
            "--kb-id",
            "kb-a",
        ]
    )
    assert result == 2


def test_manifest_rejects_duplicate_case_extra_missing_hash_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migrate, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "code" / "backend" / "data_kb_test"
    source.mkdir(parents=True)
    (source / "cninfo").mkdir()
    (source / "cninfo" / "a.pdf").write_bytes(b"pdf")

    duplicate_manifest = _write_manifest(
        tmp_path,
        [("cninfo/a.pdf", b"pdf")],
        manifest_rows=[
            ("cninfo/a.pdf", 3, migrate._hash_file(source / "cninfo" / "a.pdf")[1]),
            ("cninfo/a.pdf", 3, migrate._hash_file(source / "cninfo" / "a.pdf")[1]),
        ],
    )
    with pytest.raises(migrate.MigrationError, match="重复路径"):
        migrate.validate_source_manifest(
            source,
            duplicate_manifest,
            expected_count=2,
            expected_extensions={".pdf": 2},
        )

    case_manifest = _write_manifest(
        tmp_path / "case",
        [("cninfo/a.pdf", b"pdf")],
        manifest_rows=[
            ("cninfo/a.pdf", 3, migrate._hash_file(source / "cninfo" / "a.pdf")[1]),
            ("cninfo/A.PDF", 3, migrate._hash_file(source / "cninfo" / "a.pdf")[1]),
        ],
    )
    case_root = tmp_path / "case"
    monkeypatch.setattr(migrate, "PROJECT_ROOT", case_root)
    case_source = case_root / "code" / "backend" / "data_kb_test"
    with pytest.raises(migrate.MigrationError, match="大小写碰撞"):
        migrate.validate_source_manifest(
            case_source,
            case_manifest,
            expected_count=2,
            expected_extensions={".pdf": 2},
        )

    extra_root = tmp_path / "extra"
    extra_manifest = _write_manifest(extra_root, [("cninfo/a.pdf", b"pdf")])
    (extra_root / "code" / "backend" / "data_kb_test" / "cninfo" / "extra.pdf").write_bytes(b"x")
    monkeypatch.setattr(migrate, "PROJECT_ROOT", extra_root)
    with pytest.raises(migrate.MigrationError, match="文件集合不一致"):
        _validate_small(extra_root, [("cninfo/a.pdf", b"pdf")], extra_manifest)

    missing_root = tmp_path / "missing"
    missing_manifest = _write_manifest(
        missing_root,
        [],
        manifest_rows=[("cninfo/missing.pdf", 1, "0" * 64)],
    )
    monkeypatch.setattr(migrate, "PROJECT_ROOT", missing_root)
    with pytest.raises(migrate.MigrationError, match="文件缺失"):
        _validate_small(missing_root, [("cninfo/missing.pdf", b"x")], missing_manifest)

    hash_root = tmp_path / "hash"
    hash_manifest = _write_manifest(
        hash_root,
        [("cninfo/a.pdf", b"pdf")],
        manifest_rows=[("cninfo/a.pdf", 3, "0" * 64)],
    )
    monkeypatch.setattr(migrate, "PROJECT_ROOT", hash_root)
    with pytest.raises(migrate.MigrationError, match="hash mismatch"):
        _validate_small(hash_root, [("cninfo/a.pdf", b"pdf")], hash_manifest)

    size_root = tmp_path / "size"
    size_manifest = _write_manifest(
        size_root,
        [("cninfo/a.pdf", b"pdf")],
        manifest_rows=[("cninfo/a.pdf", 4, migrate._sha256_bytes(b"pdf"))],
    )
    monkeypatch.setattr(migrate, "PROJECT_ROOT", size_root)
    with pytest.raises(migrate.MigrationError, match="size mismatch"):
        _validate_small(size_root, [("cninfo/a.pdf", b"pdf")], size_manifest)


@pytest.mark.parametrize(
    "row",
    [
        "/code/backend/data_kb_test/cninfo/a.pdf\t1\t" + "0" * 64,
        "code/backend/data_kb_test/../escape.pdf\t1\t" + "0" * 64,
        "code/backend/data_kb_test/a.pdf\tbad\t" + "0" * 64,
        "code/backend/data_kb_test/a.pdf\t1\t" + "0" * 63,
    ],
)
def test_manifest_parser_rejects_traversal_and_bad_fields(row: str) -> None:
    with pytest.raises(migrate.MigrationError):
        migrate._parse_manifest_rows(
            "# source_root=code/backend/data_kb_test; scan_count=1; columns=x\n" + row
        )


def test_loopback_collection_t1_reserved_and_alias_guards() -> None:
    assert migrate.validate_loopback_url(
        "http://127.0.0.1:16333/", field="qdrant-url"
    ) == "http://127.0.0.1:16333"
    for value in (
        "http://example.com:16333",
        "http://localhost:16333",
        "http://127.0.0.1:16333/path",
        "http://user:pass@127.0.0.1:16333",
    ):
        with pytest.raises(migrate.MigrationError):
            migrate.validate_loopback_url(value, field="qdrant-url")
    for name in (
        migrate.T1_COLLECTION,
        "latest",
        "default",
        "production",
        "enterprise_kb_current",
    ):
        with pytest.raises(migrate.MigrationError):
            migrate.validate_collection_request(name)

    class FakeApi:
        def collection_names(self) -> list[str]:
            return []

        def aliases(self) -> dict[str, str]:
            return {"kb_sample_20260830_deadbeef": "some_collection"}

    with pytest.raises(migrate.MigrationError, match="alias"):
        migrate._prepare_remote_target(
            FakeApi(),
            collection="kb_sample_20260830_deadbeef",
            run_manifest=None,
            creating=True,
        )


def test_atomic_checkpoint_and_cross_process_lock(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    migrate._atomic_write_json(path, {"中文": "ok"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"中文": "ok"}
    assert not list(tmp_path.glob(".checkpoint.json.*.tmp"))

    first = migrate.RunLock(tmp_path / "run.lock", run_id="one")
    second = migrate.RunLock(tmp_path / "run.lock", run_id="two")
    first.acquire()
    try:
        with pytest.raises(migrate.MigrationError, match="lock"):
            second.acquire()
    finally:
        first.release()
    assert not (tmp_path / "run.lock").exists()


def test_ctrl_c_preserves_active_checkpoint_and_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "a.pdf"
    source_path.write_bytes(b"pdf")
    source = migrate.SourceFile(
        "code/backend/data_kb_test/cninfo/a.pdf",
        "cninfo/a.pdf",
        source_path,
        3,
        migrate._hash_file(source_path)[1],
        ".pdf",
    )
    run_manifest = {
        "run_id": "run-1",
        "config_sha256": "config",
        "source_manifest_sha256": "manifest",
        "code_sha256": "code",
        "script_sha256": "script",
    }
    checkpoint: dict[str, Any] = {}

    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(migrate, "build_chunks", interrupt)
    with pytest.raises(KeyboardInterrupt):
        migrate._process_file(
            source=source,
            kb_id="cninfo_report",
            run_manifest=run_manifest,
            checkpoint=checkpoint,
            run_dir=tmp_path / "run",
            store=object(),
            embedder=object(),
            batch_size=1,
            ocr_mode="local",
        )
    persisted = json.loads((tmp_path / "run" / "checkpoint.json").read_text(encoding="utf-8"))
    assert persisted["active_file"]["manifest_path"] == source.manifest_path

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("authorization=secret-token")

    monkeypatch.setattr(migrate, "build_chunks", fail)
    record = migrate._process_file(
        source=source,
        kb_id="cninfo_report",
        run_manifest=run_manifest,
        checkpoint={},
        run_dir=tmp_path / "failed-run",
        store=object(),
        embedder=object(),
        batch_size=1,
        ocr_mode="local",
    )
    assert record["status"] == "failed"
    assert record["error_summary"]
    assert "secret-token" not in record["error_summary"]


def test_error_redaction_covers_bearer_secret() -> None:
    redacted = migrate._safe_error(
        RuntimeError("Authorization: Bearer secret-token")
    )
    assert "secret-token" not in redacted


def test_scroll_all_requires_advancing_offset() -> None:
    api = object.__new__(migrate.QdrantApi)
    responses = iter(
        [
            {"result": {"points": [{"id": "1"}], "next_page_offset": "next"}},
            {"result": {"points": [{"id": "2"}], "next_page_offset": None}},
        ]
    )
    api._request = lambda method, path, **kwargs: next(responses)
    assert [point["id"] for point in api.scroll_all("kb", page_size=1)] == ["1", "2"]

    api._request = lambda method, path, **kwargs: {
        "result": {"points": [], "next_page_offset": "same"}
    }
    with pytest.raises(migrate.MigrationError, match="未前进"):
        api.scroll_all("kb", page_size=1)


def test_pipeline_fingerprint_covers_execution_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = migrate.ManifestValidation(
        manifest_path=Path("manifest"),
        source_dir=Path("source"),
        source_root="code/backend/data_kb_test",
        manifest_sha256="m" * 64,
        files=(),
        extension_counts={},
    )
    args = SimpleNamespace(
        qdrant_url="http://127.0.0.1:16333",
        ollama_url="http://127.0.0.1:11434",
        ocr_mode="local",
        collection="kb_sample",
        batch_size=7,
    )
    identity = migrate.pipeline_identity(args, validation, {"cninfo": "cninfo_report"})
    config = identity.config
    assert config["collection"] == "kb_sample"
    assert config["batch_size"] == 7


def test_full_gate_binds_test_config_and_t1_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_path = tmp_path / "full-run-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "decision": "READY",
                "script_sha256": "s" * 64,
                "test_sha256": "wrong-test" * 8,
                "config_sha256": "wrong-config" * 8,
                "manifest_sha256": "m" * 64,
                "qdrant_url": "http://127.0.0.1:16333",
                "embedding_model": "bge-m3",
                "embedding_dimension": 1024,
                "vector_distance": "Cosine",
                "staging_collection": "kb_full_20260830_deadbeef",
                "t1_snapshot": {
                    "collection": migrate.T1_COLLECTION,
                    "points": 100,
                    "vector_size": 1024,
                    "distance": "Cosine",
                    "status": "green",
                    "container_id": "wrong-container",
                    "image_digest": "wrong-digest",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(migrate, "FULL_RUN_GATE", gate_path)
    validation = migrate.ManifestValidation(
        manifest_path=tmp_path / "manifest",
        source_dir=tmp_path,
        source_root="code/backend/data_kb_test",
        manifest_sha256="m" * 64,
        files=(),
        extension_counts={},
    )
    identity = migrate.PipelineIdentity(
        "s" * 64,
        "c" * 64,
        "f" * 64,
        {"test_sha256": "expected-test", "config_sha256": "f" * 64},
    )
    with pytest.raises(migrate.MigrationError):
        migrate._full_gate_check(
            SimpleNamespace(
                collection="kb_full_20260830_deadbeef",
                qdrant_url="http://127.0.0.1:16333",
            ),
            validation,
            identity,
        )


def test_loaded_run_rejects_selected_scope_or_document_tampering() -> None:
    source = migrate.SourceFile(
        "code/backend/data_kb_test/cninfo/a.pdf",
        "cninfo/a.pdf",
        Path("a.pdf"),
        3,
        "h" * 64,
        ".pdf",
    )
    validation = migrate.ManifestValidation(
        manifest_path=Path("manifest"),
        source_dir=Path("source"),
        source_root="code/backend/data_kb_test",
        manifest_sha256="m" * 64,
        files=(source,),
        extension_counts={".pdf": 1},
    )
    identity = migrate.PipelineIdentity("s" * 64, "c" * 64, "f" * 64, {})
    args = SimpleNamespace(
        qdrant_url="http://127.0.0.1:16333",
        collection="kb_sample",
    )
    run_manifest = {
        "source_manifest_sha256": "m" * 64,
        "script_sha256": "s" * 64,
        "code_sha256": "c" * 64,
        "config_sha256": "f" * 64,
        "qdrant_url": "http://127.0.0.1:16333",
        "kb_mapping": {"cninfo": "cninfo_report"},
        "embedding_model": "bge-m3",
        "embedding_dimension": 1024,
        "vector_distance": "Cosine",
        "collection_base": "kb_sample",
        "collection_name": "kb_sample_20260830_deadbeef",
        "selected_files": [
            {
                "manifest_path": source.manifest_path,
                "source_hash": source.sha256,
                "size": source.size,
                "kb_id": "wrong-kb",
                "doc_id": "wrong-doc",
            }
        ],
    }
    with pytest.raises(migrate.FingerprintMismatch):
        migrate._check_loaded_run(
            run_manifest,
            args=args,
            validation=validation,
            mapping={"cninfo": "cninfo_report"},
            identity=identity,
        )


def test_verify_rejects_point_from_another_run() -> None:
    physical_id = migrate.point_id_for_chunk("doc-a-0", "kb-a")
    point = {
        "id": physical_id,
        "payload": {
            "kb_id": "kb-a",
            "doc_id": "doc-a",
            "chunk_id": "doc-a-0",
            "chunk_index": 0,
            "source_path": "cninfo/a.pdf",
            "source_hash": "h" * 64,
            "content_hash": "h" * 64,
            "migration_run_id": "other-run",
            "payload_version": 1,
            "embedding_model": "bge-m3",
            "embedding_dimension": 1024,
            "ingest_config_hash": "config",
            "source_manifest_hash": "manifest",
        },
    }

    class Api:
        def collection_info(self, collection: str) -> dict[str, Any]:
            return {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}

        def scroll_all(self, collection: str, **kwargs: Any) -> list[dict[str, Any]]:
            filter_body = kwargs.get("filter_body")
            if filter_body:
                return [point]
            return [point]

        def count(self, collection: str, *, filter_body: Any = None) -> int:
            return 1

    class Store:
        def get_chunk_by_id(self, chunk_id: str, *, kb_id: str) -> dict[str, Any]:
            return {"payload": {"source_hash": "h" * 64}}

        def snapshot_doc(self, doc_id: str) -> dict[str, Any]:
            return {"ids": ["doc-a-0"], "embeddings": [[1.0] + [0.0] * 1023]}

        def search(self, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
            return [{"payload": {"source_hash": "h" * 64}}]

    manifest = {
        "run_id": "expected-run",
        "collection_name": "kb_sample_20260830_deadbeef",
        "selected_files": [
            {
                "manifest_path": "code/backend/data_kb_test/cninfo/a.pdf",
                "kb_id": "kb-a",
                "chunk_count": 1,
            }
        ],
    }
    checkpoint = {
        "files": {
            "code/backend/data_kb_test/cninfo/a.pdf": {
                "status": "completed",
                "manifest_path": "code/backend/data_kb_test/cninfo/a.pdf",
                "source_hash": "h" * 64,
                "config_sha256": "config",
                "kb_id": "kb-a",
                "doc_id": "doc-a",
                "point_ids": [physical_id],
            }
        }
    }
    with pytest.raises(migrate.FingerprintMismatch, match="run"):
        migrate.verify_run(
            Api(),
            Store(),
            run_dir=Path("."),
            run_manifest=manifest,
            checkpoint=checkpoint,
            page_size=1,
        )
