"""VectorStore contract tests against an isolated temporary Chroma store."""
from __future__ import annotations

import pytest

from services.kb.vector_store import ChromaVectorStore, VectorStore
from services.kb.vector_store_contract import VectorStoreBackend


def _add(
    store: VectorStoreBackend,
    doc_id: str,
    texts: list[str],
    kb_id: str,
    *,
    source_type: str = "pdf",
    extra_metadata: list[dict[str, object]] | None = None,
) -> list[str]:
    return store.add_chunks(
        embeddings=[[1.0, 0.0, 0.0, 0.0] for _ in texts],
        texts=texts,
        doc_id=doc_id,
        doc_title=doc_id,
        source_type=source_type,
        chunk_indices=list(range(len(texts))),
        kb_id=kb_id,
        extra_metadata=extra_metadata,
    )


def test_chroma_implements_backend_contract_and_preserves_hit_compatibility(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path))

    assert isinstance(store, VectorStoreBackend)
    assert VectorStore is ChromaVectorStore

    ids = _add(store, "doc-a", ["采购制度"], "default")
    assert ids == ["doc-a-0"]

    hit = store.search([1.0, 0.0, 0.0, 0.0], top_k=1, kb_id="default")[0]
    assert {"id", "payload", "score"} <= hit.keys()
    assert hit["id"] == "doc-a-0"
    assert hit["payload"]["chunk_id"] == "doc-a-0"
    assert hit["payload"]["text"] == "采购制度"
    assert hit["chunk_id"] == hit["id"]
    assert hit["text"] == hit["payload"]["text"]
    assert hit["metadata"]["kb_id"] == "default"
    assert hit["score"] == pytest.approx(1.0)


def test_contract_scopes_filters_counts_and_deletes_by_kb(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    _add(store, "default-doc", ["默认资料"], "default")
    _add(store, "product-doc", ["产品资料"], "product")
    _add(store, "default-md", ["默认 Markdown"], "default", source_type="md")

    default_hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        top_k=10,
        where={"source_type": "pdf"},
        kb_id="default",
    )
    assert [hit["metadata"]["kb_id"] for hit in default_hits] == ["default"]
    assert [item["chunk_id"] for item in store.all_items(
        where={"source_type": "pdf"}, kb_id="default"
    )] == ["default-doc-0"]
    assert store.list_kbs() == ["default", "product"]
    assert store.count(kb_id="default") == 2
    assert store.count(kb_id="product") == 1
    assert store.count() == 3

    # The ID belongs to product; a scoped delete in default must be a no-op.
    assert store.delete_chunk_ids(["product-doc-0"], kb_id="default") == 0
    assert store.count(kb_id="product") == 1
    assert store.delete_chunk_ids(["default-doc-0"], kb_id="default") == 1
    assert store.count(kb_id="default") == 1
    assert store.delete_kb("product") == 1
    assert store.count(kb_id="product") == 0


def test_contract_snapshot_relations_mutation_sequence_and_order(tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    assert store.mutation_seq("default") == 0
    assert store.mutation_seq("product") == 0

    _add(
        store,
        "parent-doc",
        ["父块", "子块一", "子块二"],
        "default",
        extra_metadata=[
            {"chunk_type": "parent"},
            {
                "chunk_type": "child",
                "parent_id": "parent-doc-0",
                "previous_chunk_id": "parent-doc-0",
            },
            {
                "chunk_type": "child",
                "parent_id": "parent-doc-0",
                "next_chunk_id": "parent-doc-1",
            },
        ],
    )
    assert store.mutation_seq("default") == 1
    assert store.mutation_seq("product") == 0
    assert [row["chunk_id"] for row in store.get_related_chunks(
        chunk_id="parent-doc-1", kb_id="default"
    )] == ["parent-doc-0", "parent-doc-1", "parent-doc-2"]
    assert store.get_chunks_by_parent_id("parent-doc-0", kb_id="default")[1][
        "metadata"
    ]["previous_chunk_id"] == "parent-doc-0"

    snapshot = store.snapshot_doc("parent-doc")
    _add(store, "parent-doc", ["新块一", "新块二", "新块三", "新块四"], "default")
    assert store.restore_doc_snapshot(snapshot) == 3
    assert store.get_doc_ids("parent-doc") == [
        "parent-doc-0",
        "parent-doc-1",
        "parent-doc-2",
    ]
    assert store.get_chunk_by_id("parent-doc-1", kb_id="default")["text"] == "子块一"

    _add(store, "doc-10", ["十"], "default")
    _add(store, "doc-2", ["二"], "default")
    docs, total = store.list_docs(kb_id="default")
    assert total == 3
    assert [doc["doc_id"] for doc in docs] == ["doc-10", "doc-2", "parent-doc"]
