"""Live synthetic integration slice for the Qdrant VectorStore adapter."""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import pytest

from services.kb.qdrant_vector_store import QdrantVectorStore


def _allow_explicit_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the test-suite network guard only for this explicit live test."""

    def live_request(
        client: httpx.Client,
        method: str,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        # httpx.Client.request supplies auth/follow_redirects defaults, while
        # BaseClient.build_request does not accept those two arguments.
        kwargs.pop("auth", None)
        follow_redirects = kwargs.pop("follow_redirects", None)
        request = client.build_request(method, url, *args, **kwargs)
        return client.send(
            request,
            follow_redirects=(
                client.follow_redirects
                if follow_redirects is None
                else follow_redirects
            ),
        )

    monkeypatch.setattr(httpx.Client, "request", live_request)


def _vector(seed: float) -> list[float]:
    return [seed, 1.0 - seed, 0.0, 0.0]


def _add_doc(
    store: QdrantVectorStore,
    *,
    doc_id: str,
    kb_id: str,
    texts: list[str],
    source_type: str = "pdf",
    content_hash: str | None = None,
    extra_metadata: list[dict[str, object]] | None = None,
) -> list[str]:
    return store.add_chunks(
        embeddings=[_vector(0.9) for _ in texts],
        texts=texts,
        doc_id=doc_id,
        doc_title=doc_id,
        source_type=source_type,
        chunk_indices=list(range(len(texts))),
        kb_id=kb_id,
        content_hash=content_hash,
        extra_metadata=extra_metadata,
    )


def test_live_qdrant_adapter_protocol_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise one complete synthetic document lifecycle on one temp collection."""

    url = os.getenv("KB_QDRANT_TEST_URL")
    if not url:
        pytest.skip("KB_QDRANT_TEST_URL is not set")
    _allow_explicit_live_http(monkeypatch)

    collection = f"kb_t3_adapter_test_{uuid.uuid4().hex[:12]}"
    print(f"T3 temporary collection: {collection}", flush=True)
    client = httpx.Client(base_url=url.rstrip("/"), timeout=10.0)
    store: QdrantVectorStore | None = None

    try:
        store = QdrantVectorStore(
            url=url,
            collection_name=collection,
            vector_size=4,
            timeout=10.0,
            create_if_missing=True,
            scroll_page_size=2,
        )

        info_response = client.get(f"/collections/{collection}")
        assert info_response.status_code == 200
        info = info_response.json()["result"]
        vectors = info["config"]["params"]["vectors"]
        assert vectors["size"] == 4
        assert vectors["distance"] == "Cosine"
        payload_schema = info.get("payload_schema", {})
        assert "kb_id" in payload_schema

        seq_a_before = store.mutation_seq("kb-a")
        ids_a = _add_doc(
            store,
            doc_id="doc-a",
            kb_id="kb-a",
            texts=["parent", "child one", "child two"],
            content_hash="hash-a",
            extra_metadata=[
                {"chunk_type": "parent"},
                {"chunk_type": "child", "parent_id": "doc-a-0"},
                {"chunk_type": "child", "parent_id": "doc-a-0"},
            ],
        )
        ids_b = _add_doc(
            store,
            doc_id="doc-b",
            kb_id="kb-b",
            texts=["other one", "other two"],
            source_type="md",
            content_hash="hash-b",
        )
        ids_c = _add_doc(
            store,
            doc_id="doc-c",
            kb_id="kb-a",
            texts=["third document"],
            content_hash="hash-c",
        )
        assert ids_a == ["doc-a-0", "doc-a-1", "doc-a-2"]
        assert ids_b == ["doc-b-0", "doc-b-1"]
        assert ids_c == ["doc-c-0"]
        assert store.mutation_seq("kb-a") > seq_a_before

        count_before_repeat = store.count()
        assert count_before_repeat == 6
        assert _add_doc(
            store,
            doc_id="doc-a",
            kb_id="kb-a",
            texts=["parent", "child one", "child two"],
            content_hash="hash-a",
            extra_metadata=[
                {"chunk_type": "parent"},
                {"chunk_type": "child", "parent_id": "doc-a-0"},
                {"chunk_type": "child", "parent_id": "doc-a-0"},
            ],
        ) == ids_a
        assert store.count() == count_before_repeat

        hits = store.search(_vector(0.9), top_k=10, kb_id="kb-a")
        assert hits
        assert all(hit["payload"]["kb_id"] == "kb-a" for hit in hits)
        assert {"id", "payload", "score"} <= hits[0].keys()
        assert hits[0]["chunk_id"] == hits[0]["payload"]["chunk_id"]
        assert str(uuid.UUID(hits[0]["id"])) == hits[0]["id"]
        assert hits[0]["text"] == hits[0]["payload"]["text"]

        and_hits = store.search(
            _vector(0.9),
            top_k=10,
            where={"$and": [{"source_type": "pdf"}, {"doc_id": "doc-a"}]},
            kb_id="kb-a",
        )
        assert and_hits
        assert all(hit["payload"]["doc_id"] == "doc-a" for hit in and_hits)
        or_hits = store.search(
            _vector(0.9),
            top_k=10,
            where={"$or": [{"doc_id": "doc-a"}, {"doc_id": "doc-c"}]},
            kb_id="kb-a",
        )
        assert {hit["payload"]["doc_id"] for hit in or_hits} <= {"doc-a", "doc-c"}
        in_hits = store.search(
            _vector(0.9),
            top_k=10,
            where={"source_type": {"$in": ["pdf", "md"]}},
        )
        assert in_hits
        assert {hit["payload"]["source_type"] for hit in in_hits} <= {"pdf", "md"}
        scalar_hits = store.search(
            _vector(0.9), top_k=10, where={"source_type": "md"}
        )
        assert {hit["payload"]["doc_id"] for hit in scalar_hits} == {"doc-b"}
        eq_hits = store.search(
            _vector(0.9), top_k=10, where={"chunk_index": {"$eq": 1}}
        )
        assert {hit["payload"]["chunk_index"] for hit in eq_hits} == {1}
        assert len(eq_hits) == 2
        ne_hits = store.search(
            _vector(0.9), top_k=10, where={"chunk_index": {"$ne": 1}}
        )
        assert all(hit["payload"]["chunk_index"] != 1 for hit in ne_hits)
        assert len(ne_hits) == 4
        nin_hits = store.search(
            _vector(0.9),
            top_k=10,
            where={"chunk_index": {"$nin": [0, 1]}},
        )
        assert [hit["chunk_id"] for hit in nin_hits] == ["doc-a-2"]
        range_hits = store.search(
            _vector(0.9),
            top_k=10,
            where={"chunk_index": {"$gte": 1, "$lt": 3}},
        )
        assert {hit["payload"]["chunk_index"] for hit in range_hits} == {1, 2}
        assert len(range_hits) == 3

        assert store.find_doc_by_hash("hash-a", kb_id="kb-a") == "doc-a"
        assert store.find_doc_by_hash("hash-a", kb_id="kb-b") is None
        assert store.get_doc_ids("doc-a") == ids_a
        assert store.get_doc_kb_id("doc-a") == "kb-a"
        assert store.get_chunk_by_id("doc-a-1", kb_id="kb-a")["text"] == "child one"
        assert store.get_chunk_by_id("doc-a-1", kb_id="kb-b") is None
        assert [row["chunk_id"] for row in store.get_related_chunks(
            kb_id="kb-a", chunk_id="doc-a-1"
        )] == ids_a
        assert [row["chunk_id"] for row in store.get_chunks_by_parent_id(
            "doc-a-0", kb_id="kb-a"
        )] == ids_a

        assert store.count(kb_id="kb-a") == 4
        assert store.count(kb_id="kb-b") == 2
        assert {row["chunk_id"] for row in store.all_items(kb_id="kb-a")} == set(
            ids_a + ids_c
        )
        docs_a, total_a = store.list_docs(kb_id="kb-a")
        assert total_a == 2
        assert {row["doc_id"] for row in docs_a} == {"doc-a", "doc-c"}
        assert set(store.list_kbs()) >= {"kb-a", "kb-b"}

        legacy_point_id = QdrantVectorStore.point_id_for_chunk("legacy-0")
        legacy_response = client.put(
            f"/collections/{collection}/points",
            params={"wait": "true"},
            json={
                "points": [
                    {
                        "id": legacy_point_id,
                        "vector": _vector(0.2),
                        "payload": {
                            "chunk_id": "legacy-0",
                            "text": "legacy point",
                            "doc_id": "doc-legacy",
                            "doc_title": "legacy",
                            "source_type": "txt",
                            "chunk_index": 0,
                            "content_hash": "hash-legacy",
                        },
                    }
                ]
            },
        )
        assert legacy_response.status_code == 200
        assert store.get_doc_kb_id("doc-legacy") is None
        assert store.migrate_default_kb_id() == 1
        assert store.migrate_default_kb_id() == 0
        assert store.get_doc_kb_id("doc-legacy") == "default"
        assert store.get_chunk_by_id("legacy-0", kb_id="default")["text"] == "legacy point"
        assert store.get_chunk_by_id("legacy-0", kb_id="kb-a") is None
        assert store.count() == 7
        assert store.delete_doc("doc-legacy") == 1
        assert store.count() == 6

        assert store.delete_chunk_ids(["doc-b-0"], kb_id="kb-a") == 0
        assert store.count(kb_id="kb-b") == 2

        snapshot = store.snapshot_doc("doc-a")
        _add_doc(
            store,
            doc_id="doc-a",
            kb_id="kb-a",
            texts=["replacement one", "replacement two", "replacement three", "stale"],
            content_hash="hash-a-new",
        )
        assert len(store.get_doc_ids("doc-a")) == 4
        assert store.restore_doc_snapshot(snapshot) == 3
        assert store.get_doc_ids("doc-a") == ids_a
        assert store.count() == 6

        seen: set[str] = set()
        offset: int | str | None = None
        pages = 0
        offsets: list[Any] = []
        while True:
            body: dict[str, Any] = {"limit": 2, "with_payload": True}
            if offset is not None:
                body["offset"] = offset
            response = client.post(f"/collections/{collection}/points/scroll", json=body)
            assert response.status_code == 200
            result = response.json()["result"]
            points = result["points"]
            pages += 1
            seen.update(str(point["id"]) for point in points)
            next_offset = result["next_page_offset"]
            offsets.append(next_offset)
            if next_offset is None:
                break
            offset = next_offset
            assert pages < 10
        assert pages == 3
        assert len(seen) == 6
        assert offsets[-1] is None

        store.close()
        store = QdrantVectorStore(
            url=url,
            collection_name=collection,
            vector_size=4,
            timeout=10.0,
            create_if_missing=False,
            scroll_page_size=2,
        )
        assert store.count() == 6
        assert store.get_chunk_by_id("doc-a-1", kb_id="kb-a")["text"] == "child one"
        assert store.search(_vector(0.9), top_k=1, kb_id="kb-a")

        assert store.delete_doc("doc-c") == 1
        assert store.delete_kb("kb-b") == 2
        assert store.delete_doc("doc-a") == 3
        assert store.count() == 0
    finally:
        if store is not None:
            store.close()
        delete_response = client.delete(f"/collections/{collection}")
        assert delete_response.status_code in {200, 404}
        assert client.get(f"/collections/{collection}").status_code == 404
        client.close()
