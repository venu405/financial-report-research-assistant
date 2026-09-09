"""Unit tests for the isolated Qdrant VectorStore adapter."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

import services.kb.qdrant_vector_store as qdrant_module
from services.kb.qdrant_vector_store import (
    QdrantVectorStore,
    legacy_point_id_for_chunk,
    point_id_for_chunk,
)
from services.kb.vector_store_contract import VectorStoreBackend


@dataclass
class _FakeState:
    collection_mode: str = "ok"
    collection_size: int = 4
    collection_distance: str = "Cosine"
    requests: list[httpx.Request] = field(default_factory=list)


class _FakeClient:
    """Small in-process HTTP client double; it never opens a socket."""

    def __init__(self, state: _FakeState, *args: Any, **kwargs: Any) -> None:
        self.state = state

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        body = kwargs.get("json")
        if body is None and kwargs.get("content"):
            body = json.loads(kwargs["content"])
        request = httpx.Request(method, str(url), json=body)
        self.state.requests.append(request)
        path = request.url.path
        collection_path = "/collections/unit_collection"

        if path == collection_path and method == "GET":
            if self.state.collection_mode == "missing":
                return httpx.Response(
                    404,
                    json={"status": {"error": "not found"}},
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {
                                    "size": self.state.collection_size,
                                    "distance": self.state.collection_distance,
                                }
                            }
                        }
                    }
                },
                request=request,
            )

        if path == collection_path and method in {"PUT", "POST"}:
            self.state.collection_mode = "ok"
            return httpx.Response(200, json={"result": True}, request=request)

        if path.startswith(collection_path + "/"):
            if method == "DELETE":
                return httpx.Response(200, json={"result": True}, request=request)
            if path.endswith("/points/count"):
                return httpx.Response(200, json={"result": {"count": 0}}, request=request)
            if path.endswith("/points/scroll"):
                return httpx.Response(
                    200,
                    json={"result": {"points": [], "next_page_offset": None}},
                    request=request,
                )
            if "/points/search" in path or "/points/query" in path:
                return httpx.Response(200, json={"result": []}, request=request)
            if path.endswith("/points"):
                return httpx.Response(200, json={"result": []}, request=request)
            if path.endswith("/index") or "/index/" in path:
                return httpx.Response(200, json={"result": True}, request=request)

        return httpx.Response(200, json={"result": True}, request=request)


@dataclass
class _MemoryState:
    points: dict[str, dict[str, Any]] = field(default_factory=dict)


class _MemoryClient:
    """Minimal in-memory Qdrant REST double for scoped point lifecycle tests."""

    def __init__(self, state: _MemoryState) -> None:
        self.state = state

    def close(self) -> None:
        return None

    @staticmethod
    def _matches(point: dict[str, Any], clause: dict[str, Any] | None) -> bool:
        if not clause:
            return True
        if "must" in clause and not all(
            _MemoryClient._matches(point, item) for item in clause["must"]
        ):
            return False
        if "must_not" in clause and any(
            _MemoryClient._matches(point, item) for item in clause["must_not"]
        ):
            return False
        if "should" in clause and not any(
            _MemoryClient._matches(point, item) for item in clause["should"]
        ):
            return False
        if "key" in clause:
            payload = point.get("payload", {})
            value = payload.get(clause["key"])
            match = clause.get("match", {})
            if "value" in match and value != match["value"]:
                return False
            if "any" in match and value not in match["any"]:
                return False
            if "range" in clause:
                for operator, operand in clause["range"].items():
                    if operator == "gte" and not value >= operand:
                        return False
                    if operator == "gt" and not value > operand:
                        return False
                    if operator == "lte" and not value <= operand:
                        return False
                    if operator == "lt" and not value < operand:
                        return False
        return True

    @staticmethod
    def _response(
        request: httpx.Request, result: Any, status_code: int = 200
    ) -> httpx.Response:
        return httpx.Response(status_code, json={"result": result}, request=request)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        body = kwargs.get("json") or {}
        request = httpx.Request(method, str(url), json=body)
        path = request.url.path
        collection_path = "/collections/unit_collection"

        if path == collection_path and method == "GET":
            return self._response(
                request,
                {
                    "config": {
                        "params": {
                            "vectors": {"size": 4, "distance": "Cosine"}
                        }
                    }
                },
            )
        if path == collection_path and method in {"PUT", "POST"}:
            return self._response(request, True)
        if path.endswith("/index"):
            return self._response(request, True)
        if path.endswith("/points/count"):
            points = [
                point
                for point in self.state.points.values()
                if self._matches(point, body.get("filter"))
            ]
            return self._response(request, {"count": len(points)})
        if path.endswith("/points/scroll"):
            points = [
                point
                for point in self.state.points.values()
                if self._matches(point, body.get("filter"))
            ]
            points.sort(key=lambda point: str(point["id"]))
            start = int(body.get("offset") or 0)
            limit = int(body.get("limit") or len(points))
            page = points[start : start + limit]
            next_offset = start + limit if start + limit < len(points) else None
            return self._response(
                request,
                {
                    "points": [
                        self._project(point, with_vector=body.get("with_vector", False))
                        for point in page
                    ],
                    "next_page_offset": next_offset,
                },
            )
        if path.endswith("/points/search") or path.endswith("/points/query"):
            points = [
                point
                for point in self.state.points.values()
                if self._matches(point, body.get("filter"))
            ]
            points.sort(key=lambda point: str(point["id"]))
            limit = int(body.get("limit") or len(points))
            return self._response(
                request,
                [
                    {
                        **self._project(point, with_vector=False),
                        "score": 1.0,
                    }
                    for point in points[:limit]
                ],
            )
        if path.endswith("/points/delete"):
            for point_id in body.get("points", []):
                self.state.points.pop(str(point_id), None)
            return self._response(request, True)
        if path.endswith("/points") and method == "PUT":
            for point in body.get("points", []):
                self.state.points[str(point["id"])] = point
            return self._response(request, True)
        if path.endswith("/points") and method == "POST":
            points = [
                self.state.points[str(point_id)]
                for point_id in body.get("ids", [])
                if str(point_id) in self.state.points
            ]
            return self._response(
                request,
                [
                    self._project(point, with_vector=body.get("with_vector", False))
                    for point in points
                ],
            )
        if path == collection_path and method == "DELETE":
            self.state.points.clear()
            return self._response(request, True)
        return self._response(request, True)

    @staticmethod
    def _project(point: dict[str, Any], *, with_vector: bool) -> dict[str, Any]:
        result = {"id": point["id"], "payload": point["payload"]}
        if with_vector:
            result["vector"] = point["vector"]
        return result


@pytest.fixture
def memory_qdrant() -> tuple[_MemoryState, _MemoryClient]:
    state = _MemoryState()
    return state, _MemoryClient(state)


def _memory_store(client: _MemoryClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        url="http://qdrant.test",
        collection_name="unit_collection",
        vector_size=4,
        client=client,
        scroll_page_size=2,
    )


def _add_memory_chunk(
    store: QdrantVectorStore, *, kb_id: str, text: str
) -> list[str]:
    return store.add_chunks(
        embeddings=[[0.0, 0.0, 0.0, 1.0]],
        texts=[text],
        doc_id="same-doc",
        doc_title="Same document",
        source_type="synthetic",
        chunk_indices=[0],
        kb_id=kb_id,
    )


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> _FakeState:
    state = _FakeState()
    client = _FakeClient(state)
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: client)
    if hasattr(qdrant_module, "Client"):
        monkeypatch.setattr(qdrant_module, "Client", lambda *args, **kwargs: client)
    return state


def _store(state: _FakeState, **kwargs: Any) -> QdrantVectorStore:
    return QdrantVectorStore(
        url="http://qdrant.test",
        collection_name="unit_collection",
        vector_size=4,
        **kwargs,
    )


def _last_filter(state: _FakeState) -> dict[str, Any]:
    for request in reversed(state.requests):
        body = json.loads(request.content or b"{}")
        if "filter" in body:
            return body["filter"]
    raise AssertionError("the adapter did not send a Qdrant filter")


def _sorted_json(items: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(item, sort_keys=True) for item in items)


def test_point_id_is_stable_distinct_and_uuid5(fake_qdrant: _FakeState) -> None:
    _store(fake_qdrant)

    first = point_id_for_chunk("doc-a-0", "kb-a")
    same = point_id_for_chunk("doc-a-0", "kb-a")
    other = point_id_for_chunk("doc-b-0", "kb-a")
    same_chunk_other_kb = point_id_for_chunk("doc-a-0", "kb-b")

    assert first == same
    assert first != other
    assert first != same_chunk_other_kb
    parsed = uuid.UUID(first)
    assert parsed.version == 5
    assert str(parsed) == first


def test_same_chunk_id_isolated_across_kbs_for_lifecycle(
    memory_qdrant: tuple[_MemoryState, _MemoryClient],
) -> None:
    state, client = memory_qdrant
    store = _memory_store(client)
    logical_id = "same-doc-0"

    assert _add_memory_chunk(store, kb_id="kb-a", text="a-v1") == [logical_id]
    assert _add_memory_chunk(store, kb_id="kb-b", text="b-v1") == [logical_id]
    physical_a = point_id_for_chunk(logical_id, "kb-a")
    physical_b = point_id_for_chunk(logical_id, "kb-b")
    assert physical_a != physical_b
    assert set(state.points) == {physical_a, physical_b}
    assert store.count() == 2
    assert store.count(kb_id="kb-a") == 1
    assert store.count(kb_id="kb-b") == 1

    record_a = store.get_chunk_by_id(logical_id, kb_id="kb-a")
    record_b = store.get_chunk_by_id(logical_id, kb_id="kb-b")
    assert record_a is not None and record_a["text"] == "a-v1"
    assert record_b is not None and record_b["text"] == "b-v1"
    assert store.get_chunk_by_id(logical_id, kb_id="kb-missing") is None
    assert store.search(
        [0.0, 0.0, 0.0, 1.0], top_k=1, kb_id="kb-a"
    )[0]["payload"]["kb_id"] == "kb-a"
    assert store.search(
        [0.0, 0.0, 0.0, 1.0], top_k=1, kb_id="kb-b"
    )[0]["payload"]["kb_id"] == "kb-b"

    # Repeating each scoped upsert updates only its own physical point.
    assert _add_memory_chunk(store, kb_id="kb-a", text="a-v2") == [logical_id]
    assert _add_memory_chunk(store, kb_id="kb-b", text="b-v2") == [logical_id]
    assert store.count() == 2
    assert store.get_chunk_by_id(logical_id, kb_id="kb-a")["text"] == "a-v2"
    assert store.get_chunk_by_id(logical_id, kb_id="kb-b")["text"] == "b-v2"

    # A wrong scope is a no-op; each correct scope deletes its own point.
    assert store.delete_chunk_ids([logical_id], kb_id="kb-missing") == 0
    assert store.count() == 2
    assert store.delete_chunk_ids([logical_id], kb_id="kb-a") == 1
    assert store.count() == 1
    assert store.get_chunk_by_id(logical_id, kb_id="kb-a") is None
    assert store.get_chunk_by_id(logical_id, kb_id="kb-b") is not None
    assert store.delete_chunk_ids([logical_id], kb_id="kb-b") == 1
    assert store.count() == 0


def test_legacy_point_migrates_to_scoped_default_id(
    memory_qdrant: tuple[_MemoryState, _MemoryClient],
) -> None:
    state, client = memory_qdrant
    store = _memory_store(client)
    logical_id = "legacy-doc-0"
    legacy_id = legacy_point_id_for_chunk(logical_id)
    state.points[legacy_id] = {
        "id": legacy_id,
        "vector": [0.0, 0.0, 0.0, 1.0],
        "payload": {
            "chunk_id": logical_id,
            "text": "legacy",
            "doc_id": "legacy-doc",
            "doc_title": "Legacy",
            "source_type": "synthetic",
            "chunk_index": 0,
            "content_hash": "legacy-hash",
        },
    }

    assert store.get_chunk_by_id(logical_id, kb_id="kb-a") is None
    assert store.get_chunk_by_id(logical_id, kb_id="default")["text"] == "legacy"
    assert store.count(kb_id="default") == 1
    assert store.migrate_default_kb_id() == 1
    scoped_id = point_id_for_chunk(logical_id, "default")
    assert scoped_id in state.points
    assert legacy_id not in state.points
    assert store.get_chunk_by_id(logical_id, kb_id="default")["text"] == "legacy"
    assert store.get_chunk_by_id(logical_id, kb_id="kb-a") is None
    assert store.migrate_default_kb_id() == 0
    assert store.delete_kb("default") == 1
    assert store.count() == 0


def test_adapter_satisfies_backend_protocol(fake_qdrant: _FakeState) -> None:
    assert isinstance(_store(fake_qdrant), VectorStoreBackend)


def test_missing_collection_is_explicit_unless_create_if_missing(
    fake_qdrant: _FakeState,
) -> None:
    fake_qdrant.collection_mode = "missing"

    with pytest.raises(
        (RuntimeError, ValueError, httpx.HTTPError),
        match="collection|Collection|not found",
    ):
        _store(fake_qdrant)

    fake_qdrant.collection_mode = "missing"
    _store(fake_qdrant, create_if_missing=True)
    assert any(
        request.method in {"PUT", "POST"}
        and request.url.path == "/collections/unit_collection"
        for request in fake_qdrant.requests
    )


@pytest.mark.parametrize(
    ("size", "distance", "message"),
    [
        (8, "Cosine", "dimension|size|vector"),
        (4, "Dot", "distance|Cosine"),
    ],
)
def test_collection_configuration_mismatch_fails(
    fake_qdrant: _FakeState,
    size: int,
    distance: str,
    message: str,
) -> None:
    fake_qdrant.collection_size = size
    fake_qdrant.collection_distance = distance

    with pytest.raises(ValueError, match=message):
        _store(fake_qdrant)


def test_where_translates_scalar_equality_and_kb_id_and(fake_qdrant: _FakeState) -> None:
    store = _store(fake_qdrant)

    store.search([0.0, 0.0, 0.0, 1.0], where={"source_type": "pdf"}, kb_id="kb-a")

    expected = [
        {"key": "source_type", "match": {"value": "pdf"}},
        {"key": "kb_id", "match": {"value": "kb-a"}},
    ]
    assert _sorted_json(_last_filter(fake_qdrant)["must"]) == _sorted_json(expected)


def test_where_translates_and_or_and_in(fake_qdrant: _FakeState) -> None:
    store = _store(fake_qdrant)

    store.search(
        [0.0, 0.0, 0.0, 1.0],
        where={"$and": [{"source_type": "pdf"}, {"doc_id": "doc-a"}]},
    )
    first_filter = _last_filter(fake_qdrant)
    assert first_filter == {
        "must": [
            {
                "must": [{"key": "source_type", "match": {"value": "pdf"}}]
            },
            {
                "must": [{"key": "doc_id", "match": {"value": "doc-a"}}]
            },
        ]
    }

    store.search(
        [0.0, 0.0, 0.0, 1.0],
        where={"$or": [{"source_type": "pdf"}, {"source_type": "md"}]},
    )
    second_filter = _last_filter(fake_qdrant)
    assert second_filter == {
        "should": [
            {
                "must": [{"key": "source_type", "match": {"value": "pdf"}}]
            },
            {
                "must": [{"key": "source_type", "match": {"value": "md"}}]
            },
        ]
    }

    store.search(
        [0.0, 0.0, 0.0, 1.0],
        where={"source_type": {"$in": ["pdf", "md"]}},
    )
    assert _last_filter(fake_qdrant) == {
        "must": [{"key": "source_type", "match": {"any": ["pdf", "md"]}}]
    }


def test_where_rejects_unsupported_operator(fake_qdrant: _FakeState) -> None:
    store = _store(fake_qdrant)

    with pytest.raises(ValueError, match=r"\$contains|unsupported|不支持"):
        store.search(
            [0.0, 0.0, 0.0, 1.0],
            where={"source_type": {"$contains": "pdf"}},
        )


def test_input_lengths_and_empty_snapshot_fail_before_network(
    fake_qdrant: _FakeState,
) -> None:
    store = _store(fake_qdrant)
    request_count = len(fake_qdrant.requests)

    with pytest.raises(ValueError, match="length|数量|一致"):
        store.add_chunks(
            embeddings=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
            texts=["one"],
            doc_id="doc-a",
            doc_title="Doc A",
            source_type="pdf",
            chunk_indices=[0, 1],
        )
    with pytest.raises(ValueError, match="dimension|维度|长度"):
        store.search([0.0, 0.0, 1.0], top_k=1)
    with pytest.raises(ValueError, match="snapshot|快照|empty|为空"):
        store.restore_doc_snapshot({})

    assert len(fake_qdrant.requests) == request_count


def test_extra_metadata_requires_one_entry_per_chunk(fake_qdrant: _FakeState) -> None:
    store = _store(fake_qdrant)

    with pytest.raises(ValueError, match="extra_metadata|数量|一致"):
        store.add_chunks(
            embeddings=[[0.0, 0.0, 0.0, 1.0]],
            texts=["one"],
            doc_id="doc-a",
            doc_title="Doc A",
            source_type="pdf",
            chunk_indices=[0],
            extra_metadata=[],
        )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"chunk_id": "chunk-a", "parent_id": "parent-a"}],
)
def test_related_requires_exactly_one_selector(
    fake_qdrant: _FakeState, kwargs: dict[str, str]
) -> None:
    store = _store(fake_qdrant)

    with pytest.raises(ValueError, match="exactly one|one|二选一|只能"):
        store.get_related_chunks(kb_id="kb-a", **kwargs)


def test_related_accepts_one_selector(fake_qdrant: _FakeState) -> None:
    store = _store(fake_qdrant)

    assert store.get_related_chunks(kb_id="kb-a", chunk_id="chunk-a") == []
