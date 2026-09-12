"""T4 isolated Qdrant smoke test.

The test uses one uniquely named temporary collection and synthetic 1024
dimensional vectors only.  It never imports the application factory, touches
Chroma, reads the 185 source files, or uses the default collection.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from typing import Any

import httpx
import pytest

from services.kb.qdrant_vector_store import QdrantVectorStore, point_id_for_chunk

QDRANT_URL = "http://127.0.0.1:16333"
T1_CONTAINER = "kb-qdrant-preflight-20260828-141124-ca47dc59"
T1_PROJECT = T1_CONTAINER
T1_VOLUME = "kb_qdrant_preflight_data_20260828-141124-ca47dc59"
T1_COLLECTION = "qdrant_preflight_20260828_141124_ca47dc59"
T4_PREFIX = "kb_t4_1200_smoke_"
VECTOR_SIZE = 1_024
POINTS = 1_200
PAGE_SIZE = 37


def _allow_explicit_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the suite network guard for this one explicit live test."""

    def live_request(
        client: httpx.Client,
        method: str,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
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


def _vector(index: int) -> list[float]:
    """Return a deterministic non-zero vector with exactly 1024 dimensions."""

    values = [0.0] * VECTOR_SIZE
    phase = ((index * 37) % 1_201) / 1_200.0
    values[0] = 0.25 + phase
    values[1] = 1.0 - phase * 0.5
    values[2] = 0.5
    values[3] = 0.125
    values[4 + (index % (VECTOR_SIZE - 4))] = 0.01
    assert len(values) == VECTOR_SIZE
    return values


def _doc_rows(
    *, doc_id: str, kb_id: str, start: int, count: int
) -> tuple[list[list[float]], list[str], list[dict[str, object]]]:
    embeddings = [_vector(start + index) for index in range(count)]
    texts = [f"synthetic t4 {kb_id} {doc_id} chunk {index}" for index in range(count)]
    metadata = [
        {
            "page": index // 25 + 1,
            "page_start": index // 25 + 1,
            "page_end": index // 25 + 1,
            "section_path": ["T4", kb_id, doc_id],
            "is_table": index % 10 == 0,
            "synthetic_t4": True,
        }
        for index in range(count)
    ]
    return embeddings, texts, metadata


def _json_result(response: httpx.Response) -> Any:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("status") in {"ok", None}, body
    assert "result" in body, body
    return body["result"]


def _collection_names(client: httpx.Client) -> list[str]:
    collections = _json_result(client.get("/collections")).get("collections", [])
    assert isinstance(collections, list)
    names = [str(item["name"]) for item in collections]
    assert len(names) == len(set(names)), names
    return sorted(names)


def _collection_info(client: httpx.Client, name: str) -> dict[str, Any]:
    info = _json_result(client.get(f"/collections/{name}"))
    assert isinstance(info, dict)
    return info


def _wait_ready(client: httpx.Client, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = client.get("/readyz")
            if response.status_code == 200:
                return
            last_error = response.text
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise AssertionError(f"Qdrant did not become ready: {last_error}")


def _inspect_t1_container() -> dict[str, Any]:
    """Inspect and validate only the authorized T1 container."""

    result = subprocess.run(
        ["docker", "inspect", T1_CONTAINER],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr.strip()
    records = json.loads(result.stdout)
    assert isinstance(records, list) and len(records) == 1, records
    record = records[0]
    labels = record["Config"]["Labels"]
    ports = record["NetworkSettings"]["Ports"]
    mounts = record["Mounts"]

    assert record["Name"] == f"/{T1_CONTAINER}"
    assert record["Config"]["Image"] == "qdrant/qdrant:v1.19.0"
    assert labels["com.docker.compose.project"] == T1_PROJECT
    assert labels["com.docker.compose.service"] == "qdrant"
    assert record["State"]["Status"] == "running"
    assert record["State"]["Health"]["Status"] == "healthy"
    assert ports == {
        "6333/tcp": [{"HostIp": "127.0.0.1", "HostPort": "16333"}],
        "6334/tcp": [{"HostIp": "127.0.0.1", "HostPort": "16334"}],
    }
    assert len(mounts) == 3, mounts
    assert sum(
        mount["Type"] == "volume"
        and mount["Name"] == T1_VOLUME
        and mount["Destination"] == "/qdrant/storage"
        for mount in mounts
    ) == 1, mounts
    assert {
        (mount["Type"], mount["Destination"])
        for mount in mounts
    } == {
        ("volume", "/qdrant/storage"),
        ("bind", "/qdrant/logs"),
        ("bind", "/qdrant/snapshots"),
    }, mounts

    return {
        "id": record["Id"],
        "name": record["Name"],
        "image": record["Config"]["Image"],
        "image_id": record["Image"],
        "project": labels["com.docker.compose.project"],
        "service": labels["com.docker.compose.service"],
        "status": record["State"]["Status"],
        "health": record["State"]["Health"]["Status"],
        "ports": ports,
        "mounts": sorted(
            (
                {
                    "type": mount["Type"],
                    "name": mount.get("Name"),
                    "destination": mount["Destination"],
                    "rw": mount["RW"],
                }
                for mount in mounts
            ),
            key=lambda item: item["destination"],
        ),
    }


def _t1_filter(kb_id: str) -> dict[str, Any]:
    return {"must": [{"key": "kb_id", "match": {"value": kb_id}}]}


def _t1_state(client: httpx.Client) -> dict[str, Any]:
    """Read T1 count/get/search/filter state without modifying it."""

    info = _collection_info(client, T1_COLLECTION)
    vectors = info["config"]["params"]["vectors"]
    assert vectors == {"size": VECTOR_SIZE, "distance": "Cosine"}
    assert info["status"] == "green"
    assert info["optimizer_status"] == "ok"
    assert info["points_count"] == 100

    total_count = _json_result(
        client.post(
            f"/collections/{T1_COLLECTION}/points/count",
            json={"exact": True},
        )
    )["count"]
    assert total_count == 100

    sample_result = _json_result(
        client.post(
            f"/collections/{T1_COLLECTION}/points/scroll",
            json={"limit": 1, "with_payload": True, "with_vector": True},
        )
    )
    sample_points = sample_result["points"]
    assert len(sample_points) == 1
    sample = sample_points[0]
    sample_id = sample["id"]
    sample_payload = sample["payload"]
    sample_vector = sample["vector"]
    assert isinstance(sample_payload, dict)
    assert sample_payload["kb_id"] in {"synthetic-kb-a", "synthetic-kb-b"}
    assert isinstance(sample_vector, list) and len(sample_vector) == VECTOR_SIZE

    get_points = _json_result(
        client.post(
            f"/collections/{T1_COLLECTION}/points",
            json={"ids": [sample_id], "with_payload": True, "with_vector": False},
        )
    )
    assert len(get_points) == 1
    assert get_points[0]["id"] == sample_id
    assert get_points[0]["payload"] == sample_payload

    filter_counts: dict[str, int] = {}
    filtered_search_ids: dict[str, list[str]] = {}
    for kb_id in ("synthetic-kb-a", "synthetic-kb-b"):
        filter_body = _t1_filter(kb_id)
        filtered_count = _json_result(
            client.post(
                f"/collections/{T1_COLLECTION}/points/count",
                json={"exact": True, "filter": filter_body},
            )
        )["count"]
        assert filtered_count == 50
        filter_counts[kb_id] = filtered_count
        search_points = _json_result(
            client.post(
                f"/collections/{T1_COLLECTION}/points/search",
                json={
                    "vector": sample_vector,
                    "limit": 10,
                    "with_payload": True,
                    "with_vector": False,
                    "filter": filter_body,
                },
            )
        )
        assert search_points
        assert all(point["payload"]["kb_id"] == kb_id for point in search_points)
        filtered_search_ids[kb_id] = [str(point["id"]) for point in search_points]

    return {
        "collection": T1_COLLECTION,
        "status": info["status"],
        "optimizer_status": info["optimizer_status"],
        "points_count": info["points_count"],
        "vectors": vectors,
        "sample_id": str(sample_id),
        "sample_payload": sample_payload,
        "sample_vector": sample_vector,
        "get_payload": get_points[0]["payload"],
        "filter_counts": filter_counts,
        "filtered_search_ids": filtered_search_ids,
    }


def _restart_t1_container(before: dict[str, Any]) -> dict[str, Any]:
    """Restart exactly the previously validated T1 container."""

    current = _inspect_t1_container()
    for key in ("id", "image", "image_id", "project", "service", "ports", "mounts"):
        assert current[key] == before[key], (key, before[key], current[key])
    restarted = subprocess.run(
        ["docker", "restart", T1_CONTAINER],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert restarted.returncode == 0, restarted.stderr.strip()
    deadline = time.monotonic() + 60.0
    last_state = ""
    while time.monotonic() < deadline:
        health = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}|{{.State.Health.Status}}",
                T1_CONTAINER,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert health.returncode == 0, health.stderr.strip()
        last_state = health.stdout.strip()
        if last_state == "running|healthy":
            return _inspect_t1_container()
        time.sleep(0.25)
    raise AssertionError(f"T1 container did not become healthy after restart: {last_state}")


def _print_host_evidence(
    client: httpx.Client, collection: str, label: str
) -> None:
    """Print failure evidence before cleanup; pytest preserves this output."""

    evidence: dict[str, Any] = {"label": label, "collection": collection}
    try:
        response = client.get(f"/collections/{collection}")
        evidence["collection_status_code"] = response.status_code
        if response.status_code == 200:
            info = response.json().get("result", {})
            evidence["collection_points_count"] = info.get("points_count")
            evidence["collection_status"] = info.get("status")
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        evidence["collection_error"] = type(exc).__name__
    try:
        evidence["collections"] = _collection_names(client)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        evidence["collections_error"] = type(exc).__name__
    try:
        evidence["container"] = _inspect_t1_container()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        evidence["container_error"] = type(exc).__name__
    print(f"T4_HOST_EVIDENCE={json.dumps(evidence, sort_keys=True)}")


def _cleanup_collection(client: httpx.Client, collection: str) -> None:
    """Delete only the known T4 collection and prove it is absent."""

    last_status: int | None = None
    for attempt in range(2):
        response = client.delete(f"/collections/{collection}")
        last_status = response.status_code
        if response.status_code in {200, 404}:
            break
        if attempt == 0:
            time.sleep(0.25)
    assert last_status in {200, 404}, f"T4 cleanup returned HTTP {last_status}"
    absent = client.get(f"/collections/{collection}")
    assert absent.status_code == 404, absent.text
    remaining_t4 = [name for name in _collection_names(client) if name.startswith(T4_PREFIX)]
    assert remaining_t4 == [], remaining_t4
    print(
        "T4_CLEANUP_EVIDENCE="
        + json.dumps(
            {
                "collection": collection,
                "delete_status": last_status,
                "collection_status_after_delete": absent.status_code,
                "remaining_t4_collections": remaining_t4,
            },
            sort_keys=True,
        )
    )


def test_t4_isolated_1200_point_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify T4's 1200-point write, contract, restart, delete, and cleanup gates."""

    if os.getenv("KB_QDRANT_T4_RUN") != "1":
        pytest.skip(
            "T4 live gate requires KB_QDRANT_T4_RUN=1; "
            "跳过实时 Qdrant 冒烟（CI 与默认环境无受控 Qdrant 服务）"
        )
    url = os.getenv("KB_QDRANT_TEST_URL", QDRANT_URL).rstrip("/")
    assert url == QDRANT_URL, f"T4 is authorized only for {QDRANT_URL}, got {url}"
    _allow_explicit_live_http(monkeypatch)

    collection = f"{T4_PREFIX}{uuid.uuid4().hex}"
    client = httpx.Client(base_url=url, timeout=30.0)
    store: QdrantVectorStore | None = None
    main_failed = False
    container_before: dict[str, Any] | None = None
    collection_names_before: list[str] = []

    # 600 + 200 points in kb-a, 300 + 100 points in kb-b = exactly 1,200.
    docs = (
        ("t4-a1", "t4-kb-a", 0, 600),
        ("t4-a2", "t4-kb-a", 600, 200),
        ("t4-b1", "t4-kb-b", 800, 300),
        ("t4-b2", "t4-kb-b", 1100, 100),
    )

    try:
        _wait_ready(client)
        container_before = _inspect_t1_container()
        t1_before = _t1_state(client)
        collection_names_before = _collection_names(client)
        assert T1_COLLECTION in collection_names_before
        assert collection not in collection_names_before
        assert not [
            name for name in collection_names_before if name.startswith(T4_PREFIX)
        ], collection_names_before
        print(
            "T4_PRECHECK="
            + json.dumps(
                {
                    "container": container_before,
                    "collections": collection_names_before,
                    "t1": {
                        key: value
                        for key, value in t1_before.items()
                        if key != "sample_vector"
                    },
                },
                sort_keys=True,
            )
        )

        store = QdrantVectorStore(
            url=url,
            collection_name=collection,
            vector_size=VECTOR_SIZE,
            timeout=30.0,
            create_if_missing=True,
            scroll_page_size=PAGE_SIZE,
        )
        info = _collection_info(client, collection)
        assert info["config"]["params"]["vectors"] == {
            "size": VECTOR_SIZE,
            "distance": "Cosine",
        }
        required_indexes = {
            "kb_id",
            "doc_id",
            "chunk_id",
            "content_hash",
            "parent_id",
            "table_id",
            "page",
            "page_start",
            "page_end",
            "chunk_index",
            "is_table",
        }
        assert required_indexes <= set(info.get("payload_schema", {}))

        expected_logical_ids: dict[str, list[str]] = {}
        expected_physical_ids: set[str] = set()
        batches: dict[str, tuple[list[list[float]], list[str], list[dict[str, object]]]] = {}
        for doc_id, kb_id, start, count in docs:
            rows = _doc_rows(doc_id=doc_id, kb_id=kb_id, start=start, count=count)
            batches[doc_id] = rows
            embeddings, texts, metadata = rows
            logical_ids = store.add_chunks(
                embeddings=embeddings,
                texts=texts,
                doc_id=doc_id,
                doc_title=f"T4 {doc_id}",
                source_type="synthetic",
                chunk_indices=list(range(count)),
                kb_id=kb_id,
                content_hash=f"t4-hash-{doc_id}",
                extra_metadata=metadata,
            )
            expected = [f"{doc_id}-{index}" for index in range(count)]
            assert logical_ids == expected
            expected_logical_ids[doc_id] = expected
            expected_physical_ids.update(
                point_id_for_chunk(chunk_id, kb_id) for chunk_id in expected
            )

        assert sum(map(len, expected_logical_ids.values())) == POINTS
        assert len(expected_physical_ids) == POINTS
        assert store.count() == POINTS
        assert store.count(kb_id="t4-kb-a") == 800
        assert store.count(kb_id="t4-kb-b") == 400

        # Repeat every write batch. Stable logical IDs make this an idempotent upsert.
        for doc_id, kb_id, _start, count in docs:
            embeddings, texts, metadata = batches[doc_id]
            assert store.add_chunks(
                embeddings=embeddings,
                texts=texts,
                doc_id=doc_id,
                doc_title=f"T4 {doc_id}",
                source_type="synthetic",
                chunk_indices=list(range(count)),
                kb_id=kb_id,
                content_hash=f"t4-hash-{doc_id}",
                extra_metadata=metadata,
            ) == expected_logical_ids[doc_id]
        assert store.count() == POINTS

        # Canonical result fields, score semantics, and deterministic physical IDs.
        exact_hit = store.search(_vector(17), top_k=5, kb_id="t4-kb-a")[0]
        assert exact_hit["chunk_id"] == "t4-a1-17"
        assert exact_hit["id"] == point_id_for_chunk(
            exact_hit["chunk_id"], exact_hit["payload"]["kb_id"]
        )
        assert exact_hit["payload"]["chunk_id"] == exact_hit["chunk_id"]
        assert exact_hit["payload"]["text"] == exact_hit["text"]
        assert exact_hit["metadata"]["doc_id"] == "t4-a1"
        assert isinstance(exact_hit["score"], float)
        assert -1.0 <= exact_hit["score"] <= 1.000001
        assert exact_hit["score"] == pytest.approx(1.0, abs=1e-6)
        assert exact_hit["distance"] == pytest.approx(
            1.0 - exact_hit["score"], abs=1e-6
        )
        assert {
            "id",
            "payload",
            "score",
            "distance",
            "chunk_id",
            "text",
            "metadata",
        } <= exact_hit.keys()

        # kb_id is an enforced AND scope, including when an inner where is present.
        assert not store.search(
            _vector(17),
            top_k=POINTS,
            where={"doc_id": "t4-a1"},
            kb_id="t4-kb-b",
        )
        scalar_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"doc_id": "t4-b2"},
            kb_id="t4-kb-b",
        )
        assert len(scalar_hits) == 100
        assert {hit["payload"]["kb_id"] for hit in scalar_hits} == {"t4-kb-b"}
        assert {hit["payload"]["doc_id"] for hit in scalar_hits} == {"t4-b2"}

        and_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"$and": [{"doc_id": "t4-a1"}, {"page": {"$gte": 10}}]},
            kb_id="t4-kb-a",
        )
        assert len(and_hits) == 375
        assert all(
            hit["payload"]["doc_id"] == "t4-a1"
            and hit["payload"]["page"] >= 10
            and hit["payload"]["kb_id"] == "t4-kb-a"
            for hit in and_hits
        )

        or_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"$or": [{"doc_id": "t4-a1"}, {"doc_id": "t4-a2"}]},
            kb_id="t4-kb-a",
        )
        assert len(or_hits) == 800
        assert {hit["payload"]["doc_id"] for hit in or_hits} == {"t4-a1", "t4-a2"}
        assert {hit["payload"]["kb_id"] for hit in or_hits} == {"t4-kb-a"}

        in_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"doc_id": {"$in": ["t4-a1", "t4-b2"]}},
        )
        assert len(in_hits) == 700
        assert {hit["payload"]["doc_id"] for hit in in_hits} == {"t4-a1", "t4-b2"}

        range_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"chunk_index": {"$gte": 595, "$lt": 600}},
            kb_id="t4-kb-a",
        )
        assert len(range_hits) == 5
        assert {hit["payload"]["chunk_index"] for hit in range_hits} == {
            595,
            596,
            597,
            598,
            599,
        }
        assert {hit["payload"]["doc_id"] for hit in range_hits} == {"t4-a1"}

        # Small-page complete scroll: no duplicate physical or logical IDs and null terminal offset.
        seen_physical: list[str] = []
        seen_logical: list[str] = []
        seen_scoped: list[tuple[str, str]] = []
        offsets: list[Any] = []
        offset: Any = None
        page_lengths: list[int] = []
        while True:
            body: dict[str, Any] = {
                "limit": PAGE_SIZE,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            result = _json_result(
                client.post(f"/collections/{collection}/points/scroll", json=body)
            )
            page = result["points"]
            assert isinstance(page, list) and len(page) <= PAGE_SIZE
            page_physical = [str(point["id"]) for point in page]
            page_logical = [str(point["payload"]["chunk_id"]) for point in page]
            page_scoped = [
                (str(point["payload"]["kb_id"]), str(point["payload"]["chunk_id"]))
                for point in page
            ]
            assert len(page_physical) == len(set(page_physical))
            assert len(page_logical) == len(set(page_logical))
            assert len(page_scoped) == len(set(page_scoped))
            seen_physical.extend(page_physical)
            seen_logical.extend(page_logical)
            seen_scoped.extend(page_scoped)
            page_lengths.append(len(page))
            next_offset = result["next_page_offset"]
            offsets.append(next_offset)
            if next_offset is None:
                break
            assert next_offset != offset
            offset = next_offset
            assert len(page_lengths) <= (POINTS + PAGE_SIZE - 1) // PAGE_SIZE
        assert sum(page_lengths) == POINTS
        assert len(seen_physical) == POINTS
        assert len(set(seen_physical)) == POINTS
        assert len(seen_logical) == POINTS
        assert len(set(seen_logical)) == POINTS
        assert len(seen_scoped) == POINTS
        assert len(set(seen_scoped)) == POINTS
        assert set(seen_physical) == expected_physical_ids
        assert set(seen_logical) == {
            chunk_id for ids in expected_logical_ids.values() for chunk_id in ids
        }
        assert set(seen_physical) == {
            point_id_for_chunk(chunk_id, kb_id)
            for kb_id, chunk_id in seen_scoped
        }
        assert len(page_lengths) == (POINTS + PAGE_SIZE - 1) // PAGE_SIZE
        assert page_lengths[:-1] == [PAGE_SIZE] * (len(page_lengths) - 1)
        assert page_lengths[-1] == POINTS % PAGE_SIZE
        assert offsets[-1] is None

        # The adapter's point lookup must use the logical chunk ID while Qdrant stores UUID IDs.
        sample_logical = expected_logical_ids["t4-a1"][17]
        sample_record = store.get_chunk_by_id(sample_logical, kb_id="t4-kb-a")
        assert sample_record is not None
        assert sample_record["id"] == point_id_for_chunk(
            sample_logical, sample_record["payload"]["kb_id"]
        )
        assert sample_record["payload"]["chunk_id"] == sample_logical
        assert store.get_chunk_by_id(sample_logical, kb_id="t4-kb-b") is None

        # Read the same T4 data and the untouched T1 data after restarting the validated container.
        assert container_before is not None
        store.close()
        store = None
        container_after = _restart_t1_container(container_before)
        for key in ("id", "image", "image_id", "project", "service", "ports", "mounts"):
            assert container_after[key] == container_before[key], (
                key,
                container_before[key],
                container_after[key],
            )
        _wait_ready(client)
        t1_after = _t1_state(client)
        assert t1_after == t1_before
        assert _collection_names(client) == sorted(collection_names_before + [collection])

        store = QdrantVectorStore(
            url=url,
            collection_name=collection,
            vector_size=VECTOR_SIZE,
            timeout=30.0,
            create_if_missing=False,
            scroll_page_size=PAGE_SIZE,
        )
        assert store.count() == POINTS
        assert store.count(kb_id="t4-kb-a") == 800
        assert store.count(kb_id="t4-kb-b") == 400
        post_restart_record = store.get_chunk_by_id(sample_logical, kb_id="t4-kb-a")
        assert post_restart_record is not None
        assert post_restart_record["text"].endswith("chunk 17")
        post_restart_hits = store.search(_vector(17), top_k=5, kb_id="t4-kb-a")
        assert post_restart_hits and post_restart_hits[0]["chunk_id"] == sample_logical
        assert all(hit["payload"]["kb_id"] == "t4-kb-a" for hit in post_restart_hits)
        post_restart_filter_hits = store.search(
            _vector(17),
            top_k=POINTS,
            where={"doc_id": "t4-a1"},
            kb_id="t4-kb-a",
        )
        assert len(post_restart_filter_hits) == 600
        assert all(hit["payload"]["kb_id"] == "t4-kb-a" for hit in post_restart_filter_hits)

        # Wrong scope is a no-op; correct scope deletes exactly the requested chunk.
        assert store.delete_chunk_ids([sample_logical], kb_id="t4-kb-b") == 0
        assert store.count() == POINTS
        assert store.get_chunk_by_id(sample_logical, kb_id="t4-kb-a") is not None
        assert store.delete_chunk_ids([sample_logical], kb_id="t4-kb-a") == 1
        assert store.count() == POINTS - 1
        assert store.count(kb_id="t4-kb-b") == 400
        assert store.get_chunk_by_id(sample_logical, kb_id="t4-kb-a") is None

        # Re-upsert the deleted chunk, then exercise document and KB scoped deletion.
        embeddings, texts, metadata = batches["t4-a1"]
        assert store.add_chunks(
            embeddings=[embeddings[17]],
            texts=[texts[17]],
            doc_id="t4-a1",
            doc_title="T4 t4-a1",
            source_type="synthetic",
            chunk_indices=[17],
            kb_id="t4-kb-a",
            content_hash="t4-hash-t4-a1",
            extra_metadata=[metadata[17]],
        ) == [sample_logical]
        assert store.count() == POINTS
        assert store.delete_doc("t4-a2") == 200
        assert store.count() == 1_000
        assert store.count(kb_id="t4-kb-a") == 600
        assert store.count(kb_id="t4-kb-b") == 400
        assert store.get_chunk_by_id("t4-a2-0", kb_id="t4-kb-a") is None
        assert store.get_chunk_by_id("t4-b1-0", kb_id="t4-kb-b") is not None

        assert store.delete_kb("t4-kb-missing") == 0
        assert store.count() == 1_000
        assert store.delete_kb("t4-kb-b") == 400
        assert store.count() == 600
        assert store.count(kb_id="t4-kb-b") == 0
        assert store.count(kb_id="t4-kb-a") == 600
        assert store.delete_kb("t4-kb-a") == 600
        assert store.count() == 0

        # The same logical chunk_id in two KBs must occupy two physical points.
        # This is intentionally the same doc_id and chunk index on both writes;
        # changing either would evade the cross-scope collision gate.
        collision_logical_id = "t4-cross-scope-0"
        collision_metadata = [{"page": 1, "synthetic_t4": True}]
        assert store.add_chunks(
            embeddings=[_vector(2_000)],
            texts=["same chunk id in kb-a"],
            doc_id="t4-cross-scope",
            doc_title="T4 cross scope",
            source_type="synthetic",
            chunk_indices=[0],
            kb_id="t4-kb-a",
            content_hash="t4-cross-scope-hash-a",
            extra_metadata=collision_metadata,
        ) == [collision_logical_id]
        assert store.count() == 1
        assert store.count(kb_id="t4-kb-a") == 1
        assert store.count(kb_id="t4-kb-b") == 0
        collision_a = store.get_chunk_by_id(collision_logical_id, kb_id="t4-kb-a")
        assert collision_a is not None
        assert collision_a["payload"]["kb_id"] == "t4-kb-a"
        assert store.get_chunk_by_id(collision_logical_id, kb_id="t4-kb-b") is None

        assert store.add_chunks(
            embeddings=[_vector(2_000)],
            texts=["same chunk id in kb-b"],
            doc_id="t4-cross-scope",
            doc_title="T4 cross scope",
            source_type="synthetic",
            chunk_indices=[0],
            kb_id="t4-kb-b",
            content_hash="t4-cross-scope-hash-b",
            extra_metadata=collision_metadata,
        ) == [collision_logical_id]
        assert store.count() == 2
        assert store.count(kb_id="t4-kb-a") == 1
        assert store.count(kb_id="t4-kb-b") == 1
        collision_a_after = store.get_chunk_by_id(
            collision_logical_id, kb_id="t4-kb-a"
        )
        collision_b_after = store.get_chunk_by_id(
            collision_logical_id, kb_id="t4-kb-b"
        )
        assert collision_a_after is not None
        assert collision_b_after is not None
        assert collision_a_after["payload"]["kb_id"] == "t4-kb-a"
        assert collision_b_after["payload"]["kb_id"] == "t4-kb-b"
    except BaseException:
        main_failed = True
        _print_host_evidence(client, collection, "main-test-failure-before-cleanup")
        raise
    finally:
        try:
            if store is not None:
                store.close()
            _cleanup_collection(client, collection)
        except BaseException:
            _print_host_evidence(client, collection, "cleanup-failure")
            if not main_failed:
                raise
        finally:
            client.close()
