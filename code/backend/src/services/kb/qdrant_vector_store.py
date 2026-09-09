"""Isolated Qdrant implementation of the backend-neutral VectorStore seam.

This module is deliberately not wired into the application factory.  The
existing Chroma import path remains the default until a later expand-migrate-
contract change has its own integration decision and evidence.
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from threading import Lock
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid5

import httpx

from .vector_store_contract import VectorStoreBackend

DEFAULT_KB_ID = "default"

# This namespace is part of the storage contract. It must never be changed
# after points have been written, because the UUID is the Qdrant primary key.
CHUNK_POINT_NAMESPACE = UUID("3e4f6e8a-4d58-5a9e-9d9b-3a9e3b0a9c72")

_INDEX_SCHEMAS: tuple[tuple[str, str], ...] = (
    ("kb_id", "keyword"),
    ("doc_id", "keyword"),
    ("chunk_id", "keyword"),
    ("content_hash", "keyword"),
    ("parent_id", "keyword"),
    ("table_id", "keyword"),
    ("page", "integer"),
    ("page_start", "integer"),
    ("page_end", "integer"),
    ("chunk_index", "integer"),
    ("is_table", "bool"),
)
_FILTER_OPERATORS = {
    "$eq",
    "$ne",
    "$in",
    "$nin",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
}
_RESERVED_PAYLOAD_FIELDS = {
    "chunk_id",
    "text",
    "doc_id",
    "doc_title",
    "source_type",
    "chunk_index",
    "kb_id",
    "content_hash",
}


class QdrantVectorStoreError(RuntimeError):
    """A non-leaky, location-bearing Qdrant adapter error."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _normalize_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _scoped_point_name(*, kb_id: str, chunk_id: str) -> str:
    """Build an unambiguous, normalized UUIDv5 name for one KB chunk."""

    return json.dumps(
        {
            "chunk_id": _normalize_identifier(chunk_id, field="chunk_id"),
            "kb_id": _normalize_identifier(kb_id, field="kb_id"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def point_id_for_chunk(chunk_id: str, kb_id: str = DEFAULT_KB_ID) -> str:
    """Return the stable UUIDv5 for a normalized ``kb_id``/``chunk_id`` pair."""

    return str(
        uuid5(
            CHUNK_POINT_NAMESPACE,
            _scoped_point_name(kb_id=kb_id, chunk_id=chunk_id),
        )
    )


def legacy_point_id_for_chunk(chunk_id: str) -> str:
    """Return the pre-scope UUID used by points written before the T4 fix."""

    normalized_chunk_id = _normalize_identifier(chunk_id, field="chunk_id")
    return str(uuid5(CHUNK_POINT_NAMESPACE, normalized_chunk_id))


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _json_safe(value: Any, *, field: str) -> Any:
    """Validate and copy values accepted by JSON/Qdrant payloads.

    Unsupported values fail explicitly.  In particular, metadata is never
    silently stringified or dropped at this adapter boundary.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} contains an invalid object key")
            copied[key] = _json_safe(item, field=f"{field}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field} contains a non-JSON value: {type(value).__name__}")


def _validate_vector(vector: Sequence[Any], *, vector_size: int, field: str) -> list[float]:
    if isinstance(vector, (str, bytes)):
        raise ValueError(f"{field} must be a numeric vector")
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError(f"{field} must be a numeric vector") from exc
    if len(values) != vector_size:
        raise ValueError(
            f"{field} dimension mismatch: expected {vector_size}, got {len(values)}"
        )
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{field}[{index}] must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be numeric") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{field}[{index}] must be finite")
        result.append(number)
    return result


def _filter_scalar(value: Any, *, field: str) -> Any:
    if not _is_json_scalar(value) or value is None:
        raise ValueError(f"filter value for {field!r} must be a non-null scalar")
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        raise ValueError(f"filter value for {field!r} must be finite")
    return value


class QdrantVectorStore(VectorStoreBackend):
    """Qdrant REST adapter for the existing VectorStoreBackend contract."""

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        timeout: float = 10.0,
        create_if_missing: bool = False,
        scroll_page_size: int = 256,
        client: httpx.Client | None = None,
    ):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if not isinstance(collection_name, str) or not collection_name:
            raise ValueError("collection_name must be a non-empty string")
        if not isinstance(vector_size, int) or isinstance(vector_size, bool) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(scroll_page_size, int) or scroll_page_size <= 0:
            raise ValueError("scroll_page_size must be a positive integer")

        self._url = url.rstrip("/")
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._scroll_page_size = scroll_page_size
        self._api_key = api_key
        self._request_headers = {"api-key": api_key} if api_key else {}
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=self._url,
                headers=self._request_headers,
                timeout=timeout,
            )
        )
        self._mutation_seq_by_kb: dict[str, int] = {}
        self._seq_lock = Lock()

        self._ensure_collection(create_if_missing=create_if_missing)

    def close(self) -> None:
        """Close the client created by this adapter; injected clients remain owned by callers."""

        if self._owns_client:
            self._client.close()

    @staticmethod
    def point_id_for_chunk(
        chunk_id: str, kb_id: str = DEFAULT_KB_ID
    ) -> str:
        """Class-level convenience alias for the public mapping function."""

        return point_id_for_chunk(chunk_id, kb_id)

    def __enter__(self) -> QdrantVectorStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _collection_path(self, suffix: str = "") -> str:
        name = quote(self._collection_name, safe="")
        return f"/collections/{name}{suffix}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                f"{self._url}{path}",
                headers=self._request_headers,
                params=params,
                json=json_body,
            )
        except httpx.HTTPError as exc:
            raise QdrantVectorStoreError(
                f"Qdrant {method} {path} transport error: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            summary = " ".join(response.text.split())
            if self._api_key:
                summary = summary.replace(self._api_key, "[redacted]")
            if len(summary) > 256:
                summary = f"{summary[:253]}..."
            detail = f": {summary}" if summary else ""
            raise QdrantVectorStoreError(
                f"Qdrant {method} {path} returned HTTP {response.status_code}{detail}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise QdrantVectorStoreError(
                f"Qdrant {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise QdrantVectorStoreError(
                f"Qdrant {method} {path} returned a non-object response"
            )
        return data

    @staticmethod
    def _result(data: Mapping[str, Any], *, method: str, path: str) -> Any:
        if "result" not in data:
            raise QdrantVectorStoreError(
                f"Qdrant {method} {path} response omitted result"
            )
        return data["result"]

    def _get_collection_info(self) -> Mapping[str, Any]:
        data = self._request("GET", self._collection_path())
        result = self._result(
            data, method="GET", path=self._collection_path()
        )
        if not isinstance(result, Mapping):
            raise QdrantVectorStoreError("Qdrant collection info result is not an object")
        return result

    def healthcheck(self) -> bool:
        """Check collection reachability with one O(1) collection-info request."""
        self._vector_config(self._get_collection_info())
        return True

    @staticmethod
    def _vector_config(info: Mapping[str, Any]) -> tuple[int, str]:
        try:
            vectors = info["config"]["params"]["vectors"]
        except (KeyError, TypeError) as exc:
            raise QdrantVectorStoreError(
                "Qdrant collection info omitted unnamed vector configuration"
            ) from exc
        if not isinstance(vectors, Mapping) or "size" not in vectors or "distance" not in vectors:
            raise QdrantVectorStoreError(
                "Qdrant collection must use one unnamed vector configuration"
            )
        try:
            size = int(vectors["size"])
        except (TypeError, ValueError) as exc:
            raise QdrantVectorStoreError("Qdrant collection vector size is invalid") from exc
        return size, str(vectors["distance"]).lower()

    def _ensure_collection(self, *, create_if_missing: bool) -> None:
        try:
            info = self._get_collection_info()
        except QdrantVectorStoreError as exc:
            if exc.status_code != 404 or not create_if_missing:
                if exc.status_code == 404:
                    raise QdrantVectorStoreError(
                        f"Qdrant collection {self._collection_name!r} does not exist; "
                        "create_if_missing=False",
                        status_code=404,
                    ) from exc
                raise
            try:
                self._request(
                    "PUT",
                    self._collection_path(),
                    json_body={
                        "vectors": {
                            "size": self._vector_size,
                            "distance": "Cosine",
                        }
                    },
                )
            except QdrantVectorStoreError as create_exc:
                if create_exc.status_code != 409:
                    raise
            info = self._get_collection_info()

        actual_size, actual_distance = self._vector_config(info)
        if actual_size != self._vector_size or actual_distance != "cosine":
            raise ValueError(
                f"Qdrant collection {self._collection_name!r} vector mismatch: "
                f"expected size={self._vector_size}, distance=Cosine; "
                f"got size={actual_size}, distance={actual_distance}"
            )

        if create_if_missing:
            for field_name, field_schema in _INDEX_SCHEMAS:
                try:
                    self._request(
                        "PUT",
                        self._collection_path("/index"),
                        params={"wait": "true"},
                        json_body={
                            "field_name": field_name,
                            "field_schema": field_schema,
                        },
                    )
                except QdrantVectorStoreError as exc:
                    if exc.status_code != 409:
                        raise

    def mutation_seq(self, kb_id: str | None) -> int:
        key = kb_id or "__global__"
        with self._seq_lock:
            return self._mutation_seq_by_kb.get(key, 0)

    def _bump_seq(self, kb_id: str | None) -> None:
        key = kb_id or "__global__"
        with self._seq_lock:
            self._mutation_seq_by_kb[key] = self._mutation_seq_by_kb.get(key, 0) + 1

    @staticmethod
    def _merge_payload(
        *,
        required: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_extra = _json_safe(dict(extra or {}), field="extra_metadata")
        if not isinstance(safe_extra, dict):  # pragma: no cover - guarded by helper
            raise ValueError("extra_metadata must be an object")
        for key, value in required.items():
            if key in safe_extra and key in _RESERVED_PAYLOAD_FIELDS:
                if safe_extra[key] != value:
                    raise ValueError(f"extra_metadata conflicts with required field {key!r}")
        payload = dict(safe_extra)
        payload.update(_json_safe(dict(required), field="payload"))
        return payload

    def add_chunks(
        self,
        *,
        embeddings: list[list[float]],
        texts: list[str],
        doc_id: str,
        doc_title: str,
        source_type: str,
        chunk_indices: list[int],
        kb_id: str = DEFAULT_KB_ID,
        content_hash: str | None = None,
        extra_metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if len(embeddings) != len(texts) or len(embeddings) != len(chunk_indices):
            raise ValueError("embeddings, texts, and chunk_indices lengths must match")
        if extra_metadata is not None and len(extra_metadata) != len(embeddings):
            raise ValueError("extra_metadata length must match embeddings")
        if not embeddings:
            return []
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        kb_id = _normalize_identifier(kb_id, field="kb_id")
        if not isinstance(content_hash, (str, type(None))):
            raise ValueError("content_hash must be a string or None")

        points: list[dict[str, Any]] = []
        logical_ids: list[str] = []
        for index, (embedding, text, chunk_index) in enumerate(
            zip(embeddings, texts, chunk_indices, strict=True)
        ):
            if not isinstance(text, str):
                raise ValueError(f"texts[{index}] must be a string")
            if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
                raise ValueError(f"chunk_indices[{index}] must be an integer")
            vector = _validate_vector(
                embedding, vector_size=self._vector_size, field=f"embeddings[{index}]"
            )
            chunk_id = f"{doc_id}-{chunk_index}"
            metadata = extra_metadata[index] if extra_metadata is not None else {}
            if not isinstance(metadata, Mapping):
                raise ValueError(f"extra_metadata[{index}] must be an object")
            payload = self._merge_payload(
                required={
                    "chunk_id": chunk_id,
                    "text": text,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                    "kb_id": kb_id,
                    "content_hash": content_hash or "",
                },
                extra=metadata,
            )
            logical_ids.append(chunk_id)
            points.append(
                {
                    "id": point_id_for_chunk(chunk_id, kb_id),
                    "vector": vector,
                    "payload": payload,
                }
            )

        self._upsert_points(points)
        self._bump_seq(kb_id)
        return logical_ids

    def find_doc_by_hash(
        self, content_hash: str, *, kb_id: str | None = None
    ) -> str | None:
        if not content_hash:
            return None
        points = self._scroll_points(
            filter_body=self._where_to_filter({"content_hash": content_hash}, kb_id),
            with_vector=False,
            limit=1,
        )
        for point in points:
            payload = point.get("payload") or {}
            if payload.get("doc_id"):
                return str(payload["doc_id"])
        return None

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if top_k == 0:
            return []
        vector = _validate_vector(
            query_embedding, vector_size=self._vector_size, field="query_embedding"
        )
        body: dict[str, Any] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        filter_body = self._where_to_filter(where, kb_id)
        if filter_body:
            body["filter"] = filter_body
        data = self._request(
            "POST", self._collection_path("/points/search"), json_body=body
        )
        result = self._result(
            data, method="POST", path=self._collection_path("/points/search")
        )
        if result is None:
            return []
        if isinstance(result, Mapping):
            result = result.get("points")
        if not isinstance(result, list):
            raise QdrantVectorStoreError("Qdrant search result is not an array")
        return [self._point_record(point, score=point.get("score")) for point in result]

    @staticmethod
    def _simple_match(field: str, value: Any) -> dict[str, Any]:
        return {"key": field, "match": {"value": _filter_scalar(value, field=field)}}

    @classmethod
    def _compile_where_node(cls, where: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(where, Mapping):
            raise ValueError("where must be an object")
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        should: list[dict[str, Any]] = []
        for field, value in where.items():
            if not isinstance(field, str) or not field:
                raise ValueError("where fields must be non-empty strings")
            if field == "$and":
                if not isinstance(value, list):
                    raise ValueError("$and must be a list")
                for child in value:
                    child_filter = cls._compile_where_node(child)
                    if child_filter:
                        # Qdrant REST encodes a recursive Filter directly as
                        # a clause item: {"must": [{"must": [...]}]}.
                        # {"filter": {...}} is not a valid v1.19 REST shape.
                        must.append(child_filter)
                continue
            if field == "$or":
                if not isinstance(value, list) or not value:
                    raise ValueError("$or must be a non-empty list")
                for child in value:
                    child_filter = cls._compile_where_node(child)
                    if child_filter:
                        should.append(child_filter)
                if not should:
                    raise ValueError("$or must contain at least one condition")
                continue
            if field.startswith("$"):
                raise ValueError(f"unsupported where operator {field!r}")

            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(f"filter operation for {field!r} is empty")
                for operator, operand in value.items():
                    if operator not in _FILTER_OPERATORS:
                        raise ValueError(f"unsupported where operator {operator!r}")
                    if operator in {"$in", "$nin"}:
                        if not isinstance(operand, list):
                            raise ValueError(f"{operator} for {field!r} must be a list")
                        values = [
                            _filter_scalar(item, field=field) for item in operand
                        ]
                        condition = {"key": field, "match": {"any": values}}
                    elif operator in {"$gt", "$gte", "$lt", "$lte"}:
                        number = _filter_scalar(operand, field=field)
                        if isinstance(number, bool) or not isinstance(number, (int, float)):
                            raise ValueError(
                                f"{operator} for {field!r} requires a numeric value"
                            )
                        condition = {
                            "key": field,
                            "range": {operator[1:]: number},
                        }
                    else:
                        condition = cls._simple_match(field, operand)

                    if operator in {"$ne", "$nin"}:
                        must_not.append(condition)
                    else:
                        must.append(condition)
            else:
                must.append(cls._simple_match(field, value))
        compiled: dict[str, Any] = {}
        if must:
            compiled["must"] = must
        if must_not:
            compiled["must_not"] = must_not
        if should:
            compiled["should"] = should
        return compiled

    @classmethod
    def _where_to_filter(
        cls, where: Mapping[str, Any] | None, kb_id: str | None
    ) -> dict[str, Any] | None:
        if where is not None and not isinstance(where, Mapping):
            raise ValueError("where must be an object")
        compiled = cls._compile_where_node(where) if where else {}
        if kb_id is not None:
            kb_id = _normalize_identifier(kb_id, field="kb_id")
            compiled.setdefault("must", []).insert(
                0, cls._simple_match("kb_id", kb_id)
            )
        return compiled or None

    def _scroll_points(
        self,
        *,
        filter_body: dict[str, Any] | None = None,
        with_vector: bool,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        offset: Any = None
        first_page = True
        while True:
            body: dict[str, Any] = {
                "limit": limit if first_page and limit is not None else self._scroll_page_size,
                "with_payload": True,
                "with_vector": with_vector,
            }
            if filter_body:
                body["filter"] = filter_body
            if offset is not None:
                body["offset"] = offset
            data = self._request(
                "POST", self._collection_path("/points/scroll"), json_body=body
            )
            result = self._result(
                data,
                method="POST",
                path=self._collection_path("/points/scroll"),
            )
            if not isinstance(result, Mapping):
                raise QdrantVectorStoreError("Qdrant scroll result is not an object")
            page = result.get("points") or []
            if not isinstance(page, list):
                raise QdrantVectorStoreError("Qdrant scroll points is not an array")
            points.extend(point for point in page if isinstance(point, Mapping))
            if limit is not None and len(points) >= limit:
                return points[:limit]
            next_offset = result.get("next_page_offset")
            if next_offset is None:
                break
            if next_offset == offset:
                raise QdrantVectorStoreError("Qdrant scroll offset did not advance")
            offset = next_offset
            first_page = False
        return points

    def _upsert_points(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        self._request(
            "PUT",
            self._collection_path("/points"),
            params={"wait": "true"},
            json_body={"points": points},
        )

    def _delete_point_ids(self, point_ids: Iterable[str]) -> None:
        unique_ids = list(dict.fromkeys(point_ids))
        if not unique_ids:
            return
        for start in range(0, len(unique_ids), self._scroll_page_size):
            self._request(
                "POST",
                self._collection_path("/points/delete"),
                params={"wait": "true"},
                json_body={"points": unique_ids[start : start + self._scroll_page_size]},
            )

    @staticmethod
    def _point_vector(point: Mapping[str, Any]) -> list[float]:
        vector = point.get("vector")
        if isinstance(vector, Mapping):
            # Named vectors are not part of this adapter's collection contract.
            raise QdrantVectorStoreError("Qdrant point returned named vectors unexpectedly")
        if vector is None:
            raise QdrantVectorStoreError("Qdrant point omitted its vector")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise QdrantVectorStoreError("Qdrant point vector is invalid") from exc

    @staticmethod
    def _point_record(
        point: Mapping[str, Any], *, score: Any | None = None
    ) -> dict[str, Any]:
        physical_id = str(point.get("id", ""))
        raw_payload = point.get("payload") or {}
        if not isinstance(raw_payload, Mapping):
            raise QdrantVectorStoreError("Qdrant point payload is not an object")
        payload = dict(raw_payload)
        logical_id = str(payload.get("chunk_id") or physical_id)
        text = payload.get("text") or ""
        metadata = {
            key: value for key, value in payload.items() if key not in {"chunk_id", "text"}
        }
        record: dict[str, Any] = {
            "id": physical_id,
            "payload": payload,
            "chunk_id": logical_id,
            "text": text,
            "metadata": metadata,
        }
        if score is not None:
            try:
                cosine_score = float(score)
            except (TypeError, ValueError) as exc:
                raise QdrantVectorStoreError("Qdrant search score is invalid") from exc
            record["score"] = cosine_score
            record["distance"] = 1.0 - cosine_score
        return record

    @staticmethod
    def _point_matches_scope(
        point: Mapping[str, Any], *, chunk_id: str, kb_id: str
    ) -> bool:
        payload = point.get("payload") or {}
        if payload.get("chunk_id") != chunk_id:
            return False
        stored_kb_id = payload.get("kb_id")
        if stored_kb_id is None:
            # An old unscoped point is only safe to treat as the default KB.
            return kb_id == DEFAULT_KB_ID
        return stored_kb_id == kb_id

    @staticmethod
    def _point_belongs_to_kb(point: Mapping[str, Any], kb_id: str) -> bool:
        payload = point.get("payload") or {}
        stored_kb_id = payload.get("kb_id")
        return stored_kb_id == kb_id or (
            stored_kb_id is None and kb_id == DEFAULT_KB_ID
        )

    @staticmethod
    def _item_record(point: Mapping[str, Any]) -> dict[str, Any]:
        record = QdrantVectorStore._point_record(point)
        return {
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "metadata": record["metadata"],
        }

    def _scroll_points_for_kb(
        self,
        *,
        kb_id: str,
        where: Mapping[str, Any] | None = None,
        with_vector: bool = False,
    ) -> list[dict[str, Any]]:
        """Scroll one KB, treating an old missing-kb payload as default only."""

        scope_kb_id = _normalize_identifier(kb_id, field="kb_id")
        if scope_kb_id == DEFAULT_KB_ID:
            points = self._scroll_points(
                filter_body=self._where_to_filter(where, None),
                with_vector=with_vector,
            )
            return [
                point
                for point in points
                if self._point_belongs_to_kb(point, scope_kb_id)
            ]
        return self._scroll_points(
            filter_body=self._where_to_filter(where, scope_kb_id),
            with_vector=with_vector,
        )

    @staticmethod
    def _chunk_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
        metadata = item.get("metadata") or {}
        raw_index = metadata.get("chunk_index", 0)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0
        return index, str(item.get("chunk_id", ""))

    def _points_by_physical_ids(self, point_ids: list[str]) -> list[dict[str, Any]]:
        data = self._request(
            "POST",
            self._collection_path("/points"),
            json_body={
                "ids": list(dict.fromkeys(point_ids)),
                "with_payload": True,
                "with_vector": True,
            },
        )
        result = self._result(data, method="POST", path=self._collection_path("/points"))
        if result is None:
            return []
        if isinstance(result, Mapping):
            result = result.get("points")
        if not isinstance(result, list):
            raise QdrantVectorStoreError("Qdrant point lookup result is not an array")
        return [point for point in result if isinstance(point, Mapping)]

    def get_doc_ids(self, doc_id: str) -> list[str]:
        points = self._scroll_points(
            filter_body=self._where_to_filter({"doc_id": doc_id}, None),
            with_vector=False,
        )
        items = sorted((self._item_record(point) for point in points), key=self._chunk_sort_key)
        return [item["chunk_id"] for item in items]

    def get_related_chunks(
        self,
        *,
        kb_id: str,
        chunk_id: str | None = None,
        parent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if (chunk_id is None) == (parent_id is None):
            raise ValueError("exactly one of chunk_id and parent_id is required")
        if chunk_id is not None:
            target = self.get_chunk_by_id(chunk_id, kb_id=kb_id)
            if target is None:
                return []
            metadata = target.get("metadata") or {}
            parent_id = str(metadata.get("parent_id") or "")
            if not parent_id and metadata.get("chunk_type") == "parent":
                parent_id = chunk_id
            if not parent_id:
                return [target]

        related = self.all_items(where={"parent_id": parent_id}, kb_id=kb_id)
        parent = self.get_chunk_by_id(str(parent_id), kb_id=kb_id)
        if parent is not None:
            related.append(parent)
        deduped = {item["chunk_id"]: item for item in related}
        return sorted(deduped.values(), key=self._chunk_sort_key)

    def get_chunks_by_parent_id(
        self, parent_id: str, *, kb_id: str
    ) -> list[dict[str, Any]]:
        return self.get_related_chunks(parent_id=parent_id, kb_id=kb_id)

    def get_chunk_by_id(self, chunk_id: str, *, kb_id: str) -> dict[str, Any] | None:
        scope_kb_id = _normalize_identifier(kb_id, field="kb_id")
        physical_ids = [
            point_id_for_chunk(chunk_id, scope_kb_id),
            legacy_point_id_for_chunk(chunk_id),
        ]
        points = self._points_by_physical_ids(list(dict.fromkeys(physical_ids)))
        by_id = {str(point.get("id")): point for point in points}
        for physical_id in physical_ids:
            point = by_id.get(physical_id)
            if point is not None and self._point_matches_scope(
                point, chunk_id=chunk_id, kb_id=scope_kb_id
            ):
                return self._point_record(point)
        return None

    def snapshot_doc(self, doc_id: str) -> dict[str, Any]:
        points = self._scroll_points(
            filter_body=self._where_to_filter({"doc_id": doc_id}, None),
            with_vector=True,
        )
        selected: dict[tuple[str, str], Mapping[str, Any]] = {}
        kb_ids: set[str] = set()
        for point in points:
            payload = point.get("payload") or {}
            logical_id = payload.get("chunk_id")
            if not isinstance(logical_id, str) or not logical_id:
                raise QdrantVectorStoreError(
                    f"Qdrant document {doc_id!r} contains a point without chunk_id"
                )
            raw_kb_id = payload.get("kb_id")
            scope_kb_id = _normalize_identifier(
                raw_kb_id if raw_kb_id is not None else DEFAULT_KB_ID,
                field="kb_id",
            )
            kb_ids.add(scope_kb_id)
            key = (scope_kb_id, logical_id)
            current = selected.get(key)
            current_is_canonical = (
                str(point.get("id"))
                == point_id_for_chunk(logical_id, scope_kb_id)
            )
            if current is None:
                selected[key] = point
                continue
            previous_is_canonical = (
                str(current.get("id"))
                == point_id_for_chunk(logical_id, scope_kb_id)
            )
            if current_is_canonical and not previous_is_canonical:
                selected[key] = point

        if len(kb_ids) > 1:
            raise ValueError("document snapshot must belong to one kb_id")
        ordered_points = sorted(
            selected.values(),
            key=lambda point: self._chunk_sort_key(self._point_record(point)),
        )
        records = [self._point_record(point) for point in ordered_points]
        embeddings: list[list[float]] = []
        for point in ordered_points:
            vector = self._point_vector(point)
            if len(vector) != self._vector_size:
                raise QdrantVectorStoreError(
                    f"Qdrant snapshot vector dimension mismatch: expected {self._vector_size}, "
                    f"got {len(vector)}"
                )
            embeddings.append(vector)
        return {
            "ids": [record["chunk_id"] for record in records],
            "documents": [record["text"] for record in records],
            "metadatas": [record["metadata"] for record in records],
            "embeddings": embeddings,
        }

    def restore_doc_snapshot(self, snapshot: dict[str, Any]) -> int:
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be an object")
        ids = list(snapshot.get("ids") or [])
        if not ids:
            raise ValueError("document snapshot is empty")
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids):
            raise ValueError("document snapshot ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("document snapshot contains duplicate ids")
        documents = list(snapshot.get("documents") or [])
        metadatas = list(snapshot.get("metadatas") or [])
        embeddings = list(snapshot.get("embeddings") or [])
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("snapshot ids/documents/metadatas/embeddings lengths must match")
        if not isinstance(metadatas[0], Mapping):
            raise ValueError("snapshot metadata must include doc_id")
        doc_id = str(metadatas[0].get("doc_id") or "")
        if not doc_id:
            raise ValueError("document snapshot is missing doc_id")

        points: list[dict[str, Any]] = []
        logical_ids: list[str] = []
        kb_ids: set[str] = set()
        for index, (logical_id, text, metadata, embedding) in enumerate(
            zip(ids, documents, metadatas, embeddings, strict=True)
        ):
            if not isinstance(logical_id, str) or not logical_id:
                raise ValueError(f"snapshot ids[{index}] must be a non-empty string")
            if not isinstance(text, str):
                raise ValueError(f"snapshot documents[{index}] must be a string")
            if not isinstance(metadata, Mapping) or not metadata.get("doc_id"):
                raise ValueError(f"snapshot metadatas[{index}] is missing doc_id")
            if str(metadata["doc_id"]) != doc_id:
                raise ValueError("snapshot contains multiple doc_id values")
            kb_value = _normalize_identifier(
                metadata.get("kb_id") or DEFAULT_KB_ID,
                field="snapshot kb_id",
            )
            kb_ids.add(kb_value)
            vector = _validate_vector(
                embedding, vector_size=self._vector_size, field=f"snapshot embeddings[{index}]"
            )
            payload = self._merge_payload(
                required={
                    "chunk_id": logical_id,
                    "text": text,
                    "doc_id": doc_id,
                    "doc_title": metadata.get("doc_title", ""),
                    "source_type": metadata.get("source_type", ""),
                    "chunk_index": metadata.get("chunk_index", index),
                    "kb_id": kb_value,
                    "content_hash": metadata.get("content_hash", ""),
                },
                extra=metadata,
            )
            logical_ids.append(logical_id)
            points.append(
                {
                    "id": point_id_for_chunk(logical_id, kb_value),
                    "vector": vector,
                    "payload": payload,
                }
            )
        if len(kb_ids) > 1:
            raise ValueError("snapshot must belong to one kb_id")

        old_points = self._scroll_points(
            filter_body=self._where_to_filter({"doc_id": doc_id}, None),
            with_vector=False,
        )
        target_kb_id = next(iter(kb_ids))
        desired_physical_ids = {
            point_id_for_chunk(logical_id, target_kb_id)
            for logical_id in logical_ids
        }
        stale_ids = [
            str(point.get("id"))
            for point in old_points
            if self._point_belongs_to_kb(point, target_kb_id)
            and str(point.get("id")) not in desired_physical_ids
        ]
        self._upsert_points(points)
        self._delete_point_ids(stale_ids)
        self._bump_seq(target_kb_id)
        return len(points)

    def delete_chunk_ids(self, ids: list[str], kb_id: str | None = None) -> int:
        if not ids:
            return 0
        logical_ids = list(dict.fromkeys(ids))
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in logical_ids):
            raise ValueError("ids must contain non-empty strings")
        scope_kb_id = (
            _normalize_identifier(kb_id, field="kb_id")
            if kb_id is not None
            else None
        )
        if scope_kb_id is None:
            points = self._scroll_points(
                filter_body=self._where_to_filter(
                    {"chunk_id": {"$in": logical_ids}}, None
                ),
                with_vector=False,
            )
        else:
            physical_ids = [
                physical_id
                for chunk_id in logical_ids
                for physical_id in (
                    point_id_for_chunk(chunk_id, scope_kb_id),
                    legacy_point_id_for_chunk(chunk_id),
                )
            ]
            points = self._points_by_physical_ids(
                list(dict.fromkeys(physical_ids))
            )
        deletable: list[str] = []
        requested = set(logical_ids)
        for point in points:
            payload = point.get("payload") or {}
            if payload.get("chunk_id") not in requested:
                continue
            if scope_kb_id is not None and not self._point_matches_scope(
                point,
                chunk_id=str(payload.get("chunk_id")),
                kb_id=scope_kb_id,
            ):
                continue
            deletable.append(str(point.get("id")))
        self._delete_point_ids(deletable)
        if deletable:
            self._bump_seq(scope_kb_id)
        return len(deletable)

    def delete_kb(self, kb_id: str) -> int:
        scope_kb_id = _normalize_identifier(kb_id, field="kb_id")
        points = self._scroll_points_for_kb(kb_id=scope_kb_id)
        point_ids = [str(point.get("id")) for point in points]
        self._delete_point_ids(point_ids)
        if point_ids:
            self._bump_seq(scope_kb_id)
        return len(point_ids)

    def delete_doc(self, doc_id: str) -> int:
        points = self._scroll_points(
            filter_body=self._where_to_filter({"doc_id": doc_id}, None),
            with_vector=False,
        )
        point_ids = [str(point.get("id")) for point in points]
        kb_ids = {
            str((point.get("payload") or {}).get("kb_id"))
            for point in points
            if (point.get("payload") or {}).get("kb_id") is not None
        }
        self._delete_point_ids(point_ids)
        for kb_id in kb_ids:
            self._bump_seq(kb_id)
        return len(point_ids)

    def get_doc_kb_id(self, doc_id: str) -> str | None:
        points = self._scroll_points(
            filter_body=self._where_to_filter({"doc_id": doc_id}, None),
            with_vector=False,
            limit=1,
        )
        for point in points:
            payload = point.get("payload") or {}
            if "kb_id" in payload and payload["kb_id"] is not None:
                return str(payload["kb_id"])
            logical_id = payload.get("chunk_id")
            if isinstance(logical_id, str) and str(point.get("id")) == legacy_point_id_for_chunk(logical_id):
                return DEFAULT_KB_ID
        return None

    def count(self, *, kb_id: str | None = None) -> int:
        if kb_id is not None:
            return len(self._scroll_points_for_kb(kb_id=kb_id))
        return len(
            self._scroll_points(
                filter_body=self._where_to_filter(None, kb_id),
                with_vector=False,
            )
        )

    def all_items(
        self,
        *,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if kb_id is not None:
            points = self._scroll_points_for_kb(kb_id=kb_id, where=where)
        else:
            points = self._scroll_points(
                filter_body=self._where_to_filter(where, None),
                with_vector=False,
            )
        return [self._item_record(point) for point in points]

    def list_docs(
        self,
        *,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if kb_id is not None:
            points = self._scroll_points_for_kb(kb_id=kb_id)
        else:
            points = self._scroll_points(with_vector=False)
        docs: dict[str, dict[str, Any]] = {}
        for point in points:
            payload = point.get("payload") or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            key = str(doc_id)
            if key not in docs:
                docs[key] = {
                    "doc_id": key,
                    "title": payload.get("doc_title", ""),
                    "source_type": payload.get("source_type", ""),
                    "kb_id": payload.get("kb_id", DEFAULT_KB_ID),
                    "chunks": 0,
                }
            docs[key]["chunks"] += 1
        ordered = [docs[key] for key in sorted(docs)]
        return ordered[offset : offset + limit], len(ordered)

    def list_kbs(self) -> list[str]:
        points = self._scroll_points(with_vector=False)
        kbs = {
            str(payload["kb_id"]) if payload.get("kb_id") is not None else DEFAULT_KB_ID
            for point in points
            for payload in [point.get("payload") or {}]
            if "kb_id" in payload or payload.get("chunk_id")
        }
        return sorted(kbs)

    def migrate_default_kb_id(self, kb_id: str = DEFAULT_KB_ID) -> int:
        target_kb_id = _normalize_identifier(kb_id, field="kb_id")
        points = self._scroll_points(with_vector=True)
        fixes: list[dict[str, Any]] = []
        stale_ids: list[str] = []
        for point in points:
            payload = dict(point.get("payload") or {})
            if "kb_id" in payload and payload["kb_id"] is not None:
                continue
            logical_id = payload.get("chunk_id")
            if not isinstance(logical_id, str) or not logical_id:
                raise QdrantVectorStoreError(
                    "Qdrant legacy point omitted chunk_id during default KB migration"
                )
            payload["kb_id"] = target_kb_id
            vector = self._point_vector(point)
            if len(vector) != self._vector_size:
                raise QdrantVectorStoreError(
                    f"Qdrant migration vector dimension mismatch: expected {self._vector_size}, "
                    f"got {len(vector)}"
                )
            fixes.append(
                {
                    "id": point_id_for_chunk(logical_id, target_kb_id),
                    "vector": vector,
                    "payload": _json_safe(payload, field="migrated_payload"),
                }
            )
            old_id = str(point.get("id"))
            new_id = point_id_for_chunk(logical_id, target_kb_id)
            if old_id != new_id:
                stale_ids.append(old_id)
        self._upsert_points(fixes)
        self._delete_point_ids(stale_ids)
        if fixes:
            self._bump_seq(target_kb_id)
        return len(fixes)


__all__ = [
    "CHUNK_POINT_NAMESPACE",
    "DEFAULT_KB_ID",
    "QdrantVectorStore",
    "QdrantVectorStoreError",
    "VectorStoreBackend",
    "legacy_point_id_for_chunk",
    "point_id_for_chunk",
]
