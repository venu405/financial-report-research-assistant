"""后端无关的 VectorStore contract。

这个模块只描述当前业务已经使用的 seam，不导入 Chroma 或 Qdrant。
实现可以是 Chroma、Qdrant adapter 或测试 double。

Canonical search result:
  - ``id``：物理存储记录 ID。当前 Chroma 中它等于逻辑 chunk_id；未来
    Qdrant 中可以是确定性的 point UUID。
  - ``payload``：后端无关的 payload，至少包含 ``chunk_id``、``text`` 以及
    当前 metadata 字段。
  - ``score``：相似度分数，沿用当前上层的 0..1 语义；不是 Chroma distance。

为保持现有调用方兼容，实现还应提供 ``chunk_id``、``text``、``metadata``
和可选 ``distance`` 字段。业务层使用这些兼容字段，后端迁移只在 adapter
边界完成物理 ID 与 payload 的转换。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

Metadata = dict[str, Any]
Where = dict[str, Any]
VectorSnapshot = dict[str, Any]
VectorSearchHit = dict[str, Any]


@runtime_checkable
class VectorStoreBackend(Protocol):
    """当前知识库业务需要的最小向量存储接口。"""

    def mutation_seq(self, kb_id: str | None) -> int: ...

    def add_chunks(
        self,
        *,
        embeddings: list[list[float]],
        texts: list[str],
        doc_id: str,
        doc_title: str,
        source_type: str,
        chunk_indices: list[int],
        kb_id: str = "default",
        content_hash: str | None = None,
        extra_metadata: list[Metadata] | None = None,
    ) -> list[str]: ...

    def find_doc_by_hash(
        self, content_hash: str, *, kb_id: str | None = None
    ) -> str | None: ...

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        where: Where | None = None,
        kb_id: str | None = None,
    ) -> list[VectorSearchHit]: ...

    def get_doc_ids(self, doc_id: str) -> list[str]: ...

    def get_related_chunks(
        self,
        *,
        kb_id: str,
        chunk_id: str | None = None,
        parent_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_chunks_by_parent_id(
        self, parent_id: str, *, kb_id: str
    ) -> list[dict[str, Any]]: ...

    def get_chunk_by_id(
        self, chunk_id: str, *, kb_id: str
    ) -> dict[str, Any] | None: ...

    def snapshot_doc(self, doc_id: str) -> VectorSnapshot: ...

    def restore_doc_snapshot(self, snapshot: VectorSnapshot) -> int: ...

    def delete_chunk_ids(self, ids: list[str], kb_id: str | None = None) -> int: ...

    def delete_kb(self, kb_id: str) -> int: ...

    def delete_doc(self, doc_id: str) -> int: ...

    def get_doc_kb_id(self, doc_id: str) -> str | None: ...

    def count(self, *, kb_id: str | None = None) -> int: ...

    def all_items(
        self, *, where: Where | None = None, kb_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def list_docs(
        self, *, kb_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]: ...

    def list_kbs(self) -> list[str]: ...

    def migrate_default_kb_id(self, kb_id: str = "default") -> int: ...
