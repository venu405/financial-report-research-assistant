"""RAG V2 稳定性回归：旧索引、快照、跨库去重和父子引用边界。"""
from __future__ import annotations

from typing import Any

import pytest

from services.kb.qa_graph import build_qa_graph, run_qa
from services.kb.retriever import HybridRetriever
from services.kb.vector_store import DEFAULT_KB_ID, VectorStore
from tests.mocks import FakeEmbedding, FakeLLM


class _PagedCollection:
    """只返回一页的 collection stub，用来强制 snapshot 走分页。"""

    def __init__(self, rows: list[dict[str, Any]], page_size: int = 2):
        self._rows = rows
        self._page_size = page_size

    def get(
        self,
        *,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        assert where == {"doc_id": "snapshot-doc"}
        assert include and "documents" in include and "metadatas" in include
        start = int(offset or 0)
        size = self._page_size if limit is None else min(int(limit), self._page_size)
        page = self._rows[start : start + size]
        return {
            "ids": [row["id"] for row in page],
            "documents": [row["text"] for row in page],
            "metadatas": [row["metadata"] for row in page],
            "embeddings": [row["embedding"] for row in page],
        }


def test_snapshot_doc_paginates_and_preserves_metadata_and_embeddings(tmp_path):
    """长文档快照不能只保存第一页，也不能丢页码和向量。"""
    store = VectorStore(persist_dir=str(tmp_path))
    rows = [
        {
            "id": f"snapshot-doc-{index}",
            "text": f"第 {index + 1} 页正文",
            "metadata": {
                "doc_id": "snapshot-doc",
                "kb_id": "default",
                "page": index + 1,
                "page_start": index + 1,
                "content_hash": "same-document",
            },
            "embedding": [float(index), float(index) + 0.5],
        }
        for index in range(5)
    ]
    store._collection = _PagedCollection(rows)
    store._PAGE_SIZE = 2

    snapshot = store.snapshot_doc("snapshot-doc")

    assert snapshot["ids"] == [row["id"] for row in rows]
    assert snapshot["documents"] == [row["text"] for row in rows]
    assert snapshot["metadatas"] == [row["metadata"] for row in rows]
    assert snapshot["embeddings"] == [row["embedding"] for row in rows]


def test_snapshot_restore_round_trip_keeps_each_page_and_embedding(tmp_path):
    """恢复旧版本后，所有分块、页码、元数据和向量都不能被截断或错配。"""
    store = VectorStore(persist_dir=str(tmp_path))
    original_embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.6, 0.8],
    ]
    store.add_chunks(
        embeddings=original_embeddings,
        texts=["第一页", "第二页", "第三页"],
        doc_id="round-trip-doc",
        doc_title="版本一",
        source_type="pdf",
        chunk_indices=[0, 1, 2],
        kb_id="default",
        extra_metadata=[
            {"page": 1, "page_start": 1, "section_path": "第一章"},
            {"page": 2, "page_start": 2, "section_path": "第二章"},
            {"page": 3, "page_start": 3, "section_path": "第三章"},
        ],
    )
    snapshot = store.snapshot_doc("round-trip-doc")

    store.add_chunks(
        embeddings=[[0.9, 0.9, 0.9, 0.9]] * 4,
        texts=["新版本一", "新版本二", "新版本三", "新版本四"],
        doc_id="round-trip-doc",
        doc_title="版本二",
        source_type="md",
        chunk_indices=[0, 1, 2, 3],
        kb_id="default",
    )

    assert store.restore_doc_snapshot(snapshot) == 3
    restored = store.snapshot_doc("round-trip-doc")
    by_id = {
        chunk_id: (text, metadata, embedding)
        for chunk_id, text, metadata, embedding in zip(
            restored["ids"],
            restored["documents"],
            restored["metadatas"],
            restored["embeddings"],
        )
    }
    assert [by_id[f"round-trip-doc-{i}"][0] for i in range(3)] == ["第一页", "第二页", "第三页"]
    assert [by_id[f"round-trip-doc-{i}"][1]["page"] for i in range(3)] == [1, 2, 3]
    for index, expected in enumerate(original_embeddings):
        assert by_id[f"round-trip-doc-{index}"][2] == pytest.approx(expected)


def test_legacy_index_migration_keeps_old_chunk_searchable(tmp_path):
    """没有 kb_id/结构化字段的历史 chunk 迁移后仍能走混合检索。"""
    store = VectorStore(persist_dir=str(tmp_path))
    embedding = FakeEmbedding()
    text = "历史制度编号 LEGACY-2020"
    store._collection.add(
        ids=["legacy-0"],
        embeddings=embedding.embed_texts([text]),
        documents=[text],
        metadatas=[
            {
                "doc_id": "legacy-doc",
                "doc_title": "旧索引制度",
                "source_type": "md",
                "chunk_index": 0,
                "page": 3,
            }
        ],
    )

    assert store.migrate_default_kb_id() == 1
    retriever = HybridRetriever(store, embeddings=embedding, top_k=3)
    hits = retriever.search("LEGACY-2020", kb_id=DEFAULT_KB_ID)

    assert hits
    assert hits[0]["text"] == text
    assert hits[0]["metadata"]["kb_id"] == DEFAULT_KB_ID
    assert hits[0]["metadata"]["page"] == 3


def test_duplicate_content_hash_is_scoped_to_kb(tmp_path):
    """同一内容在不同知识库中都应允许存在，重复判断不能跨库误杀。"""
    store = VectorStore(persist_dir=str(tmp_path))
    for doc_id, kb_id in (("doc-a", "kb-a"), ("doc-b", "kb-b")):
        store.add_chunks(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            texts=["相同制度正文"],
            doc_id=doc_id,
            doc_title=doc_id,
            source_type="md",
            chunk_indices=[0],
            kb_id=kb_id,
            content_hash="same-hash",
        )

    assert store.find_doc_by_hash("same-hash", kb_id="kb-a") == "doc-a"
    assert store.find_doc_by_hash("same-hash", kb_id="kb-b") == "doc-b"
    assert store.find_doc_by_hash("same-hash", kb_id="missing") is None


def test_parent_context_keeps_child_page_citation(tmp_path):
    """生成可看到父块，但来源仍必须是命中的 child 及其页码。"""
    store = VectorStore(persist_dir=str(tmp_path))
    embedding = FakeEmbedding()
    store.add_chunks(
        embeddings=embedding.embed_texts(["父级章节背景", "核心证据：金额超过五万元需复核"]),
        texts=["父级章节背景", "核心证据：金额超过五万元需复核"],
        doc_id="parent-child-doc",
        doc_title="父子制度",
        source_type="pdf",
        chunk_indices=[0, 1],
        kb_id="default",
        extra_metadata=[
            {"chunk_type": "parent", "page_start": 1, "page": 1},
            {
                "chunk_type": "child",
                "parent_id": "parent-child-doc-0",
                "page_start": 7,
                "page": 7,
            },
        ],
    )
    llm = FakeLLM(
        route={
            "客服意图分类器": "kb_question",
            "仅基于以下资料": "根据父级上下文回答。[1]",
            "RAG 质量评估员": "10 10",
        }
    )
    graph = build_qa_graph(
        llm=llm,
        embeddings=embedding,
        vector_store=store,
        top_k=1,
        recall_k=2,
        metadata_filters={"chunk_type": "child"},
    )

    result = run_qa(
        graph,
        question="金额超过五万元需复核",
        kb_id="default",
    )

    generation_prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    assert "父级章节背景" in generation_prompt
    assert result["citations"][0]["chunk_id"] == "parent-child-doc-1"
    assert result["citations"][0]["page"] == 7
    assert result["citations"][0]["text"] == "核心证据：金额超过五万元需复核"
