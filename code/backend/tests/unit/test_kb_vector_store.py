"""Chroma 封装测试：kb_id 隔离 / 迁移 / 列表 / 删除。"""
from __future__ import annotations

from services.kb.vector_store import DEFAULT_KB_ID, VectorStore


def _add(store: VectorStore, doc_id: str, text: str, kb_id: str) -> None:
    store.add_chunks(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        texts=[text],
        doc_id=doc_id,
        doc_title=text[:6],
        source_type="md",
        chunk_indices=[0],
        kb_id=kb_id,
    )


def test_kb_id_isolation(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    _add(store, "d1", "采购制度内容", "default")
    _add(store, "d2", "产品手册内容", "product")

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5, kb_id="default")
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == "default"

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5, kb_id="product")
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == "product"

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)  # 不传 kb_id = 全库
    assert len(hits) == 2


def test_migrate_default_kb_id(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    # 模拟历史数据：绕过封装直接写无 kb_id 的 chunk
    store._collection.add(
        ids=["old-0"],
        embeddings=[[0.5, 0.5, 0.5, 0.5]],
        documents=["历史分块"],
        metadatas=[{"doc_id": "old", "doc_title": "旧文档", "source_type": "md", "chunk_index": 0}],
    )
    migrated = store.migrate_default_kb_id()
    assert migrated == 1
    hits = store.search([0.5, 0.5, 0.5, 0.5], top_k=5, kb_id=DEFAULT_KB_ID)
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == DEFAULT_KB_ID


def test_list_docs_filter_and_delete(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    _add(store, "d1", "文档一", "default")
    _add(store, "d2", "文档二", "product")

    docs_default, total_default = store.list_docs(kb_id="default")
    assert [d["doc_id"] for d in docs_default] == ["d1"]
    assert total_default == 1

    assert store.list_kbs() == ["default", "product"]

    deleted = store.delete_doc("d1")
    assert deleted == 1
    docs, total = store.list_docs(kb_id="default")
    assert docs == [] and total == 0


def test_list_docs_pagination(tmp_path):
    """分页：limit/offset 聚合后切片，total 为全量。"""
    store = VectorStore(persist_dir=str(tmp_path))
    for i in range(5):
        _add(store, f"d{i}", f"文档{i}", "default")

    page1, total = store.list_docs(kb_id="default", limit=2, offset=0)
    page2, _ = store.list_docs(kb_id="default", limit=2, offset=2)
    assert total == 5
    assert [d["doc_id"] for d in page1] == ["d0", "d1"]
    assert [d["doc_id"] for d in page2] == ["d2", "d3"]


def test_upsert_replaces_same_doc(tmp_path):
    """同一 doc_id 重新入库：upsert 覆盖同索引分块，且 delete_doc 只删该 doc。"""
    store = VectorStore(persist_dir=str(tmp_path))
    _add(store, "d1", "第一版", "default")
    _add(store, "d1", "第二版", "default")  # 同 doc_id 同索引 0 → 覆盖
    docs, _ = store.list_docs(kb_id="default")
    assert [d for d in docs if d["doc_id"] == "d1"][0]["chunks"] == 1


def test_snapshot_and_restore_document_version(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    store.add_chunks(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        texts=["第一版"],
        doc_id="d1",
        doc_title="第一版",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="default",
        extra_metadata=[{"page": 2, "page_start": 2, "section_path": "第二章"}],
    )
    snapshot = store.snapshot_doc("d1")
    assert snapshot["metadatas"][0]["page_start"] == 2
    assert snapshot["metadatas"][0]["section_path"] == "第二章"

    store.add_chunks(
        embeddings=[[0.9, 0.8, 0.7, 0.6], [0.6, 0.7, 0.8, 0.9]],
        texts=["第二版一", "第二版二"],
        doc_id="d1",
        doc_title="第二版",
        source_type="md",
        chunk_indices=[0, 1],
        kb_id="default",
    )
    assert set(store.get_doc_ids("d1")) == {"d1-0", "d1-1"}

    assert store.restore_doc_snapshot(snapshot) == 1
    restored = store.snapshot_doc("d1")
    assert restored["ids"] == ["d1-0"]
    assert restored["documents"] == ["第一版"]
    assert restored["metadatas"][0]["page"] == 2
    assert restored["metadatas"][0]["section_path"] == "第二章"


def test_extra_metadata_and_related_chunks_are_kb_isolated(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    store.add_chunks(
        embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
        texts=["父上下文", "子块一", "子块二"],
        doc_id="rel1",
        doc_title="关系文档",
        source_type="md",
        chunk_indices=[0, 1, 2],
        kb_id="hr",
        extra_metadata=[
            {"chunk_type": "parent", "parent_id": "", "section_path": "采购 > 审批"},
            {"chunk_type": "child", "parent_id": "rel1-0", "page_start": 2},
            {"chunk_type": "child", "parent_id": "rel1-0", "page_start": 2, "labels": ["a", "b"]},
        ],
    )
    store.add_chunks(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        texts=["别的知识库"],
        doc_id="other",
        doc_title="别的文档",
        source_type="md",
        chunk_indices=[0],
        kb_id="other",
        extra_metadata=[{"chunk_type": "parent"}],
    )

    related = store.get_related_chunks(chunk_id="rel1-1", kb_id="hr")
    assert [item["chunk_id"] for item in related] == ["rel1-0", "rel1-1", "rel1-2"]
    assert related[2]["metadata"]["labels"] == "['a', 'b']"
    assert store.get_related_chunks(chunk_id="rel1-1", kb_id="other") == []
    assert store.get_chunk_by_id("rel1-1", kb_id="hr")["metadata"]["kb_id"] == "hr"


def test_extra_metadata_requires_one_entry_per_chunk(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    try:
        store.add_chunks(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            texts=["一块"],
            doc_id="bad",
            doc_title="坏数据",
            source_type="md",
            chunk_indices=[0],
            extra_metadata=[],
        )
    except ValueError as exc:
        assert "extra_metadata" in str(exc)
    else:
        raise AssertionError("数量不一致的 extra_metadata 应显式失败")
