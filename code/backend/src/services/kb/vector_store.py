"""Chroma 向量库封装：存 / 查 / 删 + 多知识库隔离。

设计要点：
  1. 薄封装——上层只接触 add/search/delete 等方法，不感知 Chroma 细节
  2. metadata 里存 doc_id + kb_id，支持"按文档删除"和"按知识库隔离"
  3. 多知识库隔离用 metadata kb_id + where 过滤（单 collection 方案，P3）：
     - 比每库一 collection 更灵活（支持跨库检索 where={"kb_id":{"$in":[...]}}）
     - 迁移到 Qdrant 时接口不变
     - 权限就是 kb_id 过滤的自然延伸——检索层强制 where，绝不在生成后补救
  4. 后续换 Qdrant 时，只需替换本模块实现（接口不变）
"""
from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_KB_ID = "default"


class VectorStore:
    """Chroma 持久化向量库（本地目录模式）。单 collection + kb_id 隔离。"""

    def __init__(
        self,
        *,
        persist_dir: str,
        collection_name: str = "enterprise_kb",
        embedding_model: str = "",
    ):
        # Chroma 1.x：持久化客户端直接指定 path
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_model = embedding_model
        # 新库创建时把 embedding_model 写进 collection metadata；旧库读回校验
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )
        self._collection_name = collection_name
        # P0：embedding 模型一致性校验——切换模型会静默污染已有库（维度不匹配
        # 或语义错乱），启动时发现不一致即告警（不阻断，但绝不静默）。
        self._check_embedding_model()
        # 🟡12：迁移标记——实例存活期间已跑过 migrate_default_kb_id 就不再全表扫
        self._migration_done = False
        # P2-1/P1 复核：写操作自增序号，**按 kb_id 维护**——retriever 比对它判断
        # 该库 BM25 缓存是否失效。之前是全局 seq（任何库写都触发所有库重建），
        # 改为按库后只有本库写入才触发本库重建。
        self._mutation_seq: dict[str, int] = {}
        self._seq_lock = Lock()

    def _check_embedding_model(self) -> None:
        """启动校验：collection 已记录的 embedding 模型与当前配置是否一致。"""
        if not self._embedding_model:
            return
        stored = (self._collection.metadata or {}).get("embedding_model", "")
        if not stored:
            logger.warning(
                "Chroma collection 未记录 embedding_model（旧库），请确认当前模型 %s 与历史一致",
                self._embedding_model,
            )
        elif stored != self._embedding_model:
            logger.warning(
                "⚠️ embedding 模型不一致：库用 %s 建，当前配 %s——维度/语义可能不匹配，"
                "检索结果可能错误。请清库重 embedding 或改回原模型。",
                stored, self._embedding_model,
            )

    def _bump_seq(self, kb_id: str | None) -> None:
        """写操作后自增该 kb 的序号。kb_id 为 None 时用全局桶（兜底）。"""
        key = kb_id or "__global__"
        with self._seq_lock:
            self._mutation_seq[key] = self._mutation_seq.get(key, 0) + 1

    def mutation_seq(self, kb_id: str | None) -> int:
        """该 kb 的写序号（retriever 缓存比对用）。"""
        key = kb_id or "__global__"
        with self._seq_lock:
            return self._mutation_seq.get(key, 0)

    @staticmethod
    def _merge_where(
        where: dict[str, Any] | None, kb_id: str | None
    ) -> dict[str, Any] | None:
        """把 kb_id 合并进 where 过滤条件（AND 语义）。

        检索层强制带 kb_id 是权限隔离的底线——绝不在生成后补救。
        """
        if kb_id is None:
            return where
        cond = {"kb_id": kb_id}
        if not where:
            return cond
        # 已有 where → 用 $and 组合（Chroma 支持 $and / $or）
        return {"$and": [where, cond]}

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
    ) -> list[str]:
        """批量写入分块。返回生成的 chunk_id 列表（供引用溯源）。

        content_hash（P1 复核）：文件内容 SHA-256，写进 metadata 供去重查询。
        """
        if not embeddings:
            return []

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for i in range(len(embeddings)):
            cid = f"{doc_id}-{chunk_indices[i]}"
            ids.append(cid)
            meta = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source_type": source_type,
                "chunk_index": chunk_indices[i],
                "kb_id": kb_id,
            }
            if content_hash:
                meta["content_hash"] = content_hash
            metadatas.append(meta)

        # 用 upsert（id 幂等）：新建无冲突，文档更新时同 id 直接覆盖，
        # 且单次调用原子——更新失败不会留下半写状态（P0-2 原子更新依赖这一点）。
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        self._bump_seq(kb_id)
        logger.info("Chroma 写入 %d 个分块（doc=%s, kb=%s）", len(ids), doc_id, kb_id)
        return ids

    def find_doc_by_hash(self, content_hash: str) -> str | None:
        """按内容 hash 查已存在的 doc_id（去重用）。找不到返回 None。"""
        if not content_hash:
            return None
        result = self._collection.get(
            where={"content_hash": content_hash}, include=["metadatas"], limit=1
        )
        for meta in result.get("metadatas", []) or []:
            if meta and meta.get("doc_id"):
                return meta["doc_id"]
        return None

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。返回带分数与元数据的结果列表。

        kb_id：知识库隔离/权限过滤——检索层强制限定范围。
        where：额外的元数据过滤（与 kb_id AND 组合）。
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        merged = self._merge_where(where, kb_id)
        if merged:
            kwargs["where"] = merged

        result = self._collection.query(**kwargs)
        # 展平返回（Chroma 返回嵌套列表）
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        items = []
        for i, doc in enumerate(docs):
            items.append(
                {
                    "chunk_id": ids[i] if i < len(ids) else "",
                    "text": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                    # 余弦距离越小越相似，转成 0-1 相似度分数便于展示
                    "score": 1 - dists[i] if i < len(dists) else 0,
                }
            )
        return items

    def get_doc_ids(self, doc_id: str) -> list[str]:
        """查文档全部 chunk_id（原子更新时算"需要清理的旧块"用）。"""
        result = self._collection.get(where={"doc_id": doc_id}, include=[])
        return result.get("ids", [])

    def delete_chunk_ids(self, ids: list[str], kb_id: str | None = None) -> int:
        """按 chunk_id 列表删除（更新时清理旧块用）。返回删除数量。

        kb_id 已知时传入（精确 bump 该库 seq）；未知（None）用全局桶兜底。
        """
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        self._bump_seq(kb_id)
        logger.info("Chroma 删除 %d 个分块", len(ids))
        return len(ids)

    def delete_kb(self, kb_id: str) -> int:
        """删除某知识库的全部 chunk（P2：删库元数据时联动清理 Chroma，避免"复活"旧数据）。"""
        result = self._collection.get(where={"kb_id": kb_id}, include=[])
        count = len(result.get("ids", []) or [])
        if count:
            self._collection.delete(where={"kb_id": kb_id})
            self._bump_seq(kb_id)
        logger.info("删除知识库 %s 的 %d 个分块", kb_id, count)
        return count

    def delete_doc(self, doc_id: str) -> int:
        """按文档删除全部分块（文档删除时用）。返回删除数量。

        按 doc_id 删天然跨库安全——doc_id 全局唯一。
        """
        # 🟠11：只取 ids，不拉 documents/metadatas（默认 include 会带回全部内容）
        ids = self.get_doc_ids(doc_id)
        # 先查所属库，精确 bump 该库 seq（P1 复核：按库维护写序号）
        return self.delete_chunk_ids(ids, kb_id=self.get_doc_kb_id(doc_id))

    def get_doc_kb_id(self, doc_id: str) -> str | None:
        """查文档所属知识库（删除前鉴权用）。找不到返回 None。"""
        result = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        for meta in result.get("metadatas", []) or []:
            if meta and "kb_id" in meta:
                return meta["kb_id"]
        return None

    def count(self, *, kb_id: str | None = None) -> int:
        """分块总数。kb_id 指定时只数该库（BM25 重建检测用）。

        🟠11：include=[] 只取 ids——默认 get 会连 documents/metadatas 一起拉回，
        chunk 上万后每次 ask 的全表扫描开销翻倍。
        """
        if kb_id is None:
            return self._collection.count()
        result = self._collection.get(where={"kb_id": kb_id}, include=[])
        return len(result.get("ids", []) or [])

    def all_items(
        self,
        *,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """取分块（含文本与元数据）——BM25 索引构建用。可按 kb_id 过滤。"""
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        merged = self._merge_where(where, kb_id)
        if merged:
            kwargs["where"] = merged
        result = self._collection.get(**kwargs)

        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        items = []
        for i, doc in enumerate(docs):
            items.append(
                {
                    "chunk_id": ids[i] if i < len(ids) else "",
                    "text": doc or "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return items

    def list_docs(
        self,
        *,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """列出文档元信息（按 doc_id 聚合），支持分页。返回 (docs, total)。

        kb_id 指定时只列该库。limit/offset 在聚合后切片（响应体大小可控）；
        超大库（>10 万 chunk）的深层优化留给索引层，接口契约保持不变。
        """
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        merged = self._merge_where(None, kb_id)
        if merged:
            kwargs["where"] = merged
        result = self._collection.get(**kwargs)

        metas = result.get("metadatas", []) or []
        docs: dict[str, dict[str, Any]] = {}
        for meta in metas:
            if not meta:
                continue
            did = meta.get("doc_id", "")
            if not did:
                continue
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "title": meta.get("doc_title", ""),
                    "source_type": meta.get("source_type", ""),
                    "kb_id": meta.get("kb_id", DEFAULT_KB_ID),
                    "chunks": 0,
                }
            docs[did]["chunks"] += 1
        all_docs = list(docs.values())
        return all_docs[offset : offset + limit], len(all_docs)

    def list_kbs(self) -> list[str]:
        """列出所有出现过的 kb_id（去重）——知识库管理界面用。"""
        result = self._collection.get(include=["metadatas"])
        kbs: set[str] = set()
        for meta in (result.get("metadatas") or []):
            if meta and "kb_id" in meta:
                kbs.add(meta["kb_id"])
        return sorted(kbs)

    def migrate_default_kb_id(self, kb_id: str = DEFAULT_KB_ID) -> int:
        """给历史无 kb_id 的 chunk 补默认 kb_id（P3 数据迁移）。幂等。

        🟡12：实例级"已迁移"标记——迁移只对旧版本写入的历史数据有意义，
        运行期间新写入的 chunk 必然带 kb_id，故本实例跑过一次后直接跳过，
        避免每次启动（_get_kb）都全表扫描。
        """
        if self._migration_done:
            return 0
        self._migration_done = True  # 无论扫出多少条，本实例不再重复扫

        result = self._collection.get(include=["metadatas"])
        ids = result.get("ids", []) or []
        metas = result.get("metadatas", []) or []
        fix_ids: list[str] = []
        fix_metas: list[dict[str, Any]] = []
        for cid, meta in zip(ids, metas):
            if not meta or "kb_id" not in meta:
                new_meta = dict(meta or {})
                new_meta["kb_id"] = kb_id
                fix_ids.append(cid)
                fix_metas.append(new_meta)
        if fix_ids:
            self._collection.update(ids=fix_ids, metadatas=fix_metas)
            self._bump_seq(kb_id)
            logger.info("迁移：给 %d 个历史 chunk 补 kb_id=%s", len(fix_ids), kb_id)
        return len(fix_ids)
