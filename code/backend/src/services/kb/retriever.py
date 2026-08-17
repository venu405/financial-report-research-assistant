"""混合检索：向量检索 + BM25 关键词检索 → RRF 融合。

为什么需要混合检索？（面试必讲）
  - 纯向量检索的盲区：关键词精确匹配。比如查"5万元"这种数字/专有名词，
    向量空间里可能找不到精确匹配的 chunk，而 BM25 能直接命中。
  - 纯 BM25 的盲区：同义改写（"招投标" vs "公开招标"）命中不了，向量能兜住。
  - 两者互补 → RRF（Reciprocal Rank Fusion）按排名融合，不依赖分数尺度对齐。

RRF 公式：score(d) = Σ_retriever 1 / (k + rank_retriever(d))，k=60 是常见默认值
  - 只比较"排名"，不比较原始分数 → 向量分数(0-1)和 BM25 分数(无界)可以公平融合

多知识库隔离（P3）：
  - 向量检索带 where={"kb_id":xxx} 过滤
  - BM25 索引按 kb_id 分别构建与缓存——绝不把别的库的 chunk 建进本库索引
"""
from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from services.kb.embeddings import EmbeddingClient
from services.kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 常数（论文推荐 60）
# 中文分词简化：按非字母数字切分 + 过滤单字符/停用词
_STOPWORDS = {
    "的", "了", "和", "与", "或", "在", "是", "有", "为", "对", "把", "被",
    "这", "那", "个", "等", "及", "并", "而", "从", "到", "于", "之", "其",
    "公司", "我们", "你们", "他们", "以及", "关于", "进行", "一个", "如何",
}


def _tokenize(text: str) -> list[str]:
    """简易中文分词：按非字母数字切分 + 过滤停用词与单字。"""
    parts = re.split(r"[^\w\u4e00-\u9fff]+", text.lower())
    tokens = []
    for p in parts:
        if not p:
            continue
        # 中文按 2-gram 拆（弥补未用分词器）：如 "招投标" → ["招投","投标"]
        if re.fullmatch(r"[\u4e00-\u9fff]+", p) and len(p) > 1:
            tokens.extend([p[i : i + 2] for i in range(len(p) - 1)])
            tokens.append(p)  # 整词也保留（长词匹配更精准）
        else:
            tokens.append(p)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class HybridRetriever:
    """向量 + BM25 混合检索器。BM25 索引按知识库分别构建与缓存（P3）。"""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embeddings: EmbeddingClient,
        top_k: int = 5,
    ):
        self._store = vector_store
        self._embeddings = embeddings
        self._top_k = top_k
        # 按知识库分别缓存 BM25 索引（多库隔离，避免跨库串数据）
        self._bm25: dict[str, BM25Okapi | None] = {}
        self._corpus: dict[str, list[dict[str, Any]]] = {}
        # P2-1：记录每个库上次构建时的写序号（mutation_seq），替代 count() 全表扫描
        self._seq_snapshot: dict[str, int] = {}

    def _rebuild_if_needed(self, kb_id: str) -> None:
        """写序号变化后重建该库 BM25 索引（内存 seq 比对，跳过 count 全表扫）。

        局限：seq 是全局的，其它库写入也会触发本库重建（多 worker 下以本地写为准）。
        相比每次 ask 全表 get ids，重建代价仍小得多，且绝对正确。
        """
        current_seq = self._store.mutation_seq
        if kb_id not in self._bm25 or current_seq != self._seq_snapshot.get(kb_id):
            corpus = self._store.all_items(kb_id=kb_id)
            tokenized = [_tokenize(c["text"]) for c in corpus]
            self._bm25[kb_id] = BM25Okapi(tokenized) if tokenized else None
            self._corpus[kb_id] = corpus
            self._seq_snapshot[kb_id] = current_seq
            logger.info("BM25 索引重建（kb=%s）：%d 个分块", kb_id, len(corpus))

    def search(
        self, query: str, *, top_k: int | None = None, kb_id: str = "default"
    ) -> list[dict[str, Any]]:
        """混合检索：向量 Top-N + BM25 Top-N → RRF 融合排序。kb_id 限定检索范围。"""
        k = top_k or self._top_k
        self._rebuild_if_needed(kb_id)

        # ---------- 1. 向量检索 Top-K（带 kb_id 过滤）----------
        qvec = self._embeddings.embed_query(query)
        vec_hits = self._store.search(qvec, top_k=max(k * 2, 10), kb_id=kb_id)
        vec_ranks = {h["chunk_id"]: i for i, h in enumerate(vec_hits)}

        # ---------- 2. BM25 检索 Top-K（在该库的索引上）----------
        bm25_ranks: dict[str, int] = {}
        bm25 = self._bm25.get(kb_id)
        corpus = self._corpus.get(kb_id, [])
        if bm25 and corpus:
            tokens = _tokenize(query)
            if tokens:
                scores = bm25.get_scores(tokens)
                # 按分数排序取 Top-N
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for rank, idx in enumerate(order[: max(k * 2, 10)]):
                    cid = corpus[idx]["chunk_id"]
                    if scores[idx] > 0:  # 零分（无命中）不参与融合
                        bm25_ranks[cid] = rank

        # ---------- 3. RRF 融合 ----------
        fused: dict[str, float] = {}
        for cid, rank in vec_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for cid, rank in bm25_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        # 按融合分排序
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

        # 补全完整信息（从向量结果 / corpus 中取）
        vec_by_id = {h["chunk_id"]: h for h in vec_hits}
        corpus_by_id = {c["chunk_id"]: c for c in corpus}
        results = []
        for cid, score in ordered:
            if cid in vec_by_id:
                item = dict(vec_by_id[cid])
            elif cid in corpus_by_id:
                item = {
                    "chunk_id": cid,
                    "text": corpus_by_id[cid]["text"],
                    "metadata": corpus_by_id[cid]["metadata"],
                    "distance": None,
                    "score": 0.0,
                }
            else:
                continue
            item["rrf_score"] = round(score, 4)  # 融合分（调试/展示用）
            results.append(item)
        return results


class VectorOnlyRetriever:
    """纯向量检索（对照组）：接口与 HybridRetriever 一致，用于对比测试。"""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embeddings: EmbeddingClient,
        top_k: int = 5,
    ):
        self._store = vector_store
        self._embeddings = embeddings
        self._top_k = top_k

    def search(
        self, query: str, *, top_k: int | None = None, kb_id: str = "default"
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k
        qvec = self._embeddings.embed_query(query)
        return self._store.search(qvec, top_k=k, kb_id=kb_id)
