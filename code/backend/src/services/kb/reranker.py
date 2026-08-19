"""Rerank 重排层：混合检索召回后、生成前，对候选 chunk 做精排。

为什么需要 rerank？（面试必讲）
  - 召回（向量 + BM25 + RRF）是"双塔"式粗排：快、召回率高，但只看
    query 与 chunk 的表面/独立相关性，不懂交叉语义。
  - Rerank 是"交互式"精排：把 (query, chunk) 拼一起打分，能判断
    "chunk 是否真的回答了这个问题"，精度显著高于纯向量相似度。
  - 标准两段式架构：粗排（recall_k=20）保召回 -> 精排（top_k=5）保精度。

模式（KB_RERANK_MODE）：
  - llm（默认）：用现有 LLM（DeepSeek）一次调用给所有候选打分 0-10。
    零新增依赖/服务；解析失败自动降级为原始 RRF 顺序（绝不阻断问答）。
  - crossencoder：本地交叉编码器（fastembed TextCrossEncoder + bge-reranker，
    onnxruntime 推理，数据不出境）。需 `pip install fastembed` 且首次联网下载
    模型（国内可配 HF_ENDPOINT=https://hf-mirror.com）。未安装时启动告警并
    降级为 off。
  - off：不重排（对照组，测消融用）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# LLM 重排时每个候选喂给模型的文本长度（截断控 token）
_RERANK_SNIPPET_LEN = 200


class Reranker(Protocol):
    """重排器接口：rerank(query, hits, top_n) -> 按相关性降序的 hits 子集。"""

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]: ...

    @property
    def mode(self) -> str: ...


class NoopReranker:
    """不重排：直接截断 top_n（对照组 / rerank 降级兜底）。"""

    mode = "off"

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        return hits[:top_n]


class LLMReranker:
    """LLM 重排：一次调用给所有候选打分（0-10），按分排序取 top_n。

    单次调用而非逐条打分：候选通常 10-20 条，一次 prompt 排完，
    成本约几百 token、延迟一次往返，比 N 次调用便宜且快。

    容错（P0 原则--增强组件失败绝不阻断问答）：
      - LLM 异常 / 解析不出任何有效分数 -> 按原 RRF 顺序截断返回，只告警。
      - 部分候选无分数 -> 无分的排有分的后面（保持原相对顺序）。
    """

    mode = "llm"

    def __init__(self, llm, *, model: str = "deepseek-chat", timeout: float = 30.0):
        self._llm = llm
        self._model = model
        self._timeout = timeout

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        numbered = []
        for i, h in enumerate(hits):
            snippet = (h.get("text") or "").strip().replace("\n", " ")
            numbered.append(f"[{i}] {snippet[:_RERANK_SNIPPET_LEN]}")
        prompt = (
            "你是检索结果重排器（Reranker）。根据【问题】给每个候选资料的相关性打分：\n"
            "10=直接回答该问题；7-9=高度相关；4-6=部分相关；1-3=同主题但不相关；0=无关。\n\n"
            f"【问题】{query}\n\n【候选资料】\n" + "\n".join(numbered) + "\n\n"
            "只输出打分结果，格式每行一条：`编号:分数`（如 `0:9`），不要解释。"
        )
        try:
            scores = self._invoke_scorer(prompt)
        except Exception as exc:
            logger.warning("LLM 重排失败，降级为原始顺序: %s", exc)
            return hits[:top_n]

        for idx, s in scores.items():
            if 0 <= idx < len(hits):
                hits[idx]["rerank_score"] = s
        # 有分在前（按分降序，同分保持原顺序=稳定排序），无分在后
        scored = [(i, h.get("rerank_score")) for i, h in enumerate(hits)]
        scored.sort(key=lambda t: (-(t[1] if t[1] is not None else -1), t[0]))
        ordered = [hits[i] for i, _ in scored]
        if scores:
            n_scored = sum(1 for h in ordered if h.get("rerank_score") is not None)
            logger.info("LLM 重排完成：%d/%d 个候选有分数", n_scored, len(ordered))
        return ordered[:top_n]

    def _invoke_scorer(self, prompt: str) -> dict[int, int]:
        """调 LLM 并解析 `编号:分数` 行。与 qa_graph._llm_invoke 同样的双客户端兼容。"""
        if hasattr(self._llm, "invoke"):
            raw = str(self._llm.invoke([{"role": "user", "content": prompt}])).strip()
        else:
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                timeout=self._timeout,
            )
            raw = (resp.choices[0].message.content or "").strip()
        scores: dict[int, int] = {}
        for m in re.finditer(r"\[(\d+)\]\s*[:：]\s*(\d+)", raw):
            idx, s = int(m.group(1)), min(int(m.group(2)), 10)
            scores[idx] = s
        # 兼容不带方括号的 "0:9" 格式
        if not scores:
            for m in re.finditer(r"^(\d+)\s*[:：]\s*(\d+)\s*$", raw, re.M):
                idx, s = int(m.group(1)), min(int(m.group(2)), 10)
                scores[idx] = s
        return scores


class CrossEncoderReranker:
    """本地交叉编码器重排（fastembed + bge-reranker，onnxruntime 推理）。

    懒加载模型（首次 rerank 才下载/初始化，避免拖慢启动）。
    分数归一化：fastembed 输出原始 logit（无界），用 sigmoid 映射到 0-1，
    写入 hit["rerank_norm"]；rerank_score（0-10）= norm*10，与 LLM 模式对齐。
    """

    mode = "crossencoder"

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name
        self._model = None  # 懒加载

    def _ensure_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("加载 cross-encoder 重排模型：%s", self._model_name)
            self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        try:
            model = self._ensure_model()
            pairs = [(query, (h.get("text") or "")[:512]) for h in hits]
            raw_scores = list(model.rerank(pairs))
        except Exception as exc:
            logger.warning("cross-encoder 重排失败，降级为原始顺序: %s", exc)
            return hits[:top_n]

        import math

        for h, raw in zip(hits, raw_scores):
            norm = 1.0 / (1.0 + math.exp(-raw))  # sigmoid -> 0-1
            h["rerank_norm"] = round(norm, 4)
            h["rerank_score"] = round(norm * 10, 1)
        ordered = sorted(hits, key=lambda h: h.get("rerank_score", 0.0), reverse=True)
        return ordered[:top_n]


def build_reranker(mode: str, *, llm=None, model: str = "deepseek-chat") -> Reranker:
    """工厂：按 KB_RERANK_MODE 构建重排器。非法值/缺依赖统一降级并告警。"""
    if mode == "llm":
        if llm is None:
            logger.warning("rerank mode=llm 但未提供 LLM 客户端，降级为 off")
            return NoopReranker()
        return LLMReranker(llm, model=model)
    if mode == "crossencoder":
        try:
            import fastembed  # noqa: F401

            return CrossEncoderReranker()
        except ImportError:
            logger.warning(
                "KB_RERANK_MODE=crossencoder 需要 fastembed（pip install fastembed），"
                "未安装，降级为 off"
            )
            return NoopReranker()
    if mode != "off":
        logger.warning("未知 KB_RERANK_MODE=%s，降级为 off", mode)
    return NoopReranker()
