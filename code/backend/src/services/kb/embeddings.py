"""Embedding 客户端：本地 Ollama BGE-M3（唯一支持模式）。

模式（bge_m3 · 本地）：Ollama 部署 BGE-M3，数据不出境
  - 调用方式：Ollama HTTP API（/api/embed）
  - 地址：OLLAMA_HOST（默认 127.0.0.1:11434）

⚠️ 智谱云端模式（KB_EMBEDDING_MODE=zhipu）**未实现**：EmbeddingClient 只支持
Ollama 原生 /api/embed 端点，不支持 OpenAI 兼容的 /v1/embeddings + Bearer 鉴权。
配置 zhipu 会启动失败，请勿使用；如需云端 embedding，另实现 OpenAI 兼容客户端。

接口统一：embed_texts(texts) / embed_query(text)
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """BGE-M3 本地 embedding 客户端（Ollama 后端，仅支持 /api/embed）。"""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "bge-m3"):
        self._base_url = base_url.rstrip("/")
        self._model = model
        # 共享 httpx 客户端：连接复用，大文档 N 批不再 N 次 TCP 握手
        self._client = httpx.Client(timeout=120.0)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """批量向量化。内部按 batch_size 分批 + 指数退避重试。

        大文档（500 页 PDF → 上千 chunk）一次塞进单请求会超时/OOM，
        分批避免单次 payload 过大；重试兜底网络抖动/Ollama 瞬时不可用。
        """
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_batch_with_retry(batch))
        return all_embeddings

    def _embed_batch_with_retry(
        self, batch: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """单批 embedding + 指数退避重试（1s/2s/4s）。

        数量校验：Ollama 会静默丢弃空串/异常输入，若返回向量数 != 请求数，
        会导致 Chroma 里文本与向量错位（灾难性静默错误），必须抛异常。
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                # Ollama 返回 {"embeddings": [[...], [...]]}
                embeddings = resp.json().get("embeddings", [])
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"embedding 数量不匹配：请求 {len(batch)} 条，返回 {len(embeddings)} 条"
                    )
                return embeddings
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "embedding 批次失败（第 %d 次），%ds 后重试: %s",
                        attempt + 1, wait, exc,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"embedding 重试 {max_retries} 次仍失败: {last_exc}")

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""
        result = self.embed_texts([text])
        if not result:
            raise RuntimeError(f"Embedding 返回为空（model={self._model})")
        return result[0]

    def ping(self) -> bool:
        """探活：embedding 后端是否可达（短超时，不实际推理）。

        Ollama 用 /api/tags（列模型，零 GPU 开销）；失败/超时返回 False。
        /readyz 就绪探针用——embedding 挂了 ask 会 500，必须提前探出来。
        """
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False
