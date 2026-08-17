"""Embedding 客户端：支持两种模式（配置切换）。

模式 1（bge_m3 · 推荐）：本地 Ollama 部署 BGE-M3，数据不出境
  - 调用方式：Ollama HTTP API（/api/embed）
  - 地址：OLLAMA_HOST（默认 127.0.0.1:11434）

模式 2（zhipu · 备选）：智谱 API（OpenAI 兼容端点）
  - 保留实现：若本地模型不可用或想对比效果可切回
  - 通过 .env 的 KB_EMBEDDING_MODE 切换

接口统一：embed_texts(texts) / embed_query(text)
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """BGE-M3 本地 embedding 客户端（Ollama 后端）。"""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "bge-m3"):
        self._base_url = base_url.rstrip("/")
        self._model = model

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
        """单批 embedding + 指数退避重试（1s/2s/4s）。"""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = httpx.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": batch},
                    timeout=120.0,  # 本地推理可能慢，放宽超时
                )
                resp.raise_for_status()
                # Ollama 返回 {"embeddings": [[...], [...]]}
                return resp.json().get("embeddings", [])
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
