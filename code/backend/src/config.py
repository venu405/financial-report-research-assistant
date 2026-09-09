"""企业知识库与智能客服配置。"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Configuration(BaseModel):
    """从环境变量加载知识库、问答模型和管理端配置。"""

    kb_chroma_dir: str = Field(default="./chroma_data")
    kb_collection: str = Field(default="enterprise_kb")
    kb_vector_backend: Literal["chroma", "qdrant"] = Field(default="chroma")
    kb_qdrant_url: str = Field(default="http://127.0.0.1:6333")
    kb_qdrant_collection: str = Field(default="enterprise_kb")
    kb_qdrant_api_key: str | None = Field(default=None)
    kb_qdrant_vector_size: int = Field(default=1024, gt=0)
    kb_qdrant_timeout: float = Field(default=10, gt=0)
    kb_qdrant_create_if_missing: bool = Field(default=False)
    kb_embedding_model: str = Field(default="bge-m3")
    kb_embedding_mode: str = Field(default="bge_m3")
    kb_ollama_host: str = Field(default="http://127.0.0.1:11434")
    kb_chunk_size: int = Field(default=800, ge=100)
    kb_chunk_overlap: int = Field(default=100, ge=0)
    kb_chunk_profile: Literal["legacy", "structured"] = Field(default="structured")
    kb_top_k: int = Field(default=5, ge=1)
    # RAG V2：先扩大召回，再由重排器筛到 kb_top_k。
    kb_recall_k: int = Field(default=20, ge=1)
    kb_max_hits_per_doc: int = Field(default=3, ge=1)
    kb_min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    kb_rerank_mode: str = Field(default="llm")
    kb_rerank_model: str = Field(default="BAAI/bge-reranker-base")
    # 父子切片保留更完整的章节上下文；功能可通过环境变量关闭以便回滚。
    kb_parent_child_enabled: bool = Field(default=False)
    kb_parent_chunk_size: int = Field(default=1600, ge=200)
    kb_neighbor_expansion: int = Field(default=1, ge=0, le=3)
    app_env: str = Field(default="development")

    cors_origins: str = Field(
        default=(
            "http://localhost:5173,http://localhost:5174,http://localhost:3000,"
            "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000"
        )
    )
    admin_api_key: str = Field(
        default="",
        description="管理接口使用的 X-API-Key；生产环境必须配置",
    )

    llm_api_key: str | None = Field(default=None)
    llm_base_url: str | None = Field(default=None)
    llm_model_id: str | None = Field(default=None)
    llm_reasoning_effort: Literal["none", "low", "medium", "high"] | None = Field(
        default=None,
        description="可选的 OpenAI 兼容 reasoning_effort；未设置时保持默认请求形状。",
    )

    @field_validator("llm_reasoning_effort", mode="before")
    @classmethod
    def _normalize_llm_reasoning_effort(cls, value: Any) -> Any:
        """规范化本地兼容接口的思考强度，空字符串等同于未设置。"""
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized not in {"none", "low", "medium", "high"}:
            raise ValueError(
                "LLM_REASONING_EFFORT must be one of: none, low, medium, high"
            )
        return normalized

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "Configuration":
        """读取同名大写环境变量，并允许调用方覆盖指定字段。"""

        values: dict[str, Any] = {}
        for field_name in cls.model_fields:
            value = os.getenv(field_name.upper())
            if value is not None:
                values[field_name] = value
        if overrides:
            values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
