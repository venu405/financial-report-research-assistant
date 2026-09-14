"""企业知识库与智能客服 FastAPI 入口。"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, Iterator, Literal, Optional

from dotenv import load_dotenv

# 加载 .env 文件（在导入 config 之前，确保环境变量就绪）
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
# 显式进程环境（容器、测试实例、运维注入）优先；.env 只补缺省值。
# 否则多个隔离实例会被 .env 强行指向同一 Chroma 目录，破坏数据隔离。
load_dotenv(_ENV_PATH, override=False)

from fastapi import (  # noqa: E402
    Body,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.security import APIKeyHeader, HTTPBearer  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from config import Configuration  # noqa: E402

# ---- 统一日志：loguru 作为唯一后端，桥接标准 logging ----
# services/* 和 services/kb/* 用的是 logging.getLogger，若不桥接，它们的日志
# （如 "RAG 评估"、"BM25 索引重建"）默认无 handler，实际看不到。这里统一路由到 loguru。


class _InterceptHandler(logging.Handler):
    """把标准 logging 的 record 转发给 loguru，实现两套日志体系统一。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

# 移除 loguru 默认 sink（id=0，默认格式输出到 stderr），
# 否则业务日志会同时走「默认 sink + 自定义 sink」打印两遍。
logger.remove()

# 控制台 handler（level=INFO 会自动涵盖 ERROR 及以上）
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | {extra[request_id]} | <cyan>{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)

# P0：日志落盘——文件 + 轮转（生产排障/审计需要持久化日志，stderr 会随容器重启丢失）
# 默认写到 backend/logs/app.log，10MB 轮转，保留 7 天；KB_LOG_DIR 可覆盖目录。
_LOG_DIR = Path(__file__).resolve().parent.parent / os.getenv("KB_LOG_DIR", "logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    _LOG_DIR / "app.log",
    level=os.getenv("KB_LOG_LEVEL", "INFO"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <4} | {extra[request_id]} | {function} | {file}:{line} | {message}",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,  # 异步写盘，多线程安全
)

# 可选 JSON 结构化日志（KB_LOG_JSON=1 开启）——供 ELK/Loki 采集
if os.getenv("KB_LOG_JSON", "").strip() in ("1", "true", "yes"):
    logger.add(
        _LOG_DIR / "app.jsonl",
        level=os.getenv("KB_LOG_LEVEL", "INFO"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {function} | {message}",
        rotation="50 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        serialize=True,  # 每条日志一个 JSON 对象
    )

# 请求级上下文：request_id（中间件注入），每条日志自动带上，跨模块串联一次请求
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_V1_BEARER_SECURITY = HTTPBearer(auto_error=False)
_V1_API_TOKEN_SECURITY = APIKeyHeader(name="X-Api-Token", auto_error=False)
_V1_SECURITY_RESPONSES = {
    401: {"description": "缺少或无效的 API token"},
    403: {"description": "无权访问指定知识库"},
}
logger = logger.patch(lambda record: record["extra"].setdefault("request_id", _request_id_ctx.get()))


class _SlidingWindowLimiter:
    """进程内滑动窗口限流器，用于保护问答等付费接口。"""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max = max_requests
        self._window = window_seconds
        self._hits: Dict[str, deque] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and now - dq[0] > self._window:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


class KbAskRequest(BaseModel):
    """Payload for knowledge base Q&A (LangGraph orchestrated)."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    kb_id: str = "default"
    thread_id: str | None = None  # P4：对话线程 ID（同 ID 持久化对话状态）
    user_id: str | None = None  # P3 §3.4：用户 ID（传则校验对该 kb 的访问权）
    visitor_token: str | None = None  # 匿名访客的高熵会话凭据（kbv_ 前缀）
    metadata_filters: dict[str, Any] | None = None  # 可选年份/文档类型等检索过滤


class FinancialMetricCreateRequest(BaseModel):
    kb_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    company_code: str = ""
    report_period: str = Field(min_length=1)
    period_type: str
    metric_code: str
    metric_name: str = ""
    raw_value: str | None = None
    raw_unit: str = ""
    normalized_value: str | None = None
    normalized_unit: str = "元"
    statement_scope: str = "unknown"
    source_doc_id: str = ""
    source_title: str = ""
    source_page: int | None = None
    source_page_end: int | None = None
    source_chunk_id: str = ""
    source_text: str = ""
    extraction_status: str = "verified"
    created_by: str | None = None
    user_id: str | None = None


class FinancialMetricPatchRequest(BaseModel):
    raw_value: str | None = None
    raw_unit: str | None = None
    statement_scope: str | None = None
    source_doc_id: str | None = None
    source_title: str | None = None
    source_page: int | None = None
    source_page_end: int | None = None
    source_chunk_id: str | None = None
    source_text: str | None = None
    extraction_status: str | None = None
    reason: str = Field(min_length=1)
    user_id: str | None = None


class ConversationMessageRequest(BaseModel):
    """访客在人工会话中发送消息。"""

    content: str = Field(min_length=1, max_length=4000)
    visitor_token: str | None = None


class V1QueryRequest(BaseModel):
    """企业系统接入层的问答请求（字段名与内部 kb 路由解耦）。"""

    knowledge_base_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=8000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    thread_id: str | None = Field(default=None, max_length=128)
    metadata_filters: dict[str, Any] | None = None
    user_id: str | None = Field(default=None, max_length=128)

    @field_validator("history")
    @classmethod
    def _validate_history(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        for item in value:
            if set(item) - {"role", "content"}:
                raise ValueError("history 仅允许 role 和 content 字段")
            if item.get("role") not in {"user", "assistant", "system"}:
                raise ValueError("history.role 不合法")
            if not item.get("content", "").strip() or len(item["content"]) > 4000:
                raise ValueError("history.content 长度必须为 1-4000")
        return value

    @field_validator("knowledge_base_id", "question")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value

    @field_validator("metadata_filters")
    @classmethod
    def _validate_metadata_filters(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        allowed = {
            "source_type", "page_start", "page_end", "year", "report_period",
            "doc_id", "doc_title", "document_type", "company_name", "chunk_type",
        }
        if set(value) - allowed:
            raise ValueError("metadata_filters 包含不允许的字段")
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata_filters 必须是 JSON 数据") from exc
        if len(encoded.encode("utf-8")) > 4096:
            raise ValueError("metadata_filters 不能超过 4096 字节")
        return value


class V1CompanyAnalysisRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    company_name: str = Field(min_length=1, max_length=256)
    report_period: str = Field(min_length=1, max_length=64)
    comparison_period: str | None = Field(default=None, max_length=64)
    user_id: str | None = Field(default=None, max_length=128)

    @field_validator("knowledge_base_id", "company_name", "report_period")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value


class V1PeerComparisonRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    company_names: list[str] = Field(min_length=2, max_length=3)
    report_period: str = Field(min_length=1, max_length=64)
    metric_codes: list[str] | None = Field(default=None, max_length=20)
    user_id: str | None = Field(default=None, max_length=128)

    @field_validator("company_names")
    @classmethod
    def _validate_companies(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 256 for item in normalized):
            raise ValueError("company_names 不能包含空值且单项不超过 256 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("company_names 不能重复")
        return normalized

    @field_validator("metric_codes")
    @classmethod
    def _validate_metric_codes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 64 for item in normalized):
            raise ValueError("metric_codes 不能包含空值且单项不超过 64 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("metric_codes 不能重复")
        return normalized

    @field_validator("knowledge_base_id", "report_period")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value


class PeerComparisonBriefRequest(BaseModel):
    """同业简报只接收筛选条件，财务数值由服务端重新计算。"""

    kb_id: str = Field(default="default", min_length=1, max_length=64)
    company_names: list[str] = Field(min_length=2, max_length=3)
    report_period: str = Field(min_length=1, max_length=64)
    metric_codes: list[str] | None = Field(default=None, max_length=20)
    user_id: str | None = Field(default=None, max_length=128)

    @field_validator("company_names")
    @classmethod
    def _validate_companies(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 256 for item in normalized):
            raise ValueError("company_names 不能包含空值且单项不超过 256 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("company_names 不能重复")
        return normalized

    @field_validator("metric_codes")
    @classmethod
    def _validate_metric_codes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 64 for item in normalized):
            raise ValueError("metric_codes 不能包含空值且单项不超过 64 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("metric_codes 不能重复")
        return normalized

    @field_validator("kb_id", "report_period")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value


class V1ReportExportRequest(BaseModel):
    """导出统一返回 JSON，避免企业系统必须处理两种响应协议。"""

    report_type: Literal["company_analysis", "peer_comparison"]
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    company_name: str | None = Field(default=None, max_length=256)
    report_period: str = Field(min_length=1, max_length=64)
    comparison_period: str | None = Field(default=None, max_length=64)
    company_names: list[str] | None = Field(default=None, min_length=2, max_length=3)
    metric_codes: list[str] | None = Field(default=None, max_length=20)
    user_id: str | None = Field(default=None, max_length=128)

    @field_validator("company_names")
    @classmethod
    def _validate_export_companies(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 256 for item in normalized):
            raise ValueError("company_names 不能包含空值且单项不超过 256 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("company_names 不能重复")
        return normalized

    @field_validator("metric_codes")
    @classmethod
    def _validate_export_metric_codes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 64 for item in normalized):
            raise ValueError("metric_codes 不能包含空值且单项不超过 64 字符")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("metric_codes 不能重复")
        return normalized

    @field_validator("knowledge_base_id", "report_period")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value


class V1QueryResponse(BaseModel):
    answer: str | None = None
    citations: list[Any] = Field(default_factory=list)
    needs_clarification: bool = False
    request_id: str
    knowledge_base_id: str
    model_config = {"extra": "allow"}


class V1MetricsResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    request_id: str
    knowledge_base_id: str
    model_config = {"extra": "allow"}


class V1CompanyAnalysisResponse(BaseModel):
    company: dict[str, Any] = Field(default_factory=dict)
    report_period: str | None = None
    request_id: str
    knowledge_base_id: str
    model_config = {"extra": "allow"}


class V1PeerComparisonResponse(BaseModel):
    companies: list[dict[str, Any]] = Field(default_factory=list)
    report_period: str | None = None
    request_id: str
    knowledge_base_id: str
    model_config = {"extra": "allow"}


class V1ExportResponse(BaseModel):
    report_type: str
    filename: str
    content_type: str
    content: str
    request_id: str
    knowledge_base_id: str
    model_config = {"extra": "allow"}


class V1EventStreamResponse(StreamingResponse):
    """带明确媒体类型的 SSE 响应类，使 OpenAPI 与运行时契约一致。"""

    media_type = "text/event-stream"


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _safe_upload_filename(filename: str | None) -> str:
    """取上传文件的安全基名，避免把客户端路径带入标题或归档路径。"""
    raw = str(filename or "").strip().replace("\\", "/")
    name = Path(raw).name
    return name if name not in {"", ".", ".."} else "document.bin"


def _upload_title_from_filename(filename: str | None) -> str:
    """缺省标题使用原始文件名（去扩展名），绝不使用临时上传路径名。"""
    safe_name = _safe_upload_filename(filename)
    stem = Path(safe_name).stem.strip()
    return stem or safe_name


def _key_matches(provided: Optional[str], expected: str) -> bool:
    """恒定时间字符串比较（防时序攻击），与明文 != 相比不泄露逐字符差异。"""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def _validate_production_security(config: Configuration) -> None:
    """生产模式缺少必要鉴权时拒绝启动。"""
    if config.app_env.strip().lower() not in {"production", "prod"}:
        return
    problems: list[str] = []
    if not config.admin_api_key or config.admin_api_key.startswith("replace-with-"):
        problems.append("ADMIN_API_KEY 必须设置为真实随机密钥")
    require_token = os.getenv("KB_REQUIRE_TOKEN", "").strip().lower()
    if require_token not in {"1", "true", "yes"}:
        problems.append("KB_REQUIRE_TOKEN 必须设为 1")
    backup_key = os.getenv("KB_BACKUP_SIGNING_KEY", "")
    if len(backup_key) < 32:
        problems.append("KB_BACKUP_SIGNING_KEY 必须设置至少 32 位随机密钥")
    require_backup_signature = os.getenv("KB_REQUIRE_BACKUP_SIGNATURE", "").lower()
    if require_backup_signature not in {"1", "true", "yes"}:
        problems.append("KB_REQUIRE_BACKUP_SIGNATURE 必须设为 1")
    if problems:
        raise RuntimeError("生产安全配置不完整：" + "；".join(problems))


def _build_kb_vector_store(config: Configuration) -> Any:
    """按配置构造知识库向量后端，默认路径保持 Chroma 兼容行为。"""
    backend = getattr(config, "kb_vector_backend", "chroma")
    if backend == "qdrant":
        from services.kb.qdrant_vector_store import QdrantVectorStore

        collection_name = getattr(config, "kb_qdrant_collection", "enterprise_kb")
        create_if_missing = bool(
            getattr(config, "kb_qdrant_create_if_missing", False)
        )
        if str(getattr(config, "app_env", "development")).strip().lower() in {
            "test",
            "testing",
        }:
            create_if_missing = False
        logger.info(
            "KB vector backend={} collection={} create_if_missing={}",
            backend,
            collection_name,
            create_if_missing,
        )
        return QdrantVectorStore(
            url=getattr(config, "kb_qdrant_url", "http://127.0.0.1:6333"),
            collection_name=collection_name,
            vector_size=getattr(config, "kb_qdrant_vector_size", 1024),
            api_key=getattr(config, "kb_qdrant_api_key", None),
            timeout=getattr(config, "kb_qdrant_timeout", 10),
            create_if_missing=create_if_missing,
        )
    if backend != "chroma":
        raise ValueError(f"不支持的 KB_VECTOR_BACKEND: {backend}")

    from services.kb.vector_store import VectorStore

    store = VectorStore(
        persist_dir=config.kb_chroma_dir,
        collection_name=config.kb_collection,
        embedding_model=config.kb_embedding_model,
    )
    # P3：历史数据迁移——给无 kb_id 的 chunk 补默认值（幂等，仅首次执行实际写入）
    store.migrate_default_kb_id()
    logger.info(
        "KB vector backend={} collection={}", backend, config.kb_collection
    )
    return store


def _check_kb_vector_store_ready(store: Any) -> None:
    """Use a backend health check when available; retain the legacy fallback."""
    check = getattr(store, "healthcheck", None)
    if not callable(check):
        check = getattr(store, "ping", None)
    if callable(check):
        if check() is False:
            raise RuntimeError("vector store healthcheck returned false")
        return
    store.list_kbs()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        log_startup_configuration()
        start_maintenance_worker()
        try:
            yield
        finally:
            stop_maintenance_worker()

    app = FastAPI(title="企业知识库与智能客服", lifespan=lifespan)

    # CORS：从配置读允许的来源（生产禁用 *；* + credentials 浏览器会拒）
    _cfg = Configuration.from_env()
    _cors_origins = [o.strip() for o in _cfg.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins, allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(?::\d+)?$",
        # 有明确来源才允许 credentials，避免「* + credentials」非法组合
        allow_credentials=bool(_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # P1-3：request_id 注入——优先用调用方传入的 X-Request-Id，否则生成 uuid，
    # 回写响应头。配合 loguru 的 {extra[request_id]}，一次请求的日志可跨模块串联。
    @app.middleware("http")
    async def _inject_request_id(request: Request, call_next):
        candidate = request.headers.get("X-Request-Id", "")
        request_id = candidate if _REQUEST_ID_RE.fullmatch(candidate) else uuid.uuid4().hex
        request.state.request_id = request_id
        context_token = _request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            _request_id_ctx.reset(context_token)

    _admin_key = _cfg.admin_api_key

    # P0：KB 问答限流——/kb/ask 每次都是 LLM 付费调用，不能裸奔。
    # 按身份（user_id 或 anonymous）分桶限流，防单用户刷爆费用。
    # KB_ASK_RATE_LIMIT：每分钟每身份最大请求数（默认 20）。
    _kb_ask_limiter = _SlidingWindowLimiter(
        max_requests=int(os.getenv("KB_ASK_RATE_LIMIT", "20") or 20),
        window_seconds=60,
    )

    def log_startup_configuration() -> None:
        # uvicorn 启动时已用它的 LOGGING_CONFIG 重设过 root logger（加了 default handler），
        # 这里接管 root：只保留 loguru 桥接，移除 uvicorn 默认 handler，避免业务日志重复打印两遍。
        # uvicorn.access / uvicorn.error 是独立 logger（propagate=False），不受影响。
        _root = logging.getLogger()
        _root.handlers = [_InterceptHandler()]
        _root.setLevel(logging.INFO)

        config = Configuration.from_env()
        _validate_production_security(config)

        logger.info(
            "Knowledge base configuration loaded: model=%s llm_base_url=%s "
            "embedding_model=%s embedding_host=%s api_key=%s",
            config.llm_model_id or "deepseek-chat",
            config.llm_base_url or "unset",
            config.kb_embedding_model,
            config.kb_ollama_host,
            _mask_secret(config.llm_api_key),
        )

        # P1-3：安全告警——生产环境必须配置 ADMIN_API_KEY，否则管理接口无 X-API-Key 兜底
        if not config.admin_api_key:
            logger.warning(
                "ADMIN_API_KEY 未配置：/admin/* 与用户管理接口将无 X-API-Key 兜底，"
                "仅依赖 admin token 鉴权。生产环境请设置 ADMIN_API_KEY。"
            )

        # P2-3：打印百度 OCR 后备通道配置状态（masked）——环境变量是唯一配置源，
        # 排障时一眼看到"百度 OCR 是否配了、QPS 上限多少"，不用去翻 .env
        _baidu_key = os.getenv("BAIDU_OCR_API_KEY", "")
        _baidu_secret = os.getenv("BAIDU_OCR_SECRET_KEY", "")
        logger.info(
            "Baidu OCR fallback: configured=%s qps=%s",
            bool(_baidu_key and _baidu_secret),
            os.getenv("KB_BAIDU_OCR_QPS", "2"),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def ready_check() -> Dict[str, Any]:
        """就绪探针：探活下游依赖（SQLite 用户库 / vector store / Ollama embedding）。

        任一依赖不可用返回 503，供 K8s/Docker 就绪判定。轻量级 healthz 保持纯 liveness。
        """
        kb = _get_kb()  # 会初始化 Chroma/embedding（若未初始化）
        checks: dict[str, str] = {}
        try:
            kb["auth"].list_users()
            checks["sqlite_users"] = "ok"
        except Exception as exc:
            checks["sqlite_users"] = f"error: {exc}"
        try:
            _check_kb_vector_store_ready(kb["store"])
            checks["vector_store"] = "ok"
        except Exception as exc:
            checks["vector_store"] = f"error: {exc}"
        # P0：embedding 后端探活——embedding 挂了 ask 会 500，必须提前探出
        try:
            if kb["embeddings"].ping():
                checks["embedding"] = "ok"
            else:
                checks["embedding"] = "error: unreachable"
        except Exception as exc:
            checks["embedding"] = f"error: {exc}"

        ok = all(v == "ok" for v in checks.values())
        if not ok:
            # P1-3 复核修复：就绪探针语义——依赖不可用必须 503，供编排器摘流
            raise HTTPException(status_code=503, detail=f"依赖不可用: {checks}")
        return {"status": "ok", "checks": checks}

    @app.get("/admin/audit")
    def admin_audit(
        request: Request,
        limit: int = Query(default=200, ge=1, le=1000),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """审计日志（最近 N 条，需 X-API-Key 或 admin）。管理界面/合规导出用。"""
        kb = _get_kb()
        # P1-3 复核修复：与 docstring 对齐——X-API-Key 或 admin token 二选一，
        # 不再"未配 ADMIN_API_KEY 即裸开放"
        _require_kb_admin(kb, request.headers.get("X-API-Key"), x_api_token, None)
        return {"total": kb["audit"].count(), "entries": kb["audit"].recent(limit)}

    @app.get("/admin/metrics")
    def admin_metrics(
        request: Request,
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """进程内指标（计数 + 延迟分位数）。需 X-API-Key 或 admin。"""
        from services.kb.metrics import global_metrics

        kb = _get_kb()  # 仅用于身份解析
        _require_kb_admin(kb, request.headers.get("X-API-Key"), x_api_token, None)
        return global_metrics.snapshot()

    @app.get("/admin/search-logs")
    def admin_search_logs(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """检索日志（最近 N 条，需 X-API-Key 或 admin）。客服答错时回放定位用。"""
        kb = _get_kb()
        _require_kb_admin(kb, request.headers.get("X-API-Key"), x_api_token, None)
        return {"total": kb["retrieval_log"].count(), "entries": kb["retrieval_log"].recent(limit)}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        """Prometheus 文本格式指标（标准抓取端点，无鉴权，供内网监控抓取）。"""
        from fastapi.responses import Response

        from services.kb.metrics import global_metrics

        return Response(
            content=global_metrics.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # ==================== 知识库管理（KB）接口 ====================

    # KB 组件按需初始化（首次使用 KB 接口时创建）
    _kb = {}
    _kb_init_lock = Lock()
    _maintenance_stop = Event()
    _maintenance_thread: Thread | None = None

    def _get_kb():
        """懒加载 KB 组件：EmbeddingClient + VectorStore + LangGraph 图。

        加锁 + 双重检查：一旦路由改 async 或多 worker，并发首访会同时初始化
        出双份 Chroma/LLM/checkpointer 连接，这里用锁保证只初始化一次。
        """
        if "ready" in _kb:
            return _kb
        with _kb_init_lock:
            if "ready" in _kb:  # 双重检查：等锁期间可能已被前一个线程初始化完
                return _kb
            from openai import OpenAI

            from services.kb import qa_graph
            from services.kb.embeddings import EmbeddingClient

            cfg = Configuration.from_env()

            # Embedding：仅本地 bge_m3（Ollama /api/embed）。
            # zhipu 云端模式未实现（EmbeddingClient 不支持 OpenAI 兼容端点），
            # 配 KB_EMBEDDING_MODE=zhipu 会启动失败——这里忽略并回退本地 + 告警。
            if cfg.kb_embedding_mode == "zhipu":
                logger.warning(
                    "KB_EMBEDDING_MODE=zhipu 未实现（仅支持本地 bge_m3），已回退本地模式"
                )
            embeddings = EmbeddingClient(
                base_url=cfg.kb_ollama_host,
                model=cfg.kb_embedding_model,
            )

            store = _build_kb_vector_store(cfg)

            # LLM（复用现有配置：DeepSeek）
            # P0：设默认超时——LLM/网络 hang 时不拖死请求（qa_graph 内单次调用也带 timeout）
            llm = OpenAI(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url or None,
                timeout=float(os.getenv("LLM_TIMEOUT", "60") or 60),
            )
            # model 显式传给 build_qa_graph（P1：去掉 _model 私有属性 hack）

            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            from services.kb.auth import AuthStore

            # P4：SQLite checkpointer——对话状态按 thread_id 持久化（断点续跑）
            # 注意：新版 from_conn_string 返回 context manager（需 with），应用生命周期内
            # 直接用 sqlite3 连接实例化（连接常驻，服务存活期间有效）
            checkpoint_path = Path(cfg.kb_chroma_dir).parent / "kb_checkpoints.db"
            _ckpt_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
            _ckpt_conn.execute("PRAGMA journal_mode=WAL")  # P3：并发写不 locked
            _ckpt_conn.execute("PRAGMA busy_timeout=5000")
            saver = SqliteSaver(_ckpt_conn)

            # 客服改造第5项：FAQ 存储（标准问+答案+相似问，向量直答）
            from services.kb.faq_store import FAQStore
            faq_path = Path(cfg.kb_chroma_dir).parent / "kb_faq.db"
            faq_store = FAQStore(faq_path, embeddings)

            # 客服改造第6项：AI 客服人设/话术
            from services.kb.persona_store import PersonaStore
            persona_path = Path(cfg.kb_chroma_dir).parent / "kb_persona.db"
            persona_store = PersonaStore(persona_path)

            # 客服改造第7项：满意度反馈
            from services.kb.feedback_store import FeedbackStore
            feedback_path = Path(cfg.kb_chroma_dir).parent / "kb_feedback.db"
            feedback_store = FeedbackStore(feedback_path)

            # 客服改造第8项：会话与消息（转人工闭环地基）
            from services.kb.conversation_store import ConversationStore
            conv_path = Path(cfg.kb_chroma_dir).parent / "kb_conversations.db"
            conversation_store = ConversationStore(conv_path)

            # 客服改造第10项：轻量工单
            from services.kb.ticket_store import TicketStore
            ticket_path = Path(cfg.kb_chroma_dir).parent / "kb_tickets.db"
            ticket_store = TicketStore(ticket_path)

            # 客服改造第11项：快捷回复
            from services.kb.quick_reply_store import QuickReplyStore
            quick_reply_path = Path(cfg.kb_chroma_dir).parent / "kb_quick_replies.db"
            quick_reply_store = QuickReplyStore(quick_reply_path)

            # 客服改造第15项：知识库元数据
            from services.kb.kb_meta_store import KbMetaStore
            kb_meta_path = Path(cfg.kb_chroma_dir).parent / "kb_meta.db"
            kb_meta_store = KbMetaStore(kb_meta_path)

            # 文档治理、长期记忆/隐私、数据源同步和运行告警
            from services.kb.alert_store import AlertStore
            from services.kb.financial_metric_store import FinancialMetricStore
            from services.kb.governance_store import GovernanceStore
            from services.kb.privacy import PrivacyStore
            from services.kb.source_store import SourceStore

            data_dir = Path(cfg.kb_chroma_dir).parent
            governance_store = GovernanceStore(data_dir / "kb_governance.db")
            privacy_store = PrivacyStore(data_dir / "kb_privacy.db")
            source_store = SourceStore(data_dir / "kb_sources.db")
            alert_store = AlertStore(data_dir / "kb_alerts.db")
            financial_metric_store = FinancialMetricStore(data_dir / "kb_financial_metrics.db")

            # 客服改造第1项：Rerank 重排器（llm / crossencoder / off 三模式）
            from services.kb.reranker import build_reranker
            reranker = build_reranker(
                getattr(cfg, "kb_rerank_mode", "llm"),
                llm=llm,
                model=cfg.llm_model_id or "deepseek-chat",
                crossencoder_model=getattr(
                    cfg, "kb_rerank_model", "BAAI/bge-reranker-base"
                ),
            )

            graph = qa_graph.build_qa_graph(
                llm=llm,
                embeddings=embeddings,
                vector_store=store,
                top_k=cfg.kb_top_k,
                checkpointer=saver,
                model=cfg.llm_model_id or "deepseek-chat",
                reasoning_effort=cfg.llm_reasoning_effort,
                min_score=getattr(cfg, "kb_min_similarity", 0.0),
                faq_store=faq_store,
                faq_threshold=float(os.getenv("KB_FAQ_THRESHOLD", "0.8") or 0.8),
                reranker=reranker,
                persona_store=persona_store,
                recall_k=getattr(cfg, "kb_recall_k", max(cfg.kb_top_k * 4, cfg.kb_top_k)),
                max_candidates_per_doc=getattr(cfg, "kb_max_hits_per_doc", 3),
                metadata_filters=(
                    {"chunk_type": "child"}
                    if getattr(cfg, "kb_parent_child_enabled", False)
                    else None
                ),
                neighbor_expansion=getattr(cfg, "kb_neighbor_expansion", 0),
            )

            # P3 §3.4 / v3 §6.1：RBAC——用户与知识库访问权限（SQLite）
            auth_path = Path(cfg.kb_chroma_dir).parent / "kb_users.db"
            auth = AuthStore(auth_path)

            # P1-2：审计日志（SQLite，写操作留痕）
            from services.kb.audit import AuditStore
            audit_path = Path(cfg.kb_chroma_dir).parent / "kb_audit.db"
            audit = AuditStore(audit_path)

            # 客服改造第3项：检索日志留痕（客服答错时回放定位）
            from services.kb.retrieval_log import RetrievalLogStore
            retrieval_log_path = Path(cfg.kb_chroma_dir).parent / "kb_retrieval_log.db"
            retrieval_log = RetrievalLogStore(retrieval_log_path)

            # P0-1：从环境变量引导首个 admin（KB_BOOTSTRAP_ADMIN_TOKEN，幂等、永不过期）
            # 生产 bootstrap 入口，替代裸 POST /kb/users 建号。引导后应删除该环境变量。
            _bootstrap_token = os.getenv("KB_BOOTSTRAP_ADMIN_TOKEN", "").strip()
            if _bootstrap_token:
                auth.bootstrap_admin("bootstrap-admin", _bootstrap_token)
                logger.warning(
                    "已引导 bootstrap admin（KB_BOOTSTRAP_ADMIN_TOKEN）。"
                    "生产环境请删除该环境变量，改用 X-API-Key 或 admin token。"
                )

            _kb.update(
                {
                    "ready": True,
                    "embeddings": embeddings,
                    "store": store,
                    "graph": graph,
                    "llm": llm,
                    "config": cfg,
                    "auth": auth,
                    "audit": audit,
                    "retrieval_log": retrieval_log,
                    "faq_store": faq_store,
                    "persona_store": persona_store,
                    "feedback_store": feedback_store,
                    "conversation_store": conversation_store,
                    "ticket_store": ticket_store,
                    "quick_reply_store": quick_reply_store,
                    "kb_meta_store": kb_meta_store,
                    "governance_store": governance_store,
                    "privacy_store": privacy_store,
                    "source_store": source_store,
                    "alert_store": alert_store,
                    "financial_metric_store": financial_metric_store,
                    "checkpoint_conn": _ckpt_conn,  # P2：会话管理接口用（列/删 thread）
                }
            )
            return _kb

    # P0：文档级更新锁——并发 PUT 同一 doc_id 串行化。
    # 否则"读旧 ids → 写新 → 删 stale"三步并发时，A 删的 stale 可能误删 B 刚写的新块。
    _doc_locks: dict[str, Lock] = {}
    _doc_locks_guard = Lock()

    def _get_doc_lock(doc_id: str) -> Lock:
        with _doc_locks_guard:
            return _doc_locks.setdefault(doc_id, Lock())

    def _require_kb_access(
        kb: dict, user_id: str | None, kb_id: str, *, required: bool = False
    ) -> None:
        """校验 user_id 对 kb_id 的访问权。

        required=True（写操作：ingest/update/delete）：缺 user_id 直接 401——写操作必须带身份；
          同时校验写权限（readonly 角色只读，写操作 403）。
        required=False（读操作：ask/docs/kbs）：demo 兼容，未传身份不校验。
        admin 角色全通；readonly 角色可读不可写。
        """
        if not user_id:
            if required:
                raise HTTPException(status_code=401, detail="此操作需提供身份（token 或 user_id）")
            if _ENFORCE_KB_VISIBILITY and not kb["kb_meta_store"].is_public(kb_id):
                raise HTTPException(status_code=401, detail="该知识库仅限内部用户访问")
            return
        if required:
            if not kb["auth"].can_write(user_id, kb_id):
                raise HTTPException(
                    status_code=403,
                    detail=f"用户 {user_id} 无知识库 {kb_id} 的写权限（readonly 只读）",
                )
        else:
            # P0-1：读操作对「未注册 user_id」（如挂件匿名访客 visitor-*）当匿名放行，
            # 不 403——否则挂件一开口就被拦。已注册但无权才 403。
            if kb["auth"].get_user(user_id) is None:
                return
            if not kb["auth"].can_access(user_id, kb_id):
                raise HTTPException(
                    status_code=403,
                    detail=f"用户 {user_id} 无权访问知识库 {kb_id}",
                )

    # P0-1：用户/权限管理接口的统一鉴权——X-API-Key（匹配 ADMIN_API_KEY）或
    # 已认证的 admin 用户（token / user_id），二选一。杜绝「裸接口建 admin」。
    # 注意：_key_matches 用模块级定义（hmac.compare_digest 恒定时间比较）。
    def _require_kb_admin(
        kb: dict,
        x_api_key: str | None,
        x_api_token: str | None,
        admin_id: str | None,
    ) -> str:
        """管理员校验——用户/权限管理接口专用。返回操作者身份标识（审计用）。

        - X-API-Key 匹配 ADMIN_API_KEY（恒定时间比较）→ 放行，返回 "api_key:<前缀>"（运维/bootstrap）
        - 否则解析 token/user_id → 必须是 admin 角色，否则 403；返回该 user_id
        - 两者皆无 → 401
        """
        if _admin_key and _key_matches(x_api_key, _admin_key):
            return f"api_key:{(_admin_key or '')[:4]}"
        user_id = _resolve_user_id(kb, x_api_token, admin_id)
        if not user_id:
            raise HTTPException(
                status_code=401, detail="此操作需管理员身份（X-API-Key 或 admin token）"
            )
        if not kb["auth"].is_admin(user_id):
            raise HTTPException(status_code=403, detail=f"用户 {user_id} 无管理员权限")
        return user_id

    def _require_kb_agent(
        kb: dict,
        x_api_key: str | None,
        x_api_token: str | None,
        operator_id: str | None,
    ) -> str:
        """客服工作台鉴权：坐席、主管、管理员或运维密钥。"""
        if _admin_key and _key_matches(x_api_key, _admin_key):
            return f"api_key:{(_admin_key or '')[:4]}"
        user_id = _resolve_user_id(kb, x_api_token, operator_id)
        if not user_id:
            raise HTTPException(status_code=401, detail="此操作需客服身份")
        if not kb["auth"].is_customer_service(user_id):
            raise HTTPException(status_code=403, detail=f"用户 {user_id} 无客服工作台权限")
        return user_id

    def _require_agent_conversation_access(
        kb: dict, operator: str, conv_id: int
    ) -> dict[str, Any]:
        conv = kb["conversation_store"].get(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")
        if not operator.startswith("api_key:") and not kb["auth"].can_access(
            operator, conv["kb_id"]
        ):
            raise HTTPException(status_code=403, detail="无权处理该知识库的会话")
        return conv

    def _require_kb_supervisor(
        kb: dict,
        x_api_key: str | None,
        x_api_token: str | None,
        operator_id: str | None,
    ) -> str:
        operator = _require_kb_agent(kb, x_api_key, x_api_token, operator_id)
        if operator.startswith("api_key:") or kb["auth"].has_role(
            operator, {"supervisor", "admin"}
        ):
            return operator
        raise HTTPException(status_code=403, detail="此操作需客服主管或管理员权限")

    # 🟠4：token 鉴权解析——X-Api-Token 头优先，user_id 直传回退（过渡期兼容）
    _REQUIRE_TOKEN = bool(os.getenv("KB_REQUIRE_TOKEN", "").strip() not in ("", "0", "false"))

    # P0-1：自助注册开关。KB_OPEN_SIGNUP=1 时允许无鉴权建号，但角色强制 member
    # （杜绝 admin 提权）。默认关闭——生产必须走 X-API-Key 或 admin token 建号。
    _OPEN_SIGNUP = bool(os.getenv("KB_OPEN_SIGNUP", "").strip() in ("1", "true", "yes"))

    # 生产环境强制区分公开/内部知识库；开发环境可通过环境变量提前启用验证。
    _ENFORCE_KB_VISIBILITY = getattr(_cfg, "app_env", "development").strip().lower() in {
        "production",
        "prod",
    } or os.getenv("KB_ENFORCE_KB_VISIBILITY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    # P1-1：上传大小上限（MB）。流式写盘，超限 413 并清理临时文件。
    _MAX_UPLOAD_MB = int(os.getenv("KB_MAX_UPLOAD_MB", "50"))

    def _resolve_user_id(
        kb: dict, x_api_token: str | None, user_id: str | None
    ) -> str | None:
        """把请求凭据解析为 user_id。

        优先级：X-Api-Token 头（反查 users.api_token）> user_id 直传（兼容旧客户端）。
        - token 存在但无效 → 401（防止拿假 token 配 user_id 冒充）
        - 仅 user_id 且 KB_REQUIRE_TOKEN=1 → 401（生产强制 token）
        - 两者都无 → None（由 _require_kb_access/_require_kb_admin 决定 401 还是放行）
        """
        if x_api_token:
            user = kb["auth"].get_user_by_token(x_api_token.strip())
            if not user:
                raise HTTPException(status_code=401, detail="无效的 API token")
            return user["user_id"]
        if user_id:
            if _REQUIRE_TOKEN:
                raise HTTPException(
                    status_code=401,
                    detail="本服务要求 token 鉴权（X-Api-Token 头），不接受 user_id 直传",
                )
            logger.warning("user_id 直传已废弃，请改用 X-Api-Token: {}", user_id[:8])
            return user_id
        return None

    def _conversation_owner_key(
        kb: dict,
        user_id: str | None,
        visitor_token: str | None,
        *,
        required: bool = False,
    ) -> str:
        """生成不可反查的会话归属键，防止只凭 thread_id/自增 id 越权读取。"""
        if user_id and kb["auth"].get_user(user_id):
            return f"user:{user_id}"
        raw = (visitor_token or user_id or "").strip()
        if raw:
            if visitor_token and (
                not raw.startswith("kbv_") or len(raw) < 24 or len(raw) > 160
            ):
                raise HTTPException(status_code=401, detail="无效的访客会话凭据")
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return f"visitor:{digest}"
        if required:
            raise HTTPException(status_code=401, detail="此会话需要用户 token 或访客会话凭据")
        return "anonymous"

    def _owned_conversation(
        kb: dict,
        *,
        thread_id: str | None = None,
        conv_id: int | None = None,
        owner_key: str,
    ) -> dict[str, Any]:
        conv = (
            kb["conversation_store"].get_by_thread(thread_id)
            if thread_id is not None
            else kb["conversation_store"].get(conv_id or 0)
        )
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")
        if not hmac.compare_digest(conv.get("visitor_id", ""), owner_key):
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return conv

    def _prepare_chat_turn(
        kb: dict, payload: KbAskRequest, user_id: str | None
    ) -> tuple[dict[str, Any], list[dict[str, str]], str]:
        """建立归属明确的会话、加载服务端历史并记录本轮用户消息。"""
        owner_key = _conversation_owner_key(kb, user_id, payload.visitor_token)
        thread_id = payload.thread_id or uuid.uuid4().hex
        try:
            conv = kb["conversation_store"].get_or_create(
                thread_id,
                kb_id=payload.kb_id,
                visitor_id=owner_key,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if conv["kb_id"] != payload.kb_id:
            raise HTTPException(status_code=409, detail="会话所属知识库与请求不一致")
        if conv["status"] in {"waiting", "human"}:
            raise HTTPException(status_code=409, detail="会话已转人工，请通过人工消息接口继续发送")
        history = kb["conversation_store"].recent_model_history(conv["id"], limit=20)
        if not history:
            history = payload.history[-20:]
        memories = kb["privacy_store"].list_memory(owner_key)
        if memories:
            memory_text = "；".join(f"{item['key']}={item['value']}" for item in memories[:20])
            history = [
                {"role": "user", "content": f"以下是我明确同意保存的长期信息：{memory_text}"},
                {"role": "assistant", "content": "好的，我只在本次回答需要时参考这些信息。"},
                *history,
            ]
        from services.kb.privacy import redact_text

        kb["conversation_store"].add_message(
            conv["id"], "user", redact_text(payload.question)
        )
        return conv, history, thread_id

    def _finish_chat_turn(
        kb: dict,
        conv: dict[str, Any],
        result: dict[str, Any],
        *,
        actor: str,
    ) -> None:
        """保存助手回复，并在需要时把同一个会话转入人工队列。"""
        from services.kb.privacy import redact_text

        kb["conversation_store"].add_message(
            conv["id"], "assistant", redact_text(result.get("answer", ""))
        )
        if result.get("escalate"):
            transferred = kb["conversation_store"].transfer_to_human(
                conv["id"], "可回答性门槛两次不过"
            )
            if transferred:
                kb["audit"].record(
                    user_id=actor,
                    action="transfer_to_human",
                    target=str(conv["id"]),
                )
                agents = kb["auth"].available_agents(conv["kb_id"])
                if agents:
                    selected = min(
                        agents,
                        key=lambda item: kb["conversation_store"].active_load(
                            item["user_id"]
                        ),
                    )
                    if kb["conversation_store"].claim(conv["id"], selected["user_id"]):
                        kb["audit"].record(
                            user_id="system",
                            action="auto_assign_agent",
                            target=str(conv["id"]),
                            detail={"agent_id": selected["user_id"]},
                        )
        refreshed = kb["conversation_store"].get(conv["id"]) or conv
        result["conversation_id"] = conv["id"]
        result["thread_id"] = conv["thread_id"]
        result["status"] = refreshed["status"]

    def _prepare_chunks(
        kb: dict,
        file: UploadFile,
        *,
        doc_id: str,
        title: str | None,
        kb_id: str,
        ocr_mode: str = "local",
    ) -> tuple[list, list, str, str]:
        """解析 + 分块 + 向量化（不写库）。返回 (chunks, vectors, resolved_title, content_hash)。

        P0-2 原子更新的「准备阶段」：任何失败（解析/embedding）都不触碰旧数据，
        调用方（ingest/update）拿到结果后才决定写库。
        content_hash（P1 复核）：文件内容 SHA-256，供写库前去重。
        """
        import hashlib
        import uuid

        from services.kb.ingest import build_chunks, build_legacy_chunks

        safe_filename = _safe_upload_filename(file.filename)
        resolved_title = (title or "").strip() or _upload_title_from_filename(file.filename)
        suffix = Path(safe_filename).suffix
        tmp_path = Path(kb["config"].kb_chroma_dir).parent / f"_upload_{uuid.uuid4().hex}{suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        # P1-1：流式写盘（1MB 分片）+ 大小上限，避免整文件读进内存 / 磁盘写满
        max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
        size = 0
        digest = hashlib.sha256()
        with open(tmp_path, "wb") as _f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    _f.close()
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小限制 {_MAX_UPLOAD_MB}MB",
                    )
                _f.write(chunk)
                digest.update(chunk)
        content_hash = digest.hexdigest()

        cfg = kb["config"]
        try:
            if getattr(cfg, "kb_chunk_profile", "structured") == "legacy":
                chunks = build_legacy_chunks(
                    tmp_path,
                    doc_id=doc_id,
                    chunk_size=cfg.kb_chunk_size,
                    overlap=cfg.kb_chunk_overlap,
                    kb_id=kb_id,
                    ocr_mode=ocr_mode,
                )
            else:
                chunks = build_chunks(
                    tmp_path,
                    doc_id=doc_id,
                    chunk_size=cfg.kb_chunk_size,
                    overlap=cfg.kb_chunk_overlap,
                    kb_id=kb_id,
                    ocr_mode=ocr_mode,
                    document_title=resolved_title,
                    parent_chunk_size=(
                        getattr(cfg, "kb_parent_chunk_size", 1600)
                        if getattr(cfg, "kb_parent_child_enabled", False)
                        else None
                    ),
                    child_chunk_size=(
                        cfg.kb_chunk_size
                        if getattr(cfg, "kb_parent_child_enabled", False)
                        else None
                    ),
                )
            if not chunks:
                raise HTTPException(status_code=400, detail="文档解析后无有效内容")
            for chunk in chunks:
                # build_chunks 使用临时路径解析；在返回前把用户可见标题写回
                # DocumentChunk，避免临时 stem 进入后续快照/测试/兼容调用。
                chunk.doc_title = resolved_title
            # 标题参与向量化（P0）：title 是强信号，拼进 embedding 输入；
            # 存储仍用纯正文（metadata 里已有 doc_title），检索更准且展示不重复
            embed_inputs = [f"{resolved_title}\n{c.text}" for c in chunks]
            vectors = kb["embeddings"].embed_texts(embed_inputs)
            return chunks, vectors, resolved_title, content_hash
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # 清理临时文件
            except Exception as cleanup_exc:  # 清理失败不阻断（残留无害）
                logger.warning("临时文件清理失败（忽略）: {}", cleanup_exc)

    def _archive_original(kb: dict, file: UploadFile, doc_id: str) -> tuple[str, str]:
        """把原始文件保存到受控数据目录，供审计、下载和重新处理。"""
        safe_name = _safe_upload_filename(file.filename)
        suffix = Path(safe_name).suffix
        archive_dir = Path(kb["config"].kb_chroma_dir).parent / "document_originals" / doc_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{uuid.uuid4().hex}{suffix}"
        max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
        size = 0
        file.file.seek(0)
        with target.open("wb") as output:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    output.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"文件超过大小限制 {_MAX_UPLOAD_MB}MB")
                output.write(chunk)
        file.file.seek(0)
        data_dir = Path(kb["config"].kb_chroma_dir).parent.resolve()
        return safe_name, target.resolve().relative_to(data_dir).as_posix()

    def _public_version(version: dict[str, Any]) -> dict[str, Any]:
        """Remove internal snapshot and filesystem details from API responses."""
        result = dict(version)
        result.pop("snapshot", None)
        result.pop("original_path", None)
        return result

    def _version_original_path(kb: dict, stored_path: str) -> Path:
        """Resolve current relative paths and legacy absolute paths safely."""
        data_dir = Path(kb["config"].kb_chroma_dir).parent.resolve()
        candidate = Path(stored_path)
        return candidate.resolve() if candidate.is_absolute() else (data_dir / candidate).resolve()

    def _write_chunks(
        kb: dict,
        *,
        chunks: list,
        vectors: list,
        doc_id: str,
        doc_title: str,
        kb_id: str,
        content_hash: str | None = None,
    ) -> list[str]:
        """写库（单次 upsert 原子）。返回新写入的 chunk_id 列表。"""
        return kb["store"].add_chunks(
            embeddings=vectors,
            texts=[c.text for c in chunks],
            doc_id=doc_id,
            doc_title=doc_title,
            source_type=chunks[0].source_type,
            chunk_indices=[c.chunk_index for c in chunks],
            kb_id=kb_id,
            content_hash=content_hash,
            extra_metadata=[dict(c.metadata) for c in chunks],
        )

    def _snapshot_from_prepared(
        *, chunks: list, vectors: list, doc_id: str, doc_title: str, kb_id: str,
        content_hash: str | None = None,
    ) -> dict[str, Any]:
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            ids.append(f"{doc_id}-{chunk.chunk_index}")
            meta = dict(chunk.metadata)
            meta.update({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "kb_id": kb_id,
            })
            if content_hash:
                meta["content_hash"] = content_hash
            metadatas.append(meta)
        return {
            "ids": ids,
            "documents": [chunk.text for chunk in chunks],
            "metadatas": metadatas,
            "embeddings": vectors,
        }

    def _read_sync_source(source: dict[str, Any]) -> tuple[bytes, str]:
        """读取管理员配置的数据源，并执行目录/域名白名单校验。"""
        import urllib.parse
        import urllib.request

        location = source["location"]
        if source["source_type"] == "file":
            default_root = Path(__file__).resolve().parent.parent / "sync_sources"
            allowed_root = Path(os.getenv("KB_SYNC_ALLOWED_ROOT", str(default_root))).resolve()
            configured_path = Path(location)
            path = (
                configured_path.resolve()
                if configured_path.is_absolute()
                else (allowed_root / configured_path).resolve()
            )
            if allowed_root != path and allowed_root not in path.parents:
                raise ValueError(f"文件数据源必须位于允许目录: {allowed_root}")
            if not path.is_file():
                raise ValueError("数据源文件不存在")
            return path.read_bytes(), path.name
        parsed = urllib.parse.urlparse(location)
        allowed_hosts = {
            host.strip().lower()
            for host in os.getenv("KB_SYNC_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        }
        if parsed.scheme != "https" or not parsed.hostname or parsed.hostname.lower() not in allowed_hosts:
            raise ValueError("HTTP 数据源只允许 HTTPS 且域名必须在 KB_SYNC_ALLOWED_HOSTS 中")
        request = urllib.request.Request(location, headers={"User-Agent": "EnterpriseKB-Sync/1.0"})
        max_bytes = _MAX_UPLOAD_MB * 1024 * 1024

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=20) as response:  # noqa: S310
            final = urllib.parse.urlparse(response.geturl())
            if (
                final.scheme != "https"
                or not final.hostname
                or final.hostname.lower() not in allowed_hosts
            ):
                raise ValueError("数据源重定向到了未授权域名")
            content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("数据源内容超过上传限制")
        filename = Path(parsed.path).name or f"source-{source['id']}.txt"
        return content, filename

    def _run_source_sync(kb: dict, source: dict[str, Any], *, actor: str) -> dict[str, Any]:
        import io

        lease_stop = Event()

        def renew_lease() -> None:
            while not lease_stop.wait(60):
                try:
                    if not kb["source_store"].renew(source["id"]):
                        return
                except Exception as exc:
                    logger.warning("数据源 %s 续租失败，将继续当前任务: %s", source["id"], exc)

        lease_thread = Thread(
            target=renew_lease, name=f"kb-source-lease-{source['id']}", daemon=True
        )
        lease_thread.start()
        try:
            return _run_source_sync_under_lease(kb, source, actor=actor, io_module=io)
        finally:
            lease_stop.set()
            lease_thread.join(timeout=2)

    def _run_source_sync_under_lease(
        kb: dict, source: dict[str, Any], *, actor: str, io_module: Any
    ) -> dict[str, Any]:

        content, filename = _read_sync_source(source)
        digest = hashlib.sha256(content).hexdigest()
        if digest == source.get("last_hash"):
            kb["source_store"].mark_result(source["id"], ok=True, content_hash=digest)
            kb["alert_store"].resolve(f"source-sync-{source['id']}")
            return {"source_id": source["id"], "changed": False, "doc_id": source.get("doc_id", "")}
        doc_id = source.get("doc_id") or uuid.uuid4().hex
        upload = UploadFile(file=io_module.BytesIO(content), filename=filename)
        chunks, vectors, title, content_hash = _prepare_chunks(
            kb, upload, doc_id=doc_id, title=source["name"], kb_id=source["kb_id"]
        )
        original_name, original_path = _archive_original(kb, upload, doc_id)
        with _get_doc_lock(doc_id):
            version = kb["governance_store"].create_version(
                doc_id=doc_id,
                kb_id=source["kb_id"],
                title=title,
                snapshot=_snapshot_from_prepared(
                    chunks=chunks,
                    vectors=vectors,
                    doc_id=doc_id,
                    doc_title=title,
                    kb_id=source["kb_id"],
                    content_hash=content_hash,
                ),
                created_by=actor,
                state="review",
                review_note=f"数据源 {source['id']} 自动同步，等待审核",
                original_name=original_name,
                original_path=original_path,
            )
        kb["source_store"].mark_result(
            source["id"], ok=True, content_hash=digest, doc_id=doc_id
        )
        kb["alert_store"].resolve(f"source-sync-{source['id']}")
        kb["audit"].record(
            user_id=actor, action="sync_source", target=str(source["id"]),
            detail={"doc_id": doc_id, "kb_id": source["kb_id"], "chunks": len(chunks)},
        )
        return {
            "source_id": source["id"], "changed": True, "doc_id": doc_id,
            "chunks": len(chunks), "version_id": version["id"], "status": "pending_review",
        }

    def _scan_operational_alerts(kb: dict) -> None:
        import shutil

        breached_conversations = kb["conversation_store"].sla_breaches()
        conversation_alerts = {f"conversation-sla-{conv['id']}" for conv in breached_conversations}
        for conv in breached_conversations:
            kb["alert_store"].emit(
                dedupe_key=f"conversation-sla-{conv['id']}",
                alert_type="conversation_sla",
                severity="high",
                target=str(conv["id"]),
                message=f"会话 {conv['thread_id']} 已超过首次响应 SLA",
            )
        kb["alert_store"].resolve_missing("conversation-sla-", conversation_alerts)
        overdue_tickets = kb["ticket_store"].overdue()
        ticket_alerts = {f"ticket-overdue-{ticket['id']}" for ticket in overdue_tickets}
        for ticket in overdue_tickets:
            kb["alert_store"].emit(
                dedupe_key=f"ticket-overdue-{ticket['id']}",
                alert_type="ticket_overdue",
                severity="high",
                target=str(ticket["id"]),
                message=f"工单 {ticket['ticket_no']} 已超过处理期限",
            )
        kb["alert_store"].resolve_missing("ticket-overdue-", ticket_alerts)
        rag = kb["retrieval_log"].stats()
        if rag["total"] >= 10 and float(rag["avg_faithfulness"]) < 7:
            kb["alert_store"].emit(
                dedupe_key="rag-low-faithfulness", alert_type="rag_quality", severity="high",
                target="rag", message=f"RAG 平均忠实度降至 {rag['avg_faithfulness']}",
            )
        else:
            kb["alert_store"].resolve("rag-low-faithfulness")
        feedback = kb["feedback_store"].stats()
        if feedback["total"] >= 10 and float(feedback["satisfaction_rate"]) < 0.8:
            kb["alert_store"].emit(
                dedupe_key="feedback-low-satisfaction", alert_type="customer_satisfaction",
                severity="medium", target="feedback",
                message=f"满意率降至 {float(feedback['satisfaction_rate']) * 100:.1f}%",
            )
        else:
            kb["alert_store"].resolve("feedback-low-satisfaction")
        waiting = kb["conversation_store"].stats()["waiting"]
        waiting_limit = max(1, int(os.getenv("KB_ALERT_WAITING_COUNT", "20")))
        if waiting >= waiting_limit:
            kb["alert_store"].emit(
                dedupe_key="queue-backlog", alert_type="queue_backlog", severity="high",
                target="waiting", message=f"人工队列积压 {waiting} 条会话",
            )
        else:
            kb["alert_store"].resolve("queue-backlog")
        data_dir = Path(kb["config"].kb_chroma_dir).parent
        free_gb = shutil.disk_usage(data_dir).free / (1024**3)
        min_free_gb = float(os.getenv("KB_ALERT_MIN_FREE_GB", "2"))
        if free_gb < min_free_gb:
            kb["alert_store"].emit(
                dedupe_key="disk-low", alert_type="disk", severity="critical",
                target=str(data_dir), message=f"数据盘剩余空间仅 {free_gb:.2f}GB",
            )
        else:
            kb["alert_store"].resolve("disk-low")

    @app.post("/kb/ingest")
    def kb_ingest(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        kb_id: str = Form(default="default"),
        user_id: str | None = Form(default=None),
        ocr_mode: Literal["local", "baidu", "auto"] = Form(default="local"),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """上传文档入库：解析 → 分块 → 向量化 → 写入 Chroma（指定知识库）。

        ocr_mode：图片/扫描 PDF 的 OCR 模式。local=本地（默认，零费用）；baidu=强制
        百度手写识别（难图/潦草字用，需配 BAIDU_OCR_API_KEY/SECRET_KEY）；auto=本地
        优先、空/低置信兜底到百度。
        鉴权：X-Api-Token 头优先，user_id 直传兼容（KB_REQUIRE_TOKEN=1 时仅认 token）。
        """
        try:
            import uuid

            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, user_id)
            _require_kb_access(kb, user_id, kb_id, required=True)
            doc_id = uuid.uuid4().hex
            chunks, vectors, resolved_title, content_hash = _prepare_chunks(
                kb, file, doc_id=doc_id, title=title, kb_id=kb_id, ocr_mode=ocr_mode
            )
            # P1 复核：内容去重——同 hash 已存在则拒绝重复入库，返回已有 doc_id
            dup_doc = kb["store"].find_doc_by_hash(content_hash, kb_id=kb_id)
            if dup_doc:
                raise HTTPException(
                    status_code=409,
                    detail=f"文档内容已存在（doc_id={dup_doc}），请勿重复上传",
                )
            original_name, original_path = _archive_original(kb, file, doc_id)
            _write_chunks(
                kb, chunks=chunks, vectors=vectors, doc_id=doc_id,
                doc_title=resolved_title, kb_id=kb_id, content_hash=content_hash,
            )
            kb["governance_store"].create_version(
                doc_id=doc_id,
                kb_id=kb_id,
                title=resolved_title,
                snapshot=kb["store"].snapshot_doc(doc_id),
                created_by=user_id or "anonymous",
                state="published",
                original_name=original_name,
                original_path=original_path,
            )
            kb["audit"].record(
                user_id=user_id or "anonymous", action="ingest", target=doc_id,
                detail={"kb_id": kb_id, "chunks": len(chunks), "title": resolved_title},
            )
            return {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "title": resolved_title,
                "kb_id": kb_id,
            }
        except HTTPException:
            raise
        except ValueError as exc:  # 不支持的文件类型 / 无扩展名等：参数问题，映射 400
            raise HTTPException(status_code=400, detail=f"入库失败: {exc}") from exc
        except Exception as exc:
            logger.error("KB ingest failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc

    @app.post("/kb/ask")
    def kb_ask(
        payload: KbAskRequest = Body(...), x_api_token: str | None = Header(default=None)
    ) -> Dict[str, Any]:
        """基于知识库问答（LangGraph 编排）。

        权限（P3 §3.4）：X-Api-Token 头优先解析身份，无则 user_id 兼容；校验对该 kb_id 的访问权。
        限流（P0）：按身份分桶限流，防刷爆 LLM 付费调用（KB_ASK_RATE_LIMIT）。
        """
        try:
            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, payload.user_id)
            # RBAC：越权在检索前拦截（绝不在生成后补救）
            _require_kb_access(kb, user_id, payload.kb_id)
            # 限流：按 user_id（未认证用 anonymous 共用桶）
            if not _kb_ask_limiter.allow(user_id or "anonymous"):
                raise HTTPException(status_code=429, detail="问答请求过于频繁，请稍后再试")
            conv, history, thread_id = _prepare_chat_turn(kb, payload, user_id)
            from services.kb import qa_graph

            start = time.time()
            result = qa_graph.run_qa(
                kb["graph"],
                question=payload.question,
                history=history,
                kb_id=payload.kb_id,
                metadata_filters=payload.metadata_filters,
                thread_id=thread_id,
            )
            # 客服改造第3项：检索日志留痕（答错时回放定位）
            from services.kb.privacy import redact_text

            meta = result.get("search_meta", {})
            kb["retrieval_log"].record(
                kb_id=payload.kb_id,
                thread_id=thread_id,
                question=redact_text(payload.question),
                rewritten=redact_text(meta.get("rewritten", "")),
                rerank_mode=os.getenv("KB_RERANK_MODE", "llm"),
                answerable=not meta.get("escalate", False),
                evidence_score=meta.get("top_score", 0.0),
                escalate=meta.get("escalate", False),
                attempts=meta.get("attempts", 1),
                latency_ms={"total": (time.time() - start) * 1000.0},
                hits=meta.get("recall_raw", []),
                final_hits=meta.get("contexts", []),
                faithfulness=result.get("score", 0),
                answer=redact_text(result.get("answer", "")),
            )
            _finish_chat_turn(kb, conv, result, actor=user_id or "anonymous")
            return result
        except HTTPException:
            raise  # 403 等 HTTP 异常直接抛出，不被转 500
        except Exception as exc:
            logger.exception("KB ask failed")
            raise HTTPException(status_code=500, detail="问答服务暂时不可用") from exc

    @app.get("/kb/conversation/{conv_id}/status")
    def kb_conversation_status(
        conv_id: int,
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """查当前用户拥有的会话状态。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(kb, user_id, visitor_token, required=True)
        conv = _owned_conversation(kb, conv_id=conv_id, owner_key=owner_key)
        return {"conversation_id": conv_id, "status": conv["status"], "agent_id": conv["agent_id"]}

    @app.get("/kb/conversations/{thread_id}/messages")
    def kb_conversation_messages(
        thread_id: str,
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """恢复当前用户拥有的完整会话消息。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(kb, user_id, visitor_token, required=True)
        conv = _owned_conversation(kb, thread_id=thread_id, owner_key=owner_key)
        _require_kb_access(kb, user_id, conv["kb_id"])
        return {
            "conversation": conv,
            "messages": kb["conversation_store"].list_messages(conv["id"]),
        }

    @app.get("/kb/conversations")
    def kb_list_owned_conversations(
        kb_id: str | None = Query(default=None),
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> Dict[str, Any]:
        """列出当前登录用户或匿名访客自己的会话。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(kb, user_id, visitor_token, required=True)
        conversations = kb["conversation_store"].list_for_owner(owner_key, limit=limit)
        if kb_id:
            _require_kb_access(kb, user_id, kb_id)
            conversations = [c for c in conversations if c["kb_id"] == kb_id]
        return {"conversations": conversations}

    @app.delete("/kb/conversations/{thread_id}")
    def kb_delete_owned_conversation(
        thread_id: str,
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """删除当前用户自己的会话消息和模型检查点。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(kb, user_id, visitor_token, required=True)
        _owned_conversation(kb, thread_id=thread_id, owner_key=owner_key)
        removed = kb["conversation_store"].delete_owned(thread_id, owner_key)
        conn = kb["checkpoint_conn"]
        checkpoint_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('checkpoints','writes')"
            ).fetchall()
        }
        for table in checkpoint_tables:
            conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
        conn.commit()
        return {"thread_id": thread_id, "deleted": removed}

    @app.post("/kb/conversations/{thread_id}/messages")
    def kb_send_human_message(
        thread_id: str,
        payload: ConversationMessageRequest = Body(...),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """访客在等待或人工服务期间继续发送消息。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(
            kb, user_id, payload.visitor_token, required=True
        )
        conv = _owned_conversation(kb, thread_id=thread_id, owner_key=owner_key)
        _require_kb_access(kb, user_id, conv["kb_id"])
        if conv["status"] not in {"waiting", "human"}:
            raise HTTPException(status_code=409, detail="该会话当前不在人工服务中")
        from services.kb.privacy import redact_text

        safe_content = redact_text(payload.content.strip())
        message_id = kb["conversation_store"].add_message(
            conv["id"], "user", safe_content
        )
        return {
            "conversation": kb["conversation_store"].get(conv["id"]),
            "message": {
                "id": message_id,
                "role": "user",
                "content": safe_content,
            },
        }

    @app.post("/kb/conversations/{thread_id}/close")
    def kb_close_owned_human_conversation(
        thread_id: str,
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """访客主动结束自己的人工会话，保留历史记录并返回智能客服。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, None)
        owner_key = _conversation_owner_key(kb, user_id, visitor_token, required=True)
        conv = _owned_conversation(kb, thread_id=thread_id, owner_key=owner_key)
        _require_kb_access(kb, user_id, conv["kb_id"])
        if conv["status"] not in {"waiting", "human", "closed"}:
            raise HTTPException(status_code=409, detail="该会话当前不在人工服务中")
        if conv["status"] != "closed":
            kb["conversation_store"].close(conv["id"])
            kb["audit"].record(
                user_id=owner_key,
                action="visitor_close_conversation",
                target=str(conv["id"]),
            )
        return {"thread_id": thread_id, "closed": True}

    @app.post("/kb/ask/stream")
    def kb_ask_stream(
        payload: KbAskRequest = Body(...), x_api_token: str | None = Header(default=None)
    ) -> StreamingResponse:
        """流式问答（SSE）：按节点推送进度，前端实时反馈（P2）。

        鉴权/限流与 /kb/ask 一致。事件格式：data: {"type":"node","node":...}
        最后一条 type="final" 携带完整答案与引用。
        """
        try:
            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, payload.user_id)
            _require_kb_access(kb, user_id, payload.kb_id)
            if not _kb_ask_limiter.allow(user_id or "anonymous"):
                raise HTTPException(status_code=429, detail="问答请求过于频繁，请稍后再试")
            conv, history, thread_id = _prepare_chat_turn(kb, payload, user_id)
        except HTTPException:
            raise

        from services.kb import qa_graph
        stream_request_id = _request_id_ctx.get()

        def event_iterator() -> Iterator[str]:
            start = time.time()
            final_event: Dict[str, Any] | None = None
            try:
                for event in qa_graph.run_qa_stream(
                    kb["graph"],
                    question=payload.question,
                    history=history,
                    kb_id=payload.kb_id,
                    metadata_filters=payload.metadata_filters,
                    thread_id=thread_id,
                ):
                    if event.get("type") == "final":
                        # P0-2：final 先缓冲不 yield，等会话善后回填 conversation_id/status 再发
                        final_event = dict(event)
                        final_event["request_id"] = stream_request_id
                        continue
                    event_payload = dict(event)
                    event_payload["request_id"] = stream_request_id
                    yield f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"
                # P1：final 之后补齐与同步 /kb/ask 一致的检索日志落库 + escalate 转人工闭环
                if final_event is not None:
                    from services.kb.privacy import redact_text

                    meta = final_event.get("search_meta", {})
                    kb["retrieval_log"].record(
                        kb_id=payload.kb_id,
                        thread_id=thread_id,
                        question=redact_text(payload.question),
                        rewritten=redact_text(meta.get("rewritten", "")),
                        rerank_mode=os.getenv("KB_RERANK_MODE", "llm"),
                        answerable=not meta.get("escalate", False),
                        evidence_score=meta.get("top_score", 0.0),
                        escalate=meta.get("escalate", False),
                        attempts=meta.get("attempts", 1),
                        latency_ms={"total": (time.time() - start) * 1000.0},
                        hits=meta.get("recall_raw", []),
                        final_hits=meta.get("contexts", []),
                        faithfulness=final_event.get("score", 0),
                        answer=redact_text(final_event.get("answer", "")),
                    )
                    _finish_chat_turn(
                        kb, conv, final_event, actor=user_id or "anonymous"
                    )
                    # 善后完成后才 yield final
                    yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("KB ask stream failed")
                yield f"data: {json.dumps({'type': 'error', 'code': 'INTERNAL_ERROR', 'message': '流式问答服务暂时不可用', 'request_id': stream_request_id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/kb/financial-metrics")
    def list_financial_metrics(
        kb_id: str = Query(default="default", min_length=1),
        company_name: str | None = Query(default=None),
        report_period: str | None = Query(default=None),
        metric_code: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, kb_id)
        try:
            items, total = kb["financial_metric_store"].list(
                kb_id=kb_id,
                company_name=company_name,
                report_period=report_period,
                metric_code=metric_code,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get("/kb/company-analysis")
    def get_company_analysis(
        kb_id: str = Query(default="default", min_length=1),
        company_name: str = Query(..., min_length=1),
        report_period: str = Query(..., min_length=1),
        comparison_period: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        from services.kb.company_analysis import analyze_company

        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, kb_id)
        try:
            return analyze_company(
                kb["financial_metric_store"],
                kb_id=kb_id,
                company_name=company_name,
                report_period=report_period,
                comparison_period=comparison_period,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/kb/peer-comparison")
    def get_peer_comparison(
        company_names: list[str] = Query(..., min_length=2, max_length=3),
        report_period: str = Query(..., min_length=1),
        kb_id: str = Query(default="default", min_length=1),
        metric_codes: list[str] | None = Query(default=None),
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        from services.kb.peer_comparison import compare_companies

        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, kb_id)
        try:
            return compare_companies(
                kb["financial_metric_store"],
                kb_id=kb_id,
                company_names=company_names,
                report_period=report_period,
                metric_codes=metric_codes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/kb/peer-comparison/brief")
    def create_peer_comparison_brief(
        payload: PeerComparisonBriefRequest,
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        from services.kb.peer_comparison import compare_companies
        from services.kb.peer_comparison_brief import generate_peer_comparison_brief

        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, payload.user_id)
        _require_kb_access(kb, resolved_user, payload.kb_id)
        try:
            comparison = compare_companies(
                kb["financial_metric_store"],
                kb_id=payload.kb_id,
                company_names=payload.company_names,
                report_period=payload.report_period,
                metric_codes=payload.metric_codes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return generate_peer_comparison_brief(
            comparison,
            llm=kb["llm"],
            model=kb["config"].llm_model_id or "deepseek-chat",
            reasoning_effort=kb["config"].llm_reasoning_effort,
        )

    @app.get("/kb/company-analysis/export")
    def export_company_analysis(
        kb_id: str = Query(default="default", min_length=1),
        company_name: str = Query(..., min_length=1),
        report_period: str = Query(..., min_length=1),
        comparison_period: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Response:
        from services.kb.company_analysis import analyze_company
        from services.kb.research_export import render_company_analysis

        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, kb_id)
        try:
            result = analyze_company(
                kb["financial_metric_store"],
                kb_id=kb_id,
                company_name=company_name,
                report_period=report_period,
                comparison_period=comparison_period,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=render_company_analysis(result, kb_id=kb_id),
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="company-analysis-export.md"'},
        )

    @app.get("/kb/peer-comparison/export")
    def export_peer_comparison(
        company_names: list[str] = Query(..., min_length=2, max_length=3),
        report_period: str = Query(..., min_length=1),
        kb_id: str = Query(default="default", min_length=1),
        metric_codes: list[str] | None = Query(default=None),
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Response:
        from services.kb.peer_comparison import compare_companies
        from services.kb.research_export import render_peer_comparison

        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, kb_id)
        try:
            result = compare_companies(
                kb["financial_metric_store"],
                kb_id=kb_id,
                company_names=company_names,
                report_period=report_period,
                metric_codes=metric_codes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=render_peer_comparison(result, kb_id=kb_id),
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="peer-comparison-export.md"'},
        )

    @app.post("/kb/financial-metrics")
    def create_financial_metric(
        payload: FinancialMetricCreateRequest,
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        resolved_user = _resolve_user_id(kb, x_api_token, payload.user_id)
        _require_kb_access(kb, resolved_user, payload.kb_id, required=True)
        data = payload.model_dump(exclude={"user_id", "created_by", "normalized_value", "normalized_unit"})
        data["created_by"] = resolved_user or ""
        try:
            item = kb["financial_metric_store"].create(data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"item": item}

    # ==================== 企业系统接入层（v1） ====================
    # 该层只负责协议适配：保留 /kb/* 作为内部/兼容接口，企业系统使用稳定的
    # knowledge_base_id 字段和统一的 request_id 响应。业务权限仍由原有 RBAC 校验。

    def _v1_auth_token(
        x_api_token: str | None, authorization: str | None
    ) -> str:
        """解析双凭据；v1 始终要求 token，避免退回 user_id/匿名鉴权。"""
        header_token: str | None = None
        bearer_token: str | None = None
        if x_api_token is not None:
            candidate = x_api_token.strip()
            if not candidate or candidate != x_api_token or any(ch.isspace() for ch in candidate):
                raise HTTPException(status_code=401, detail="X-Api-Token 格式无效")
            header_token = candidate
        if authorization is not None:
            parts = authorization.split(" ")
            if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
                raise HTTPException(status_code=401, detail="Authorization 必须使用 Bearer token")
            if any(ch.isspace() for ch in parts[1]):
                raise HTTPException(status_code=401, detail="Authorization token 格式无效")
            bearer_token = parts[1]
        if header_token and bearer_token:
            if not hmac.compare_digest(header_token, bearer_token):
                raise HTTPException(status_code=401, detail="两个凭据不一致")
            return header_token
        if header_token or bearer_token:
            token = header_token or bearer_token
            assert token is not None
            return token
        raise HTTPException(status_code=401, detail="v1 接口必须提供 API token")

    def _v1_response(
        payload: dict[str, Any], knowledge_base_id: str
    ) -> dict[str, Any]:
        result = dict(payload)
        result["knowledge_base_id"] = knowledge_base_id
        result["request_id"] = _request_id_ctx.get()
        return result

    def _v1_safe_query_response(
        payload: dict[str, Any], knowledge_base_id: str
    ) -> dict[str, Any]:
        """仅向企业系统暴露问答业务字段，隐藏检索/评估内部调试数据。"""
        allowed = {
            "answer", "citations", "sources", "status", "thread_id",
            "conversation_id", "escalate", "intent", "faq_hit", "needs_clarification",
        }
        return _v1_response(
            {key: value for key, value in payload.items() if key in allowed},
            knowledge_base_id,
        )

    def _v1_query_param(
        value: str | None, name: str, max_length: int
    ) -> str | None:
        if value is None:
            return None
        if not value.strip() or len(value) > max_length:
            raise HTTPException(status_code=422, detail=f"{name} 不能为空且不超过 {max_length} 字符")
        return value

    @app.post(
        "/api/v1/query", response_model=V1QueryResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    @app.post("/api/v1/queries", include_in_schema=False)
    def v1_query(
        payload: V1QueryRequest = Body(...),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> V1QueryResponse:
        token = _v1_auth_token(x_api_token, authorization)
        internal = KbAskRequest(
            question=payload.question,
            history=payload.history,
            kb_id=payload.knowledge_base_id,
            thread_id=payload.thread_id,
            user_id=payload.user_id,
            metadata_filters=payload.metadata_filters,
        )
        return _v1_safe_query_response(kb_ask(internal, x_api_token=token), payload.knowledge_base_id)

    @app.post(
        "/api/v1/query/stream", response_class=V1EventStreamResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    @app.post("/api/v1/queries/stream", include_in_schema=False)
    def v1_query_stream(
        payload: V1QueryRequest = Body(...),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> StreamingResponse:
        token = _v1_auth_token(x_api_token, authorization)
        internal = KbAskRequest(
            question=payload.question,
            history=payload.history,
            kb_id=payload.knowledge_base_id,
            thread_id=payload.thread_id,
            user_id=payload.user_id,
            metadata_filters=payload.metadata_filters,
        )
        response = kb_ask_stream(internal, x_api_token=token)
        response.headers["X-Request-Id"] = _request_id_ctx.get()
        return response

    @app.get(
        "/api/v1/financial-metrics", response_model=V1MetricsResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    def v1_financial_metrics(
        knowledge_base_id: str = Query(..., min_length=1, max_length=64),
        company_name: str | None = Query(default=None, max_length=256),
        report_period: str | None = Query(default=None, max_length=64),
        metric_code: str | None = Query(default=None, max_length=64),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        user_id: str | None = Query(default=None, max_length=128),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> V1MetricsResponse:
        token = _v1_auth_token(x_api_token, authorization)
        _v1_query_param(knowledge_base_id, "knowledge_base_id", 64)
        _v1_query_param(company_name, "company_name", 256)
        _v1_query_param(report_period, "report_period", 64)
        _v1_query_param(metric_code, "metric_code", 64)
        _v1_query_param(user_id, "user_id", 128)
        result = list_financial_metrics(
            kb_id=knowledge_base_id,
            company_name=company_name,
            report_period=report_period,
            metric_code=metric_code,
            limit=limit,
            offset=offset,
            user_id=user_id,
            x_api_token=token,
        )
        return _v1_response(result, knowledge_base_id)

    @app.post(
        "/api/v1/company-analyses", response_model=V1CompanyAnalysisResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    def v1_company_analysis(
        payload: V1CompanyAnalysisRequest = Body(...),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> V1CompanyAnalysisResponse:
        token = _v1_auth_token(x_api_token, authorization)
        result = get_company_analysis(
            kb_id=payload.knowledge_base_id,
            company_name=payload.company_name,
            report_period=payload.report_period,
            comparison_period=payload.comparison_period,
            user_id=payload.user_id,
            x_api_token=token,
        )
        return _v1_response(result, payload.knowledge_base_id)

    @app.post(
        "/api/v1/peer-comparisons", response_model=V1PeerComparisonResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    def v1_peer_comparison(
        payload: V1PeerComparisonRequest = Body(...),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> V1PeerComparisonResponse:
        token = _v1_auth_token(x_api_token, authorization)
        result = get_peer_comparison(
            company_names=payload.company_names,
            report_period=payload.report_period,
            kb_id=payload.knowledge_base_id,
            metric_codes=payload.metric_codes,
            user_id=payload.user_id,
            x_api_token=token,
        )
        return _v1_response(result, payload.knowledge_base_id)

    @app.post(
        "/api/v1/reports/export", response_model=V1ExportResponse,
        responses=_V1_SECURITY_RESPONSES,
        dependencies=[Security(_V1_BEARER_SECURITY), Security(_V1_API_TOKEN_SECURITY)],
    )
    def v1_reports_export(
        payload: V1ReportExportRequest = Body(...),
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> V1ExportResponse:
        """导出研究底稿，返回 JSON 内容供已有系统保存或下载。"""
        token = _v1_auth_token(x_api_token, authorization)
        if payload.report_type == "company_analysis":
            if not payload.company_name:
                raise HTTPException(status_code=422, detail="company_name 不能为空")
            response = export_company_analysis(
                kb_id=payload.knowledge_base_id,
                company_name=payload.company_name,
                report_period=payload.report_period,
                comparison_period=payload.comparison_period,
                user_id=payload.user_id,
                x_api_token=token,
            )
            filename = "company-analysis-export.md"
        else:
            if not payload.company_names or not 2 <= len(payload.company_names) <= 3:
                raise HTTPException(status_code=422, detail="company_names 必须包含 2 至 3 家公司")
            response = export_peer_comparison(
                company_names=payload.company_names,
                report_period=payload.report_period,
                kb_id=payload.knowledge_base_id,
                metric_codes=payload.metric_codes,
                user_id=payload.user_id,
                x_api_token=token,
            )
            filename = "peer-comparison-export.md"
        content = response.body.decode("utf-8") if isinstance(response.body, bytes) else str(response.body)
        return _v1_response(
            {
                "report_type": payload.report_type,
                "filename": filename,
                "content_type": "text/markdown; charset=utf-8",
                "content": content,
            },
            payload.knowledge_base_id,
        )

    @app.get("/kb/financial-metrics/{metric_id}/revisions")
    def list_financial_metric_revisions(
        metric_id: int,
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        item = kb["financial_metric_store"].get(metric_id)
        if not item:
            raise HTTPException(status_code=404, detail="财务指标记录不存在")
        resolved_user = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, resolved_user, item["kb_id"])
        return {"revisions": kb["financial_metric_store"].revisions(metric_id)}

    @app.patch("/kb/financial-metrics/{metric_id}")
    def patch_financial_metric(
        metric_id: int,
        payload: FinancialMetricPatchRequest,
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        existing = kb["financial_metric_store"].get(metric_id)
        if not existing:
            raise HTTPException(status_code=404, detail="财务指标记录不存在")
        resolved_user = _resolve_user_id(kb, x_api_token, payload.user_id)
        _require_kb_access(kb, resolved_user, existing["kb_id"], required=True)
        patch = payload.model_dump(exclude={"reason", "user_id"}, exclude_unset=True)
        try:
            item = kb["financial_metric_store"].update(
                metric_id, patch, actor=resolved_user or "", reason=payload.reason
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not item:
            raise HTTPException(status_code=404, detail="财务指标记录不存在")
        return {"item": item}

    @app.get("/kb/docs")
    def kb_list_docs(
        kb_id: str | None = None,
        user_id: str | None = None,
        x_api_token: str | None = Header(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        """列出知识库文档（按 doc_id 聚合），支持分页。kb_id 指定时只列该库。

        鉴权：X-Api-Token 头优先，user_id 直传兼容；按用户可访问范围过滤。
        limit/offset 透传给 store（P2-1：响应体大小可控）。
        """
        try:
            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, user_id)
            if user_id and kb_id:
                _require_kb_access(kb, user_id, kb_id)
            docs, total = kb["store"].list_docs(kb_id=kb_id, limit=limit, offset=offset)
            if user_id and not kb_id:
                allowed = set(kb["auth"].get_allowed_kbs(user_id))
                docs = [d for d in docs if d.get("kb_id") in allowed]
            elif not user_id and not kb_id and _ENFORCE_KB_VISIBILITY:
                public_ids = kb["kb_meta_store"].public_ids()
                docs = [d for d in docs if d.get("kb_id") in public_ids]
            return {
                "docs": docs,
                "total": total,
                "total_chunks": sum(d["chunks"] for d in docs),
                "kb_id": kb_id,
                "limit": limit,
                "offset": offset,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.get("/kb/kbs")
    def kb_list_kbs(
        user_id: str | None = None, x_api_token: str | None = Header(default=None)
    ) -> Dict[str, Any]:
        """列出所有知识库（kb_id 去重）——知识库管理界面用。

        鉴权：X-Api-Token 头优先，user_id 直传兼容；有身份则只返回可访问的库。
        """
        try:
            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, user_id)
            kbs = kb["store"].list_kbs()
            if user_id:
                if not kb["auth"].is_admin(user_id):
                    allowed = set(kb["auth"].get_allowed_kbs(user_id))
                    kbs = [k for k in kbs if k in allowed]
            elif _ENFORCE_KB_VISIBILITY:
                public_ids = kb["kb_meta_store"].public_ids()
                kbs = [k for k in kbs if k in public_ids]
            return {"kbs": kbs}
        except HTTPException:
            raise  # 401/403 直接抛出，不被转 500
        except Exception as exc:
            logger.error("KB list kbs failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.delete("/kb/docs/{doc_id}")
    def kb_delete_doc(
        doc_id: str, user_id: str | None = None, x_api_token: str | None = Header(default=None)
    ) -> Dict[str, Any]:
        """删除文档及其全部分块。写操作必须带身份，并校验该文档所属库的访问权。"""
        try:
            kb = _get_kb()
            # 鉴权：写操作强制身份 + 查 doc 所属库 → 校验访问权
            user_id = _resolve_user_id(kb, x_api_token, user_id)
            if not user_id:
                raise HTTPException(status_code=401, detail="此操作需提供身份（token 或 user_id）")
            doc_kb = kb["store"].get_doc_kb_id(doc_id)
            if doc_kb:
                _require_kb_access(kb, user_id, doc_kb, required=True)
            deleted = kb["store"].delete_doc(doc_id)
            kb["audit"].record(
                user_id=user_id or "anonymous", action="delete", target=doc_id,
                detail={"kb_id": doc_kb or "", "chunks_removed": deleted},
            )
            return {"doc_id": doc_id, "deleted_chunks": deleted}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删除失败: {exc}") from exc

    @app.put("/kb/docs/{doc_id}")
    def kb_update_doc(
        doc_id: str,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        kb_id: str = Form(default="default"),
        user_id: str | None = Form(default=None),
        ocr_mode: Literal["local", "baidu", "auto"] = Form(default="local"),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """更新文档：先准备（解析+向量化）→ 原子 upsert → 清理 stale 旧块。

        保持 doc_id 不变 → 历史引用/书签不失效。kb_id 决定新归属（可跨库迁移）。
        P0-2 原子化：解析/向量化失败时旧文档完好（先删旧块会丢数据）。
        ocr_mode：见 /kb/ingest。
        鉴权：X-Api-Token 头优先，user_id 直传兼容；校验对目标 kb_id 的写权限。
        """
        try:
            kb = _get_kb()
            user_id = _resolve_user_id(kb, x_api_token, user_id)
            _require_kb_access(kb, user_id, kb_id, required=True)
            # P0：文档级锁——prepare+写+删 stale 三步串行化，防并发误删
            with _get_doc_lock(doc_id):
                # 1. 准备（不碰旧数据；任何失败 → 旧文档完好）
                chunks, vectors, resolved_title, content_hash = _prepare_chunks(
                    kb, file, doc_id=doc_id, title=title, kb_id=kb_id, ocr_mode=ocr_mode
                )
                original_name, original_path = _archive_original(kb, file, doc_id)
                # 2. 原子写（upsert 同 id 覆盖，单次调用）
                old_ids = set(kb["store"].get_doc_ids(doc_id))
                new_ids = _write_chunks(
                    kb, chunks=chunks, vectors=vectors, doc_id=doc_id,
                    doc_title=resolved_title, kb_id=kb_id, content_hash=content_hash,
                )
                # 3. 清理 stale（分块数变少时多余的旧块；写后删，失败不影响新文档）
                new_set = set(new_ids)  # 复核修复：提出循环，避免每元素重建 set
                stale = [cid for cid in old_ids if cid not in new_set]
                deleted = kb["store"].delete_chunk_ids(stale, kb_id=kb_id)
                kb["governance_store"].create_version(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    title=resolved_title,
                    snapshot=kb["store"].snapshot_doc(doc_id),
                    created_by=user_id or "anonymous",
                    state="published",
                    original_name=original_name,
                    original_path=original_path,
                )
            kb["audit"].record(
                user_id=user_id or "anonymous", action="update", target=doc_id,
                detail={"kb_id": kb_id, "chunks": len(chunks), "stale_removed": deleted},
            )
            return {
                "doc_id": doc_id,
                "deleted_chunks": deleted,
                "chunks": len(chunks),
                "title": resolved_title,
                "kb_id": kb_id,
            }
        except HTTPException:
            raise
        except ValueError as exc:  # 不支持的文件类型 / 无扩展名等：参数问题，映射 400
            raise HTTPException(status_code=400, detail=f"更新失败: {exc}") from exc
        except Exception as exc:
            logger.error("KB update failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"更新失败: {exc}") from exc

    @app.post("/kb/docs/{doc_id}/versions")
    def kb_stage_doc_version(
        doc_id: str,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        kb_id: str = Form(default="default"),
        ocr_mode: Literal["local", "baidu", "auto"] = Form(default="local"),
        user_id: str | None = Form(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """创建不影响线上检索的文档草稿版本。"""
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, user_id)
        _require_kb_access(kb, user_id, kb_id, required=True)
        existing_kb = kb["store"].get_doc_kb_id(doc_id)
        if not existing_kb:
            raise HTTPException(status_code=404, detail="文档不存在")
        if existing_kb != kb_id:
            raise HTTPException(status_code=409, detail="文档所属知识库不一致")
        chunks, vectors, resolved_title, content_hash = _prepare_chunks(
            kb, file, doc_id=doc_id, title=title, kb_id=kb_id, ocr_mode=ocr_mode
        )
        original_name, original_path = _archive_original(kb, file, doc_id)
        version = kb["governance_store"].create_version(
            doc_id=doc_id,
            kb_id=kb_id,
            title=resolved_title,
            snapshot=_snapshot_from_prepared(
                chunks=chunks,
                vectors=vectors,
                doc_id=doc_id,
                doc_title=resolved_title,
                kb_id=kb_id,
                content_hash=content_hash,
            ),
            created_by=user_id or "anonymous",
            state="draft",
            original_name=original_name,
            original_path=original_path,
        )
        kb["audit"].record(
            user_id=user_id or "anonymous", action="create_doc_draft", target=str(version["id"])
        )
        return _public_version(version)

    @app.get("/kb/docs/{doc_id}/versions")
    def kb_list_doc_versions(
        doc_id: str,
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, user_id)
        versions = kb["governance_store"].list_versions(doc_id)
        kb_id = kb["store"].get_doc_kb_id(doc_id)
        if not kb_id and versions:
            kb_id = versions[0]["kb_id"]
        if kb_id:
            _require_kb_access(kb, user_id, kb_id, required=True)
        if not versions and kb_id:
            snapshot = kb["store"].snapshot_doc(doc_id)
            if snapshot.get("ids"):
                metadata = (snapshot.get("metadatas") or [{}])[0] or {}
                kb["governance_store"].create_version(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    title=str(metadata.get("doc_title", doc_id)),
                    snapshot=snapshot,
                    created_by="migration",
                    state="published",
                    review_note="Existing document baseline",
                )
                versions = kb["governance_store"].list_versions(doc_id)
        return {"versions": versions}

    @app.get("/kb/docs/versions/pending")
    def kb_list_pending_doc_versions(
        kb_id: str | None = Query(default=None),
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_supervisor(kb, x_api_key, x_api_token, operator_id)
        versions = kb["governance_store"].list_pending(kb_id)
        if not operator.startswith("api_key:"):
            versions = [
                version for version in versions
                if kb["auth"].can_access(operator, version["kb_id"])
            ]
        return {"versions": versions}

    @app.post("/kb/docs/versions/{version_id}/submit")
    def kb_submit_doc_version(
        version_id: int,
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, user_id)
        version = kb["governance_store"].get(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="文档版本不存在")
        _require_kb_access(kb, user_id, version["kb_id"], required=True)
        updated = kb["governance_store"].transition(
            version_id, "review", actor=user_id or "anonymous"
        )
        return _public_version(updated) if updated else {}

    @app.get("/kb/docs/versions/{version_id}/original")
    def kb_download_doc_original(
        version_id: int,
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> FileResponse:
        kb = _get_kb()
        user_id = _resolve_user_id(kb, x_api_token, user_id)
        version = kb["governance_store"].get(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="文档版本不存在")
        _require_kb_access(kb, user_id, version["kb_id"])
        path = _version_original_path(kb, version.get("original_path", ""))
        allowed_root = (
            Path(kb["config"].kb_chroma_dir).parent / "document_originals"
        ).resolve()
        if not path.is_file() or allowed_root not in path.parents:
            raise HTTPException(status_code=404, detail="原始文件不存在")
        return FileResponse(path, filename=version.get("original_name") or path.name)

    @app.post("/kb/docs/versions/{version_id}/review")
    def kb_review_doc_version(
        version_id: int,
        action: Literal["approve", "reject"] = Form(...),
        note: str = Form(default=""),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
        target = "approved" if action == "approve" else "rejected"
        version = kb["governance_store"].get(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="文档版本不存在")
        if not operator.startswith("api_key:"):
            _require_kb_access(kb, operator, version["kb_id"], required=True)
        try:
            updated = kb["governance_store"].transition(
                version_id, target, actor=operator, note=note
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not updated:
            raise HTTPException(status_code=404, detail="文档版本不存在")
        return _public_version(updated)

    @app.post("/kb/docs/versions/{version_id}/publish")
    def kb_publish_doc_version(
        version_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
        version = kb["governance_store"].get(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="文档版本不存在")
        if not operator.startswith("api_key:"):
            _require_kb_access(kb, operator, version["kb_id"], required=True)
        if version["state"] != "approved":
            raise HTTPException(status_code=409, detail="只有已批准版本才能发布")
        with _get_doc_lock(version["doc_id"]):
            chunks = kb["store"].restore_doc_snapshot(version["snapshot"])
            updated = kb["governance_store"].transition(
                version_id, "published", actor=operator
            )
        kb["audit"].record(
            user_id=operator, action="publish_doc_version", target=str(version_id)
        )
        if updated:
            result = _public_version(updated)
            result["chunks"] = chunks
            return result
        return {}

    @app.post("/kb/docs/versions/{version_id}/rollback")
    def kb_rollback_doc_version(
        version_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
        source = kb["governance_store"].get(version_id)
        if not source or source["state"] != "published":
            raise HTTPException(status_code=409, detail="只能回滚到已发布版本")
        if not operator.startswith("api_key:"):
            _require_kb_access(kb, operator, source["kb_id"], required=True)
        with _get_doc_lock(source["doc_id"]):
            chunks = kb["store"].restore_doc_snapshot(source["snapshot"])
            rolled = kb["governance_store"].create_rollback(version_id, actor=operator)
        if rolled:
            result = _public_version(rolled)
            result["chunks"] = chunks
        else:
            result = {}
        kb["audit"].record(
            user_id=operator, action="rollback_doc_version", target=str(version_id)
        )
        return result

    # ==================== 数据源同步 ====================

    @app.get("/kb/sources")
    def kb_list_sources(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        return {"sources": kb["source_store"].list()}

    @app.post("/kb/sources")
    def kb_create_source(
        payload: Dict[str, Any] = Body(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        kb_id = str(payload.get("kb_id", "default"))
        try:
            source = kb["source_store"].create(
                name=str(payload.get("name", "")),
                kb_id=kb_id,
                source_type=str(payload.get("source_type", "file")),
                location=str(payload.get("location", "")),
                interval_minutes=int(payload.get("interval_minutes", 60)),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        kb["audit"].record(user_id=operator, action="create_source", target=str(source["id"]))
        return source

    @app.post("/kb/sources/{source_id}/run")
    def kb_run_source(
        source_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        source = kb["source_store"].get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="数据源不存在")
        if not kb["source_store"].claim(source_id):
            raise HTTPException(status_code=409, detail="数据源正在同步，请稍后重试")
        source = kb["source_store"].get(source_id) or source
        try:
            return _run_source_sync(kb, source, actor=operator)
        except Exception as exc:
            kb["source_store"].mark_result(source_id, ok=False, error=str(exc))
            kb["alert_store"].emit(
                dedupe_key=f"source-sync-{source_id}", alert_type="source_sync",
                severity="high", target=str(source_id), message=f"数据源同步失败：{exc}",
            )
            raise HTTPException(status_code=400, detail=f"同步失败: {exc}") from exc

    @app.delete("/kb/sources/{source_id}")
    def kb_delete_source(
        source_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        removed = kb["source_store"].delete(source_id)
        kb["audit"].record(user_id=operator, action="delete_source", target=str(source_id))
        return {"id": source_id, "deleted": removed}

    # ==================== 用户授权长期记忆与保留策略 ====================

    def _memory_owner(
        kb: dict, x_api_token: str | None, visitor_token: str | None
    ) -> tuple[str, str | None]:
        user_id = _resolve_user_id(kb, x_api_token, None)
        return _conversation_owner_key(kb, user_id, visitor_token, required=True), user_id

    @app.get("/kb/memory")
    def kb_get_memory(
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        owner, _ = _memory_owner(kb, x_api_token, visitor_token)
        return {
            "consent": kb["privacy_store"].has_consent(owner),
            "memories": kb["privacy_store"].list_memory(owner),
        }

    @app.put("/kb/memory/consent")
    def kb_set_memory_consent(
        payload: Dict[str, Any] = Body(...),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        owner, user_id = _memory_owner(kb, x_api_token, payload.get("visitor_token"))
        enabled = bool(payload.get("enabled"))
        kb["privacy_store"].set_consent(owner, enabled)
        kb["audit"].record(
            user_id=user_id or "visitor", action="memory_consent", target="enabled" if enabled else "disabled"
        )
        return {"consent": enabled}

    @app.post("/kb/memory")
    def kb_set_memory(
        payload: Dict[str, Any] = Body(...),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        owner, _ = _memory_owner(kb, x_api_token, payload.get("visitor_token"))
        try:
            memory = kb["privacy_store"].set_memory(
                owner,
                str(payload.get("key", "")),
                str(payload.get("value", "")),
                expires_days=int(payload["expires_days"]) if payload.get("expires_days") else None,
            )
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return memory

    @app.delete("/kb/memory/{memory_id}")
    def kb_delete_memory(
        memory_id: int,
        visitor_token: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        owner, _ = _memory_owner(kb, x_api_token, visitor_token)
        return {"id": memory_id, "deleted": kb["privacy_store"].delete_memory(owner, memory_id)}

    @app.get("/kb/privacy/policy")
    def kb_get_retention_policy(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        return {"policy": kb["privacy_store"].policy()}

    @app.put("/kb/privacy/policy")
    def kb_update_retention_policy(
        payload: Dict[str, Any] = Body(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        try:
            policy = kb["privacy_store"].update_policy(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        kb["audit"].record(user_id=operator, action="update_retention_policy", target="privacy")
        return {"policy": policy}

    @app.post("/kb/privacy/cleanup")
    def kb_run_retention_cleanup(
        dry_run: bool = Query(default=True),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        from services.kb.privacy import run_retention_cleanup

        kb = _get_kb()
        operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
        result = run_retention_cleanup(
            Path(kb["config"].kb_chroma_dir).parent,
            kb["privacy_store"].policy(),
            dry_run=dry_run,
        )
        if not dry_run:
            kb["audit"].record(user_id=operator, action="retention_cleanup", target="all", detail=result)
        return {"dry_run": dry_run, "deleted": result}

    # ==================== 客服 SLA、运营看板与告警 ====================

    @app.put("/kb/agent/availability")
    def kb_set_agent_availability(
        available: bool = Body(embed=True),
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_agent(kb, x_api_key, x_api_token, operator_id)
        if operator.startswith("api_key:"):
            raise HTTPException(status_code=400, detail="运维密钥不能作为坐席上线")
        kb["auth"].set_agent_available(operator, available)
        return {"user_id": operator, "available": available}

    @app.put("/kb/agent/conversations/{conv_id}/priority")
    def kb_set_conversation_priority(
        conv_id: int,
        priority: str = Body(embed=True),
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_agent(kb, x_api_key, x_api_token, operator_id)
        _require_agent_conversation_access(kb, operator, conv_id)
        try:
            updated = kb["conversation_store"].set_priority(conv_id, priority)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"conversation_id": conv_id, "priority": priority, "updated": updated}

    @app.get("/kb/operations/dashboard")
    def kb_operations_dashboard(
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        _require_kb_supervisor(kb, x_api_key, x_api_token, operator_id)
        _scan_operational_alerts(kb)
        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "conversations": kb["conversation_store"].stats(),
            "tickets": kb["ticket_store"].stats(),
            "feedback": kb["feedback_store"].stats(),
            "rag": kb["retrieval_log"].stats(),
            "open_alerts": kb["alert_store"].count_open(),
        }

    @app.get("/kb/operations/alerts")
    def kb_operations_alerts(
        status: str | None = Query(default=None),
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        _require_kb_supervisor(kb, x_api_key, x_api_token, operator_id)
        _scan_operational_alerts(kb)
        return {"alerts": kb["alert_store"].list(status=status)}

    @app.post("/kb/operations/alerts/{alert_id}/ack")
    def kb_ack_alert(
        alert_id: int,
        operator_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        operator = _require_kb_supervisor(kb, x_api_key, x_api_token, operator_id)
        return {"id": alert_id, "acknowledged": kb["alert_store"].acknowledge(alert_id, operator)}

    def _maintenance_loop() -> None:
        last_cleanup = ""
        while not _maintenance_stop.wait(60):
            try:
                kb = _get_kb()
                for source in kb["source_store"].claim_due():
                    try:
                        _run_source_sync(kb, source, actor="system")
                    except Exception as exc:
                        kb["source_store"].mark_result(source["id"], ok=False, error=str(exc))
                        kb["alert_store"].emit(
                            dedupe_key=f"source-sync-{source['id']}", alert_type="source_sync",
                            severity="high", target=str(source["id"]), message=f"数据源同步失败：{exc}",
                        )
                _scan_operational_alerts(kb)
                today = time.strftime("%Y-%m-%d")
                if today != last_cleanup and (
                    getattr(_cfg, "app_env", "development").lower() in {"production", "prod"}
                    or os.getenv("KB_AUTO_RETENTION", "").lower() in {"1", "true", "yes"}
                ):
                    from services.kb.privacy import run_retention_cleanup

                    run_retention_cleanup(
                        Path(kb["config"].kb_chroma_dir).parent,
                        kb["privacy_store"].policy(),
                        dry_run=False,
                    )
                    last_cleanup = today
            except Exception as exc:
                logger.warning("后台维护任务失败（下轮重试）: {}", exc)

    def start_maintenance_worker() -> None:
        nonlocal _maintenance_thread
        if _maintenance_thread is None or not _maintenance_thread.is_alive():
            _maintenance_stop.clear()
            _maintenance_thread = Thread(
                target=_maintenance_loop, name="kb-maintenance", daemon=True
            )
            _maintenance_thread.start()

    def stop_maintenance_worker() -> None:
        _maintenance_stop.set()

    # ==================== 用户与权限管理（RBAC，P3 §3.4）====================

    @app.get("/kb/me")
    def kb_current_user(
        user_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        kb = _get_kb()
        resolved = _resolve_user_id(kb, x_api_token, user_id)
        if not resolved:
            raise HTTPException(status_code=401, detail="尚未登录")
        user = kb["auth"].get_user(resolved)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return user

    @app.post("/kb/users")
    def kb_create_user(
        name: str = Form(...),
        role: str = Form(default="member"),
        x_api_key: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
        admin_id: str | None = Query(default=None),
    ) -> Dict[str, Any]:
        """新建用户，返回 user_id 与 API token（明文仅此一次）。role: member | admin。

        鉴权（P0-1）：
        - 默认需管理员（X-API-Key 匹配 ADMIN_API_KEY 或 admin token/user_id）。
        - KB_OPEN_SIGNUP=1 时允许自助注册，但角色强制 member，杜绝 admin 提权。
        首个 admin 建议走 KB_BOOTSTRAP_ADMIN_TOKEN 环境变量引导，而非本接口。
        """
        try:
            kb = _get_kb()
            if _OPEN_SIGNUP:
                # 自助注册：角色强制 member，防提权
                if role != "member":
                    raise HTTPException(status_code=403, detail="自助注册只能创建 member 账号")
                operator = "anonymous"
            else:
                operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            uid, token = kb["auth"].create_user(name, role)
            kb["audit"].record(
                user_id=operator, action="create_user", target=uid,
                detail={"name": name, "role": role},
            )
            return {"user_id": uid, "name": name, "role": role, "api_token": token}
        except HTTPException:
            raise  # 401/403 直接抛出，不被转 500（P0-1 审核修复）
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB create user failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"建用户失败: {exc}") from exc

    @app.get("/kb/users")
    def kb_list_users(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """列出所有用户及其可访问的知识库（需管理员）。X-API-Key 或 admin token/user_id 均可。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"users": kb["auth"].list_users()}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list users failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.get("/kb/users/{user_id}")
    def kb_get_user(
        user_id: str,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """用户详情（含可访问的 kb 列表，需管理员）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            user = kb["auth"].get_user(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="用户不存在")
            return user
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB get user failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.post("/kb/users/{user_id}/access")
    def kb_grant_access(
        user_id: str,
        kb_id: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """授权用户访问某知识库（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["auth"].get_user(user_id):
                raise HTTPException(status_code=404, detail="用户不存在")
            kb["auth"].grant_access(user_id, kb_id)
            kb["audit"].record(
                user_id=operator, action="grant", target=user_id,
                detail={"kb_id": kb_id},
            )
            return {"user_id": user_id, "kb_id": kb_id, "granted": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB grant failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"授权失败: {exc}") from exc

    @app.post("/kb/users/{user_id}/role")
    def kb_set_role(
        user_id: str,
        role: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """设置用户角色（member | admin，需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["auth"].get_user(user_id):
                raise HTTPException(status_code=404, detail="用户不存在")
            ok = kb["auth"].set_role(user_id, role)
            kb["audit"].record(
                user_id=operator, action="set_role", target=user_id,
                detail={"role": role},
            )
            return {"user_id": user_id, "role": role, "updated": ok}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB set role failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"设置角色失败: {exc}") from exc

    @app.delete("/kb/users/{user_id}/access")
    def kb_revoke_access(
        user_id: str,
        kb_id: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """撤销用户对某知识库的访问权（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            removed = kb["auth"].revoke_access(user_id, kb_id)
            kb["audit"].record(
                user_id=operator, action="revoke", target=user_id,
                detail={"kb_id": kb_id},
            )
            return {"user_id": user_id, "kb_id": kb_id, "removed": removed}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB revoke failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"撤销失败: {exc}") from exc

    @app.post("/kb/users/{user_id}/token")
    def kb_reset_token(
        user_id: str,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """重置用户 API token（需管理员；token 泄露时用，旧 token 立即失效）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["auth"].get_user(user_id):
                raise HTTPException(status_code=404, detail="用户不存在")
            token = kb["auth"].reset_token(user_id)
            kb["audit"].record(
                user_id=operator, action="reset_token", target=user_id,
            )
            return {"user_id": user_id, "api_token": token}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB reset token failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"重置 token 失败: {exc}") from exc

    # ==================== 会话管理（P2：thread 无限累积需清理）====================

    @app.get("/kb/threads")
    def kb_list_threads(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """列出所有会话线程（thread_id + checkpoint 数），需管理员。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            conn = kb["checkpoint_conn"]
            rows = conn.execute(
                "SELECT thread_id, COUNT(*) AS n FROM checkpoints "
                "GROUP BY thread_id ORDER BY n DESC"
            ).fetchall()
            return {
                "threads": [{"thread_id": r[0], "checkpoints": r[1]} for r in rows],
                "total": len(rows),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list threads failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.delete("/kb/threads/{thread_id}")
    def kb_delete_thread(
        thread_id: str,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """删除某会话线程的全部 checkpoint（需管理员；会话清理用）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            conn = kb["checkpoint_conn"]
            cur1 = conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
            conn.commit()
            kb["audit"].record(
                user_id=operator, action="delete_thread", target=thread_id,
            )
            return {"thread_id": thread_id, "deleted_checkpoints": cur1.rowcount}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete thread failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删除失败: {exc}") from exc

    # ==================== FAQ 管理（客服改造第5项：标准问+答案+相似问）====================

    @app.post("/kb/faqs")
    def kb_add_faq(
        kb_id: str = Form(default="default"),
        question: str = Form(...),
        answer: str = Form(...),
        similar: str = Form(default=""),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """新增 FAQ（需管理员）。similar 用分号/逗号分隔多个相似问。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            similars = kb["faq_store"]._split_similars(similar)
            faq_id = kb["faq_store"].add_faq(kb_id, question, answer, similars)
            kb["audit"].record(
                user_id=operator, action="add_faq", target=str(faq_id),
                detail={"kb_id": kb_id, "question": question[:50]},
            )
            return {"faq_id": faq_id, "question": question}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB add faq failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"新增 FAQ 失败: {exc}") from exc

    @app.get("/kb/faqs")
    def kb_list_faqs(
        kb_id: str | None = None,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """列出 FAQ（需管理员；kb_id 可筛选）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"faqs": kb["faq_store"].list_faqs(kb_id=kb_id)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list faqs failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询 FAQ 失败: {exc}") from exc

    @app.delete("/kb/faqs/{faq_id}")
    def kb_delete_faq(
        faq_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """删除 FAQ（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            removed = kb["faq_store"].delete_faq(faq_id)
            kb["audit"].record(
                user_id=operator, action="delete_faq", target=str(faq_id),
            )
            return {"faq_id": faq_id, "removed": removed}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete faq failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删除 FAQ 失败: {exc}") from exc

    @app.post("/kb/faqs/import")
    def kb_import_faqs(
        file: UploadFile = File(...),
        kb_id: str = Form(default="default"),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """Excel 批量导入 FAQ（需管理员）。模板：标准问 | 答案 | 相似问（分号分隔）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            result = kb["faq_store"].import_excel(file.file.read(), kb_id)
            kb["audit"].record(
                user_id=operator, action="import_faq", target=kb_id,
                detail={"imported": result["imported"], "skipped": result["skipped"]},
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB import faqs failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"导入 FAQ 失败: {exc}") from exc

    @app.get("/kb/faqs/export")
    def kb_export_faqs(
        kb_id: str | None = None,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Response:
        """导出 FAQ 为 Excel（需管理员）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            content = kb["faq_store"].export_excel(kb_id=kb_id)
            return Response(
                content=content,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=faqs.xlsx"},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB export faqs failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"导出 FAQ 失败: {exc}") from exc

    # ==================== 客服人设（第6项）+ 满意度（第7项）====================

    @app.get("/kb/persona")
    def kb_get_persona(
        kb_id: str = Query(default="default"),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """获取客服人设（需管理员）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"kb_id": kb_id, "persona": kb["persona_store"].get_persona(kb_id)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB get persona failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"获取人设失败: {exc}") from exc

    @app.put("/kb/persona")
    def kb_set_persona(
        payload: Dict[str, Any] = Body(...),
        kb_id: str = Query(default="default"),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """更新客服人设（需管理员）。字段：company_name/service_hours/tone/
        refuse_message/transfer_message。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            kb["persona_store"].set_persona(kb_id, payload)
            kb["audit"].record(user_id=operator, action="set_persona", target=kb_id)
            return {"kb_id": kb_id, "persona": kb["persona_store"].get_persona(kb_id)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB set persona failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"更新人设失败: {exc}") from exc

    @app.post("/kb/feedback")
    def kb_feedback(
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """满意度评价（无鉴权，访客/用户均可）。rating: 1=👍 / 0=👎；comment 可选。"""
        try:
            kb = _get_kb()
            # P2：无鉴权端点必须限流（防刷）
            if not _kb_ask_limiter.allow("feedback"):
                raise HTTPException(status_code=429, detail="反馈过于频繁，请稍后再试")
            from services.kb.privacy import redact_text

            kb_id = str(payload.get("kb_id", "default"))[:64]
            question = redact_text(str(payload.get("question", "")))[:500]
            answer = redact_text(str(payload.get("answer", "")))[:500]
            rating = 1 if payload.get("rating", 1) else 0
            comment = redact_text(str(payload.get("comment", "")))[:500]
            fid = kb["feedback_store"].record(
                kb_id=kb_id, question=question, answer=answer, rating=rating, comment=comment,
            )
            return {"feedback_id": fid, "rating": rating}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB feedback failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"提交反馈失败: {exc}") from exc

    @app.get("/kb/feedback")
    def kb_list_feedback(
        negative_only: bool = Query(default=False),
        limit: int = Query(default=100, ge=1, le=1000),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """满意度反馈列表（需管理员）；negative_only=True 只看差评（复盘用）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {
                "total": kb["feedback_store"].count(negative_only=negative_only),
                "entries": kb["feedback_store"].recent(limit, negative_only=negative_only),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list feedback failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询反馈失败: {exc}") from exc

    # ==================== 客服工作台（第9项：转人工闭环）====================

    @app.get("/kb/agent/queue")
    def kb_agent_queue(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """待接入池（waiting 会话），客服工作台首页。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            conversations = kb["conversation_store"].list_by_status("waiting")
            human = kb["conversation_store"].list_by_status("human")
            if operator.startswith("api_key:") or kb["auth"].has_role(
                operator, {"supervisor", "admin"}
            ):
                conversations.extend(human)
            else:
                conversations.extend(c for c in human if c["agent_id"] == operator)
            if not operator.startswith("api_key:") and not kb["auth"].is_admin(operator):
                allowed = set(kb["auth"].get_allowed_kbs(operator))
                conversations = [c for c in conversations if c["kb_id"] in allowed]
            return {"conversations": conversations}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent queue failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询待接入池失败: {exc}") from exc

    @app.post("/kb/agent/claim/{conv_id}")
    def kb_agent_claim(
        conv_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """坐席领取会话（waiting → human）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            _require_agent_conversation_access(kb, operator, conv_id)
            claimed = kb["conversation_store"].claim(conv_id, operator)
            if claimed:
                kb["audit"].record(user_id=operator, action="agent_claim", target=str(conv_id))
            return {"conversation_id": conv_id, "claimed": claimed}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent claim failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"领取失败: {exc}") from exc

    @app.post("/kb/agent/reply/{conv_id}")
    def kb_agent_reply(
        conv_id: int,
        content: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """坐席回复（存 agent 消息，清零未读）。校验会话存在 + 归属 + 状态。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            conv = _require_agent_conversation_access(kb, operator, conv_id)
            if conv["status"] != "human":
                raise HTTPException(status_code=409, detail="会话未在人工服务中")
            if conv["agent_id"] and conv["agent_id"] != operator:
                raise HTTPException(status_code=403, detail="该会话由其他坐席处理")
            from services.kb.privacy import redact_text

            kb["conversation_store"].add_message(conv_id, "agent", redact_text(content))
            kb["conversation_store"].mark_read(conv_id)
            kb["audit"].record(user_id=operator, action="agent_reply", target=str(conv_id))
            return {"conversation_id": conv_id, "replied": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent reply failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"回复失败: {exc}") from exc

    @app.post("/kb/agent/close/{conv_id}")
    def kb_agent_close(
        conv_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """结束会话。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            _require_agent_conversation_access(kb, operator, conv_id)
            kb["conversation_store"].close(conv_id)
            kb["audit"].record(user_id=operator, action="agent_close", target=str(conv_id))
            return {"conversation_id": conv_id, "closed": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent close failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"结束会话失败: {exc}") from exc

    @app.get("/kb/agent/conversations/{conv_id}/messages")
    def kb_agent_messages(
        conv_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """会话消息记录（坐席查看）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            _require_agent_conversation_access(kb, operator, conv_id)
            return {"messages": kb["conversation_store"].list_messages(conv_id)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent messages failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询消息失败: {exc}") from exc

    @app.post("/kb/agent/conversations/{conv_id}/tag")
    def kb_agent_tag(
        conv_id: int,
        tag: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """会话打标签（如"退款咨询"），便于统计。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            _require_agent_conversation_access(kb, operator, conv_id)
            kb["conversation_store"].set_tag(conv_id, tag)
            kb["audit"].record(user_id=operator, action="agent_tag", target=str(conv_id))
            return {"conversation_id": conv_id, "tag": tag}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent tag failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"打标签失败: {exc}") from exc

    # ==================== 工单（第10项）====================

    def _require_ticket_access(kb: dict, operator: str, ticket: dict[str, Any]) -> None:
        if not operator.startswith("api_key:") and not kb["auth"].can_access(
            operator, ticket["kb_id"]
        ):
            raise HTTPException(status_code=403, detail="无权处理该知识库的工单")

    @app.post("/kb/tickets")
    def kb_create_ticket(
        payload: Dict[str, Any] = Body(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """建工单（需管理员）。payload: conversation_id 可选, title, description。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            conversation_id = payload.get("conversation_id")
            kb_id = str(payload.get("kb_id", "default"))
            if conversation_id is not None:
                conv = _require_agent_conversation_access(kb, operator, int(conversation_id))
                kb_id = conv["kb_id"]
            elif not operator.startswith("api_key:"):
                _require_kb_access(kb, operator, kb_id, required=True)
            ticket = kb["ticket_store"].create(
                conversation_id=conversation_id,
                title=payload.get("title", ""),
                description=payload.get("description", ""),
                kb_id=kb_id,
                priority=payload.get("priority", "normal"),
                due_hours=int(payload.get("due_hours", 24)),
            )
            kb["audit"].record(user_id=operator, action="create_ticket", target=ticket["ticket_no"])
            return ticket
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB create ticket failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"建单失败: {exc}") from exc

    @app.get("/kb/tickets")
    def kb_list_tickets(
        status: str | None = Query(default=None),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """工单列表（需管理员，可筛状态）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            tickets = kb["ticket_store"].list(status=status)
            if not operator.startswith("api_key:"):
                tickets = [
                    ticket for ticket in tickets
                    if kb["auth"].can_access(operator, ticket["kb_id"])
                ]
            return {"tickets": tickets}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list tickets failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询工单失败: {exc}") from exc

    @app.get("/kb/tickets/{ticket_id}")
    def kb_get_ticket(
        ticket_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """工单详情（含时间线）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            ticket = kb["ticket_store"].get(ticket_id)
            if not ticket:
                raise HTTPException(status_code=404, detail="工单不存在")
            _require_ticket_access(kb, operator, ticket)
            return ticket
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB get ticket failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询工单失败: {exc}") from exc

    @app.post("/kb/tickets/{ticket_id}/status")
    def kb_update_ticket_status(
        ticket_id: int,
        status: str = Form(...),
        note: str = Form(default=""),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """变更工单状态（pending/processing/resolved/closed）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            ticket = kb["ticket_store"].get(ticket_id)
            if not ticket:
                raise HTTPException(status_code=404, detail="工单不存在")
            _require_ticket_access(kb, operator, ticket)
            updated = kb["ticket_store"].update_status(ticket_id, status, note)
            if not updated:
                raise HTTPException(status_code=404, detail="工单不存在")
            kb["audit"].record(user_id=operator, action="ticket_status", target=str(ticket_id))
            return {"ticket_id": ticket_id, "status": status}
        except HTTPException:
            raise
        except ValueError as exc:  # 非法流转/非法状态
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB update ticket failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"更新工单失败: {exc}") from exc

    @app.post("/kb/tickets/{ticket_id}/assign")
    def kb_assign_ticket(
        ticket_id: int,
        assignee: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """指派工单给坐席。"""
        try:
            kb = _get_kb()
            operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
            ticket = kb["ticket_store"].get(ticket_id)
            if not ticket:
                raise HTTPException(status_code=404, detail="工单不存在")
            _require_ticket_access(kb, operator, ticket)
            assignee_user = kb["auth"].get_user(assignee)
            if not assignee_user or not kb["auth"].is_customer_service(assignee):
                raise HTTPException(status_code=400, detail="指派目标不是有效客服账号")
            if not kb["auth"].can_access(assignee, ticket["kb_id"]):
                raise HTTPException(status_code=400, detail="指派目标无权访问该知识库")
            online_ids = {
                agent["user_id"] for agent in kb["auth"].available_agents(ticket["kb_id"])
            }
            if assignee not in online_ids:
                raise HTTPException(status_code=409, detail="指派目标当前不在线")
            assigned = kb["ticket_store"].assign(ticket_id, assignee)
            if not assigned:
                raise HTTPException(status_code=404, detail="工单不存在")
            kb["audit"].record(user_id=operator, action="ticket_assign", target=str(ticket_id))
            return {"ticket_id": ticket_id, "assignee": assignee}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB assign ticket failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"指派失败: {exc}") from exc

    # ==================== 快捷回复（第11项）====================

    @app.get("/kb/quick-replies")
    def kb_list_quick_replies(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """快捷回复列表（需管理员）。"""
        try:
            kb = _get_kb()
            _require_kb_agent(kb, x_api_key, x_api_token, admin_id)
            return {"quick_replies": kb["quick_reply_store"].list()}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list quick replies failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询快捷回复失败: {exc}") from exc

    @app.post("/kb/quick-replies")
    def kb_add_quick_reply(
        title: str = Form(...),
        content: str = Form(...),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """新增快捷回复（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
            rid = kb["quick_reply_store"].add(title, content)
            kb["audit"].record(user_id=operator, action="add_quick_reply", target=str(rid))
            return {"id": rid, "title": title}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB add quick reply failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"新增快捷回复失败: {exc}") from exc

    @app.delete("/kb/quick-replies/{reply_id}")
    def kb_delete_quick_reply(
        reply_id: int,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """删除快捷回复（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_supervisor(kb, x_api_key, x_api_token, admin_id)
            removed = kb["quick_reply_store"].delete(reply_id)
            kb["audit"].record(user_id=operator, action="delete_quick_reply", target=str(reply_id))
            return {"id": reply_id, "removed": removed}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete quick reply failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删除快捷回复失败: {exc}") from exc

    # ==================== 知识库元数据 CRUD（第15项）====================

    @app.get("/kb/meta")
    def kb_list_meta(
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """知识库元数据列表（需管理员）。"""
        try:
            kb = _get_kb()
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"kbs": kb["kb_meta_store"].list()}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list meta failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询知识库失败: {exc}") from exc

    @app.post("/kb/meta")
    def kb_create_meta(
        kb_id: str = Form(...),
        name: str = Form(...),
        description: str = Form(default=""),
        visibility: str = Form(default="internal"),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """建知识库（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            meta = kb["kb_meta_store"].create(kb_id, name, description, visibility)
            kb["audit"].record(user_id=operator, action="create_kb", target=kb_id)
            return meta
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB create meta failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"建库失败: {exc}") from exc

    @app.put("/kb/meta/{kb_id}")
    def kb_update_meta(
        kb_id: str,
        name: str | None = Form(default=None),
        description: str | None = Form(default=None),
        visibility: str | None = Form(default=None),
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """改知识库名称/描述（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            kb["kb_meta_store"].update(
                kb_id, name=name, description=description, visibility=visibility
            )
            kb["audit"].record(user_id=operator, action="update_kb", target=kb_id)
            return kb["kb_meta_store"].get(kb_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB update meta failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"改库失败: {exc}") from exc

    @app.delete("/kb/meta/{kb_id}")
    def kb_delete_meta(
        kb_id: str,
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """删知识库元数据（需管理员）。联动清理该库 Chroma 文档，避免重建时"复活"旧数据。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            removed = kb["kb_meta_store"].delete(kb_id)
            # P2：双数据源联动——删 meta 同时清 Chroma 文档
            chunks_deleted = kb["store"].delete_kb(kb_id)
            kb["audit"].record(
                user_id=operator, action="delete_kb", target=kb_id,
                detail={"meta_removed": removed, "chunks_deleted": chunks_deleted},
            )
            return {"kb_id": kb_id, "removed": removed, "chunks_deleted": chunks_deleted}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete meta failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删库失败: {exc}") from exc

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Windows 下 reload 不稳定，改用手动重启
        log_level="info"
    )
