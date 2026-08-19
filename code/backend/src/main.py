"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, Literal, Optional

from dotenv import load_dotenv

# 加载 .env 文件（在导入 config 之前，确保环境变量就绪）
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response, StreamingResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import Configuration, SearchAPI  # noqa: E402
from services.token_budget import TokenBudget, TokenBudgetExceeded, global_stats  # noqa: E402
# DeepResearchAgent 已改为路由内懒加载：KB 功能独立运行，不依赖 hello_agents

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
logger = logger.patch(lambda record: record["extra"].setdefault("request_id", _request_id_ctx.get()))


class _SlidingWindowLimiter:
    """进程内滑动窗口限流器。用于保护深度研究等重资源/付费接口。"""

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


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    research_depth: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="P1: research depth (1=fast, 2=standard, 3=deep) overriding MAX_WEB_RESEARCH_LOOPS",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


class RegenerateRequest(BaseModel):
    """Payload for re-running a single task (D4 改造③)."""

    topic: str = Field(..., description="Research topic")
    task_id: int = Field(..., description="Which task to re-run")
    tasks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current state of all tasks (rebuilt server-side)",
    )


class KbAskRequest(BaseModel):
    """Payload for knowledge base Q&A (LangGraph orchestrated)."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    kb_id: str = "default"
    thread_id: str | None = None  # P4：对话线程 ID（同 ID 持久化对话状态）
    user_id: str | None = None  # P3 §3.4：用户 ID（传则校验对该 kb 的访问权）


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _key_matches(provided: Optional[str], expected: str) -> bool:
    """恒定时间字符串比较（防时序攻击），与明文 != 相比不泄露逐字符差异。"""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    # P1: 前端"研究深度"覆盖轮数上限（1=快速 / 2=标准 / 3=深度）
    if payload.research_depth is not None:
        overrides["max_web_research_loops"] = payload.research_depth

    return Configuration.from_env(overrides=overrides)


def create_app() -> FastAPI:
    app = FastAPI(title="HelloAgents Deep Researcher")

    # CORS：从配置读允许的来源（生产禁用 *；* + credentials 浏览器会拒）
    _cfg = Configuration.from_env()
    _cors_origins = [o.strip() for o in _cfg.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        # 有明确来源才允许 credentials，避免「* + credentials」非法组合
        allow_credentials=bool(_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # P1-3：request_id 注入——优先用调用方传入的 X-Request-Id，否则生成 uuid，
    # 回写响应头。配合 loguru 的 {extra[request_id]}，一次请求的日志可跨模块串联。
    @app.middleware("http")
    async def _inject_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        _request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    # ---- 深度研究接口鉴权 + 限流（防烧付费 API）----
    _admin_key = _cfg.admin_api_key
    _research_limiter = _SlidingWindowLimiter(max_requests=10, window_seconds=60)

    # P0：KB 问答限流——/kb/ask 每次都是 LLM 付费调用，不能裸奔。
    # 按身份（user_id 或 anonymous）分桶限流，防单用户刷爆费用。
    # KB_ASK_RATE_LIMIT：每分钟每身份最大请求数（默认 20）。
    _kb_ask_limiter = _SlidingWindowLimiter(
        max_requests=int(os.getenv("KB_ASK_RATE_LIMIT", "20") or 20),
        window_seconds=60,
    )

    def _check_research_access(request: Request) -> None:
        """深度研究接口的统一鉴权入口：可选 API key + 全局限流。"""
        if _admin_key and not _key_matches(request.headers.get("X-API-Key"), _admin_key):
            raise HTTPException(status_code=401, detail="无效的 API Key")
        if not _research_limiter.allow("research"):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        # uvicorn 启动时已用它的 LOGGING_CONFIG 重设过 root logger（加了 default handler），
        # 这里接管 root：只保留 loguru 桥接，移除 uvicorn 默认 handler，避免业务日志重复打印两遍。
        # uvicorn.access / uvicorn.error 是独立 logger（propagate=False），不受影响。
        _root = logging.getLogger()
        _root.handlers = [_InterceptHandler()]
        _root.setLevel(logging.INFO)

        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
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
        """就绪探针：探活下游依赖（SQLite 用户库 / Chroma / Ollama embedding）。

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
            kb["store"].list_kbs()
            checks["chroma"] = "ok"
        except Exception as exc:
            checks["chroma"] = f"error: {exc}"
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

    @app.get("/admin/diag")
    def admin_diag(request: Request) -> Dict[str, Any]:
        """运行时诊断——暴露研究调用的累计统计（token 消耗、调用次数、超限次数）。

        鉴权：若配置了 ADMIN_API_KEY 则要求 X-API-Key 头匹配（与 /research 相同）。
        """
        if _admin_key and not _key_matches(request.headers.get("X-API-Key"), _admin_key):
            raise HTTPException(status_code=401, detail="无效的 API Key")
        stats = global_stats.snapshot()
        stats["token_budget_limit"] = _cfg.research_token_budget
        stats["rate_limiter"] = f"{_research_limiter._window}s / {_research_limiter._max} req"
        return stats

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

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载：研究功能需 hello_agents
            config = _build_config(payload)
            budget = TokenBudget(limit=config.research_token_budget)
            global_stats.record_run_start(payload.topic)
            agent = DeepResearchAgent(config=config, token_budget=budget, stats=global_stats)
            result = agent.run(payload.topic)
        except TokenBudgetExceeded as exc:
            global_stats.record_budget_exceeded()
            raise HTTPException(
                status_code=429,
                detail=f"研究因 token 预算超限被终止（{exc.used}/{exc.limit} tokens）。请缩小研究范围或提高 KB_RESEARCH_TOKEN_BUDGET",
            ) from exc
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest, request: Request) -> StreamingResponse:
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载
            config = _build_config(payload)
            budget = TokenBudget(limit=config.research_token_budget)
            global_stats.record_run_start(payload.topic)
            agent = DeepResearchAgent(config=config, token_budget=budget, stats=global_stats)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    if event.get("type") == "budget_exceeded":
                        global_stats.record_budget_exceeded()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/research/regenerate")
    def regenerate_task(payload: RegenerateRequest, request: Request) -> StreamingResponse:
        """=D4 改造③= 只重新执行单个任务，然后基于所有任务重新生成报告。"""
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载
            config = Configuration.from_env()
            budget = TokenBudget(limit=config.research_token_budget)
            global_stats.record_run_start(payload.topic)
            agent = DeepResearchAgent(config=config, token_budget=budget, stats=global_stats)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.regenerate_task(
                    topic=payload.topic,
                    task_id=payload.task_id,
                    tasks_payload=payload.tasks,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Regenerate failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ==================== 知识库管理（KB）接口 ====================

    # KB 组件按需初始化（首次使用 KB 接口时创建，避免污染研究功能启动）
    _kb = {}
    _kb_init_lock = Lock()

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
            from services.kb.embeddings import EmbeddingClient
            from services.kb.vector_store import VectorStore
            from services.kb import qa_graph
            from openai import OpenAI

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

            store = VectorStore(
                persist_dir=cfg.kb_chroma_dir,
                collection_name=cfg.kb_collection,
                embedding_model=cfg.kb_embedding_model,
            )
            # P3：历史数据迁移——给无 kb_id 的 chunk 补默认值（幂等，仅首次执行实际写入）
            store.migrate_default_kb_id()

            # LLM（复用现有配置：DeepSeek）
            # P0：设默认超时——LLM/网络 hang 时不拖死请求（qa_graph 内单次调用也带 timeout）
            llm = OpenAI(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url or None,
                timeout=float(os.getenv("LLM_TIMEOUT", "60") or 60),
            )
            # model 显式传给 build_qa_graph（P1：去掉 _model 私有属性 hack）

            from langgraph.checkpoint.sqlite import SqliteSaver
            from services.kb.auth import AuthStore
            import sqlite3

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

            # 客服改造第1项：Rerank 重排器（llm / crossencoder / off 三模式）
            from services.kb.reranker import build_reranker
            reranker = build_reranker(
                os.getenv("KB_RERANK_MODE", "llm"),
                llm=llm,
                model=cfg.llm_model_id or "deepseek-chat",
            )

            graph = qa_graph.build_qa_graph(
                llm=llm,
                embeddings=embeddings,
                vector_store=store,
                top_k=cfg.kb_top_k,
                checkpointer=saver,
                model=cfg.llm_model_id or "deepseek-chat",
                min_score=float(os.getenv("KB_MIN_SIMILARITY", "0") or 0),
                faq_store=faq_store,
                faq_threshold=float(os.getenv("KB_FAQ_THRESHOLD", "0.8") or 0.8),
                reranker=reranker,
                persona_store=persona_store,
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
            return
        if required:
            if not kb["auth"].can_write(user_id, kb_id):
                raise HTTPException(
                    status_code=403,
                    detail=f"用户 {user_id} 无知识库 {kb_id} 的写权限（readonly 只读）",
                )
        else:
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

    # 🟠4：token 鉴权解析——X-Api-Token 头优先，user_id 直传回退（过渡期兼容）
    _REQUIRE_TOKEN = bool(os.getenv("KB_REQUIRE_TOKEN", "").strip() not in ("", "0", "false"))

    # P0-1：自助注册开关。KB_OPEN_SIGNUP=1 时允许无鉴权建号，但角色强制 member
    # （杜绝 admin 提权）。默认关闭——生产必须走 X-API-Key 或 admin token 建号。
    _OPEN_SIGNUP = bool(os.getenv("KB_OPEN_SIGNUP", "").strip() in ("1", "true", "yes"))

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
        from services.kb.ingest import build_chunks

        suffix = Path(file.filename or "upload").suffix
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
            chunks = build_chunks(
                tmp_path,
                doc_id=doc_id,
                chunk_size=cfg.kb_chunk_size,
                overlap=cfg.kb_chunk_overlap,
                kb_id=kb_id,
                ocr_mode=ocr_mode,
            )
            if not chunks:
                raise HTTPException(status_code=400, detail="文档解析后无有效内容")
            resolved_title = (title or "").strip() or tmp_path.stem
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
        )

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
            dup_doc = kb["store"].find_doc_by_hash(content_hash)
            if dup_doc:
                raise HTTPException(
                    status_code=409,
                    detail=f"文档内容已存在（doc_id={dup_doc}），请勿重复上传",
                )
            _write_chunks(
                kb, chunks=chunks, vectors=vectors, doc_id=doc_id,
                doc_title=resolved_title, kb_id=kb_id, content_hash=content_hash,
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
            from services.kb import qa_graph

            start = time.time()
            result = qa_graph.run_qa(
                kb["graph"],
                question=payload.question,
                history=payload.history,
                kb_id=payload.kb_id,
                thread_id=payload.thread_id,
            )
            # 客服改造第3项：检索日志留痕（答错时回放定位）
            meta = result.get("search_meta", {})
            kb["retrieval_log"].record(
                kb_id=payload.kb_id,
                thread_id=payload.thread_id or "",
                question=payload.question,
                rewritten=meta.get("rewritten", ""),
                rerank_mode=os.getenv("KB_RERANK_MODE", "llm"),
                answerable=not meta.get("escalate", False),
                evidence_score=meta.get("top_score", 0.0),
                escalate=meta.get("escalate", False),
                attempts=meta.get("attempts", 1),
                latency_ms={"total": (time.time() - start) * 1000.0},
                hits=meta.get("recall_raw", []),
                final_hits=meta.get("contexts", []),
                faithfulness=result.get("score", 0),
                answer=result.get("answer", ""),
            )
            # 客服改造第9项：转人工——escalate 时建会话 + 进待接入池
            if result.get("escalate"):
                conv = kb["conversation_store"].get_or_create(
                    payload.thread_id or uuid.uuid4().hex,
                    kb_id=payload.kb_id,
                    visitor_id=user_id or "anonymous",
                )
                kb["conversation_store"].add_message(conv["id"], "user", payload.question)
                kb["conversation_store"].add_message(
                    conv["id"], "assistant", result.get("answer", "")
                )
                transferred = kb["conversation_store"].transfer_to_human(
                    conv["id"], "可回答性门槛两次不过"
                )
                if transferred:
                    kb["audit"].record(
                        user_id=user_id or "anonymous",
                        action="transfer_to_human",
                        target=str(conv["id"]),
                    )
                result["conversation_id"] = conv["id"]
                result["status"] = "waiting"
            return result
        except HTTPException:
            raise  # 403 等 HTTP 异常直接抛出，不被转 500
        except Exception as exc:
            logger.error("KB ask failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc

    @app.get("/kb/conversation/{conv_id}/status")
    def kb_conversation_status(conv_id: int) -> Dict[str, Any]:
        """查会话状态（访客转人工后轮询坐席是否接入用，无鉴权）。

        返回 {status: ai/waiting/human/closed, agent_id}。会话不存在 404。
        """
        kb = _get_kb()
        conv = kb["conversation_store"].get(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"conversation_id": conv_id, "status": conv["status"], "agent_id": conv["agent_id"]}

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
        except HTTPException:
            raise

        from services.kb import qa_graph

        def event_iterator() -> Iterator[str]:
            start = time.time()
            final_event: Dict[str, Any] | None = None
            try:
                for event in qa_graph.run_qa_stream(
                    kb["graph"],
                    question=payload.question,
                    history=payload.history,
                    kb_id=payload.kb_id,
                    thread_id=payload.thread_id,
                ):
                    if event.get("type") == "final":
                        final_event = event
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # P1：final 之后补齐与同步 /kb/ask 一致的检索日志落库 + escalate 转人工闭环
                if final_event is not None:
                    meta = final_event.get("search_meta", {})
                    kb["retrieval_log"].record(
                        kb_id=payload.kb_id,
                        thread_id=payload.thread_id or "",
                        question=payload.question,
                        rewritten=meta.get("rewritten", ""),
                        rerank_mode=os.getenv("KB_RERANK_MODE", "llm"),
                        answerable=not meta.get("escalate", False),
                        evidence_score=meta.get("top_score", 0.0),
                        escalate=meta.get("escalate", False),
                        attempts=meta.get("attempts", 1),
                        latency_ms={"total": (time.time() - start) * 1000.0},
                        hits=meta.get("recall_raw", []),
                        final_hits=meta.get("contexts", []),
                        faithfulness=final_event.get("score", 0),
                        answer=final_event.get("answer", ""),
                    )
                    if final_event.get("escalate"):
                        conv = kb["conversation_store"].get_or_create(
                            payload.thread_id or uuid.uuid4().hex,
                            kb_id=payload.kb_id,
                            visitor_id=user_id or "anonymous",
                        )
                        kb["conversation_store"].add_message(conv["id"], "user", payload.question)
                        kb["conversation_store"].add_message(
                            conv["id"], "assistant", final_event.get("answer", "")
                        )
                        transferred = kb["conversation_store"].transfer_to_human(
                            conv["id"], "可回答性门槛两次不过"
                        )
                        if transferred:
                            kb["audit"].record(
                                user_id=user_id or "anonymous",
                                action="transfer_to_human",
                                target=str(conv["id"]),
                            )
            except Exception as exc:
                logger.exception("KB ask stream failed")
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

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
                allowed = set(kb["auth"].get_allowed_kbs(user_id))
                kbs = [k for k in kbs if k in allowed]
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

    # ==================== 用户与权限管理（RBAC，P3 §3.4）====================

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
            cur2 = conn.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
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
            kb_id = str(payload.get("kb_id", "default"))[:64]
            question = str(payload.get("question", ""))[:500]
            answer = str(payload.get("answer", ""))[:500]
            rating = 1 if payload.get("rating", 1) else 0
            comment = str(payload.get("comment", ""))[:500]
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
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"conversations": kb["conversation_store"].list_by_status("waiting")}
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["conversation_store"].get(conv_id):
                raise HTTPException(status_code=404, detail="会话不存在")
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            conv = kb["conversation_store"].get(conv_id)
            if not conv:
                raise HTTPException(status_code=404, detail="会话不存在")
            if conv["status"] != "human":
                raise HTTPException(status_code=409, detail="会话未在人工服务中")
            if conv["agent_id"] and conv["agent_id"] != operator:
                raise HTTPException(status_code=403, detail="该会话由其他坐席处理")
            kb["conversation_store"].add_message(conv_id, "agent", content)
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["conversation_store"].get(conv_id):
                raise HTTPException(status_code=404, detail="会话不存在")
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
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            if not kb["conversation_store"].get(conv_id):
                raise HTTPException(status_code=404, detail="会话不存在")
            kb["conversation_store"].set_tag(conv_id, tag)
            kb["audit"].record(user_id=operator, action="agent_tag", target=str(conv_id))
            return {"conversation_id": conv_id, "tag": tag}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB agent tag failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"打标签失败: {exc}") from exc

    # ==================== 工单（第10项）====================

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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            ticket = kb["ticket_store"].create(
                conversation_id=payload.get("conversation_id"),
                title=payload.get("title", ""),
                description=payload.get("description", ""),
            )
            kb["audit"].record(user_id=operator, action="create_ticket", target=ticket["ticket_no"])
            return ticket
        except HTTPException:
            raise
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
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            return {"tickets": kb["ticket_store"].list(status=status)}
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
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            ticket = kb["ticket_store"].get(ticket_id)
            if not ticket:
                raise HTTPException(status_code=404, detail="工单不存在")
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
            _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
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
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """建知识库（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            meta = kb["kb_meta_store"].create(kb_id, name, description)
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
        admin_id: str | None = Query(default=None),
        x_api_token: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Dict[str, Any]:
        """改知识库名称/描述（需管理员）。"""
        try:
            kb = _get_kb()
            operator = _require_kb_admin(kb, x_api_key, x_api_token, admin_id)
            kb["kb_meta_store"].update(kb_id, name=name, description=description)
            kb["audit"].record(user_id=operator, action="update_kb", target=kb_id)
            return kb["kb_meta_store"].get(kb_id)
        except HTTPException:
            raise
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
