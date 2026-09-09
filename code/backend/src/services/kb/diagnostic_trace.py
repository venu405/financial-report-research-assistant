"""诊断追踪（默认关闭）——失败题候选级取证，不参与任何生产决策。

为什么单独一个模块？
  正式评测 JSON 只保存聚合指标（hit@K / page_hit / evidence_rate），无法回答
  "目标页到底进没进原始召回""是在精排还是 top-K 截断时丢的""核验拒绝了哪条
  claim"。本模块在检索与生成链路上埋点，把这些中间态落盘，供离线归因使用。

设计约束（必须保持）：
  - **默认关闭**：只有 KB_DIAGNOSTIC_TRACE_ENABLED=1 且 KB_DIAGNOSTIC_TRACE_PATH
    非空时才工作；未启用时所有入口都是 no-op，调用方用 `if is_enabled()` 守卫，
    连 payload 构造都不会发生，因此零额外开销、零行为差异。
  - **不新增模型调用**：只记录现有流程已经算出来的结果，绝不为了追踪重跑
    embedding / rerank / LLM。
  - **不改接口响应**：追踪完全旁路，不写回 state，不影响返回值。
  - **多请求不覆盖**：JSONL 追加写，一行一个请求。
  - **不落密钥**：键名命中敏感词的值一律脱敏；只记录配置"值"（如 top_k），
    不记录任何环境变量原始内容以外的凭据。

载体为什么是"token 注册表"而不是 ContextVar（重要，改回去会再次静默失效）：
  LangGraph 的同步 invoke 会把每个节点放进事件循环里各自独立的 Task 执行，
  而 Task 在创建时**复制**当前 contextvars 上下文。实测（真实 StateGraph，
  两个空壳节点）结论是：
    - 节点 A 里 `_current.set(...)`，节点 B **看不到**；
    - 图跑完后，外层调用方 `run_qa` **也看不到**。
  于是"节点里 start_request → 外层 finish_request"这条链路全程 no-op：
  不落盘、不报错、连告警都没有（is_enabled 为 True，只是拿不到记录）。
  因此记录必须放在**跨上下文共享**的地方：模块级 `_records[token]`，
  token 由图状态的 trace_id 携带，外层用同一个 token 收尾。
  ContextVar 只降级为"同一节点内"的快捷指针——retriever 的埋点就在
  node_retrieve 的调用栈里，同属一个上下文，用它能少传一层参数。

产出格式：每行一个 JSON 对象
  {trace_id, ts_start, ts_end, basic:{...}, stages:[{stage, ts, data}], outcome:{...}}
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import threading
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENABLED_ENV = "KB_DIAGNOSTIC_TRACE_ENABLED"
PATH_ENV = "KB_DIAGNOSTIC_TRACE_PATH"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# ---- 体积上限：追踪是给人看片段的，不是把整份年报写进日志 ----
MAX_CANDIDATE_TEXT = 400
MAX_ANSWER_TEXT = 2000
MAX_REASON_TEXT = 400
MAX_QUESTION_TEXT = 500
MAX_CANDIDATES = 40
MAX_CLAIMS = 60
MAX_LIST_ITEMS = 40
MAX_STRING = 4000
MAX_SCRUB_DEPTH = 6
# 注册表兜底容量：正常流程 finish_request 会 pop，这里只防异常路径下的无限增长。
MAX_PENDING_RECORDS = 64

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|authorization|cookie"
    r"|signature|private[_-]?key|access[_-]?key|bearer)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"
_TRUNCATED_MARK = "...[truncated]"

_lock = threading.Lock()
# 请求级主载体：跨 LangGraph 节点/上下文共享，key 是图状态里的 trace_id。
# 见模块顶部"载体为什么是 token 注册表"——这里绝不能只靠 ContextVar。
_records: dict[str, dict[str, Any]] = {}
# 节点内快捷指针：只在同一节点、同一调用栈内有效（retriever 埋点靠它）。
# 跨节点/跨到 run_qa 一律失效，所以收尾必须走 _records[token]。
_current: ContextVar[dict[str, Any] | None] = ContextVar(
    "kb_diagnostic_trace_record", default=None
)


# ---------------------------------------------------------------- 开关与生命周期


def is_enabled() -> bool:
    """追踪是否启用。每次调用都实时读环境变量，便于测试期开关。"""
    if str(os.getenv(ENABLED_ENV, "") or "").strip().casefold() not in _TRUTHY:
        return False
    return bool(str(os.getenv(PATH_ENV, "") or "").strip())


def trace_path() -> str:
    return str(os.getenv(PATH_ENV, "") or "").strip()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def start_request(basic: dict[str, Any] | None = None, token: str = "") -> str:
    """开启一次请求级追踪，返回 trace_id；未启用返回空串。

    token：图状态里携带的请求标识（QaState["trace_id"]）。给了 token 就走
    注册表，节点之间、节点与 run_qa 之间都能拿到同一条记录；不给 token 则
    退化为纯 ContextVar 行为（仅供单节点内的离线测试/旁路调用）。

    若同一个 token 上还有没收尾的旧记录（异常路径），先把它刷盘再开新的，
    保证"多个请求不会互相覆盖"。
    """
    if not is_enabled():
        return ""
    try:
        record = _new_record(basic)
        with _lock:
            key = str(token or "").strip()
            if key:
                previous = _records.pop(key, None)
                _records[key] = record
                # 只弹出、不落盘；落盘必须等退出锁之后（见 _evict_overflow_locked 注释）。
                stale = _evict_overflow_locked()
            else:
                previous = _current.get()
                stale = []
        # 锁外落盘：这里再拿 _lock 是安全的，因为外层已经释放。
        if stale:
            _flush_stale(stale)
        if previous is not None:
            _flush(previous, truncated=True)
        _current.set(record)
        return str(record["trace_id"])
    except Exception as exc:  # noqa: BLE001 —— 追踪绝不阻断主流程
        logger.warning("诊断追踪启动失败：%s", exc)
        return ""


def bind(token: str) -> bool:
    """把注册表里的记录挂到当前上下文，供同一节点内的调用栈（如 retriever）使用。

    返回是否绑定成功。节点入口先调 ensure_started/bind，之后同一栈内的
    add_stage 才能命中记录——因为跨节点时 ContextVar 必然失效。
    """
    if not is_enabled():
        return False
    key = str(token or "").strip()
    if not key:
        return _current.get() is not None
    try:
        with _lock:
            record = _records.get(key)
        if record is None:
            return False
        _current.set(record)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("诊断追踪绑定失败（token=%s）：%s", key, exc)
        return False


def ensure_started(basic: dict[str, Any] | None = None, token: str = "") -> str:
    """同一次请求内只开一次追踪；已有活动记录则直接复用。

    gate 会在证据分不足时回到 retrieve 重跑，若每次都 start_request 会把一次
    问答拆成多条记录。这里只在没有活动记录时才开启。
    """
    if not is_enabled():
        return ""
    key = str(token or "").strip()
    with _lock:
        record = _records.get(key) if key else _current.get()
    if record is not None:
        _current.set(record)
        return str(record.get("trace_id", ""))
    return start_request(basic, token=key)


def add_stage(stage: str, payload: dict[str, Any] | None = None) -> None:
    """追加一个阶段快照。未启用或无活动请求时直接返回。

    依赖 `_current`：调用方节点必须先 ensure_started/bind 把记录挂上。
    """
    if not is_enabled():
        return
    record = _current.get()
    if record is None:
        return
    try:
        record["stages"].append(
            {"stage": stage, "ts": _utc_now(), "data": _scrub(payload or {})}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("诊断追踪阶段写入失败（stage=%s）：%s", stage, exc)


def finish_request(outcome: dict[str, Any] | None = None, token: str = "") -> None:
    """收尾并刷盘。token 必须与 start/ensure 时一致（图状态的 trace_id）。"""
    if not is_enabled():
        return
    key = str(token or "").strip()
    record: dict[str, Any] | None = None
    try:
        if key:
            with _lock:
                record = _records.pop(key, None)
            if record is None:
                # 给了 token 却对不上：本次收尾作废，绝不退回上下文指针——
                # 否则会误把别人的在途记录刷盘并清掉，造成"张冠李戴"。
                return
        else:
            # 没给 token：退回上下文指针（单节点/离线路径）
            record = _current.get()
            _current.set(None)
            if record is not None:
                with _lock:
                    for existing_key, existing in list(_records.items()):
                        if existing is record:
                            _records.pop(existing_key, None)
                            break
    except Exception as exc:  # noqa: BLE001
        logger.warning("诊断追踪收尾取记录失败（token=%s）：%s", key, exc)
        return
    if record is None:
        return
    if _current.get() is record:
        _current.set(None)
    _flush(record, outcome=outcome)


def error_outcome(exc: BaseException, status: str = "exception") -> dict[str, Any]:
    """异常收尾专用的 outcome：只留错误类型 + 受限长度摘要。

    调用方拿到它刷盘后**必须把原异常重新抛出**——追踪只是旁路，绝不能吞掉
    异常或改变接口行为。摘要走 `clip_text`，避免把超长堆栈/整段提示词写进文件。
    """
    return {
        "status": status,
        "error_type": type(exc).__name__,
        "error": clip_text(exc, MAX_REASON_TEXT),
    }


def pending_count() -> int:
    """未收尾记录数（测试/自检用）。"""
    with _lock:
        return len(_records)


def reset_state() -> None:
    """清空注册表与上下文指针（仅测试用）。"""
    with _lock:
        _records.clear()
    _current.set(None)


def _new_record(basic: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "trace_id": uuid.uuid4().hex[:12],
        "ts_start": _utc_now(),
        "basic": _scrub(basic or {}),
        "stages": [],
    }


def _evict_overflow_locked() -> list[dict[str, Any]]:
    """异常路径下注册表可能堆积，超限就把最老的挑出来，避免内存泄漏。

    **只在锁内弹出，不落盘**：`_flush()` 自己会再获取 `_lock`，而 `_lock` 是
    不可重入的普通 Lock——在持锁期间调用 `_flush()` 会直接死锁（历史缺陷）。
    所以这里把待清理记录返回，由调用方退出锁之后再逐个 `_flush`。
    （不要改成 RLock：那只是掩盖"持锁期间做 IO"的结构问题。）

    dict 保持插入顺序，因此 keys()[:overflow] 就是最老的几条。
    """
    if len(_records) <= MAX_PENDING_RECORDS:
        return []
    overflow = len(_records) - MAX_PENDING_RECORDS
    stale: list[dict[str, Any]] = []
    for key in list(_records.keys())[:overflow]:
        record = _records.pop(key, None)
        if record is not None:
            stale.append(record)
    return stale


def _flush_stale(stale: list[dict[str, Any]]) -> None:
    """把 `_evict_overflow_locked` 弹出的记录落盘。**必须在锁外调用。**"""
    for record in stale:
        _flush(record, truncated=True)


def _flush(
    record: dict[str, Any],
    *,
    truncated: bool = False,
    outcome: dict[str, Any] | None = None,
) -> None:
    try:
        record["ts_end"] = _utc_now()
        if truncated:
            record["truncated"] = True
        if outcome:
            record["outcome"] = _scrub(outcome)
        line = json.dumps(record, ensure_ascii=False, default=str)
        path = Path(trace_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("诊断追踪落盘失败：%s", exc)


# ---------------------------------------------------------------- 脱敏与裁剪


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATED_MARK


def clip_text(value: Any, limit: int = MAX_CANDIDATE_TEXT) -> str:
    """公开裁剪入口：调用方截断长文本，避免把整份文档写进追踪文件。"""
    return _clip(value, limit)


def _scrub(value: Any, _depth: int = 0) -> Any:
    """递归脱敏 + 限长。键名命中敏感词的值一律替换，字符串超长一律截断。"""
    if _depth > MAX_SCRUB_DEPTH:
        return "<max-depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_LIST_ITEMS * 4:
                out["<more-keys-omitted>"] = True
                break
            name = str(key)
            out[name] = _REDACTED if _SENSITIVE_KEY_RE.search(name) else _scrub(item, _depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [_scrub(item, _depth + 1) for item in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            result.append(f"<{len(items) - MAX_LIST_ITEMS} more omitted>")
        return result
    if isinstance(value, str):
        return _clip(value, MAX_STRING)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return _clip(value, MAX_STRING)


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


# ---------------------------------------------------------------- 结构化快照


def _table_flag(metadata: dict[str, Any]) -> bool:
    raw = metadata.get("is_table")
    return raw is True or str(raw or "").strip().casefold() in {"true", "1", "yes"}


def _stored_metrics(metadata: dict[str, Any]) -> set[str]:
    return {
        item.strip()
        for item in str(metadata.get("financial_metrics") or "").split("|")
        if item.strip()
    }


def candidate_record(hit: dict[str, Any], rank: int, source: str) -> dict[str, Any]:
    """把一条检索候选压成追踪用的最小可判读结构。"""
    metadata = hit.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "rank": rank,
        "source": source,
        "chunk_id": str(hit.get("chunk_id") or ""),
        "rrf_score": _round(hit.get("rrf_score")),
        "vec_score": _round(hit.get("score")),
        "rerank_score": _round(hit.get("rerank_score")),
        "kb_id": str(metadata.get("kb_id") or ""),
        "doc_id": str(metadata.get("doc_id") or ""),
        "company": str(metadata.get("company") or ""),
        "report_period": str(metadata.get("report_period") or ""),
        "doc_title": _clip(metadata.get("doc_title"), 120),
        "page": metadata.get("page") or metadata.get("page_start"),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "section_path": _clip(metadata.get("section_path"), 160),
        "is_table": _table_flag(metadata),
        "table_name": _clip(metadata.get("table_name"), 80),
        "financial_metrics": sorted(_stored_metrics(metadata)),
        "has_financial_facts": bool(metadata.get("financial_facts_json")),
        "text": _clip(hit.get("text"), MAX_CANDIDATE_TEXT),
    }


def candidate_snapshot(
    hits: list[Any],
    source: str,
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    return [
        candidate_record(hit, index + 1, source)
        for index, hit in enumerate(list(hits)[:limit])
    ]


def ranking_ids(hits: list[Any], limit: int = MAX_CANDIDATES) -> list[str]:
    """只取排序序列（chunk_id），用于比对精排/截断前后的位置变化。"""
    out: list[str] = []
    for hit in list(hits)[:limit]:
        if not isinstance(hit, dict):
            out.append("<non-dict>")
            continue
        out.append(str(hit.get("chunk_id") or f"<no-chunk-id@{len(out)}>"))
    return out


def claim_record(claim: Any) -> dict[str, Any]:
    """NumericClaim → 追踪字典（含 metric/unit/period/scope/adjustment 绑定结果）。"""
    return {
        "raw": _clip(getattr(claim, "raw", ""), 60),
        "metric": str(getattr(claim, "metric", "") or ""),
        "line_metric": str(getattr(claim, "line_metric", "") or ""),
        "unit": str(getattr(claim, "unit", "") or ""),
        "report_period": str(getattr(claim, "report_period", "") or ""),
        "statement_scope": str(getattr(claim, "statement_scope", "") or ""),
        "adjustment": str(getattr(claim, "adjustment", "") or ""),
        "is_percent": bool(getattr(claim, "is_percent", False)),
        "canonical_value": str(getattr(claim, "canonical_value", "") or ""),
    }


def claim_snapshot(claims: list[Any], limit: int = MAX_CLAIMS) -> list[dict[str, Any]]:
    return [claim_record(claim) for claim in list(claims)[:limit]]
