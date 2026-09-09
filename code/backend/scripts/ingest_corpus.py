"""企业知识库 PDF 一键增量入库。

脚本只负责发现文件变化、记录可续跑状态并调用现有入库接口；不会删除服务端
文档，也不会把访问令牌写入状态清单或日志。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import dotenv_values

SCRIPT_VERSION = "ingest-corpus-v1"
BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TESTSET = BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml"
DEFAULT_STATE = BACKEND_DIR / ".rag_state" / "ingest_corpus_state.json"
DEFAULT_KB_ID = "cninfo_report"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 1800.0
HASH_CHUNK_SIZE = 1024 * 1024
REMOTE_PAGE_SIZE = 500
SUCCESS_STATUSES = {"success", "skipped"}
FINGERPRINT_ENV_KEYS = (
    "KB_CHUNK_PROFILE",
    "KB_CHUNK_SIZE",
    "KB_CHUNK_OVERLAP",
    "KB_PARENT_CHILD_ENABLED",
    "KB_PARENT_CHUNK_SIZE",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_MODE",
    "KB_NEIGHBOR_EXPANSION",
)


class CorpusError(RuntimeError):
    """用户可以直接处理的输入、状态或服务错误。"""


class ServiceError(CorpusError):
    """服务健康检查失败。"""


@dataclass
class DocumentInput:
    path: Path
    title: str
    sha256: str = ""
    size: int = 0
    mtime: int = 0
    error: str = ""


@dataclass
class PlannedDocument:
    document: DocumentInput
    action: str
    reason: str
    previous: dict[str, Any]
    pipeline_changed: bool = False
    remote_verified: bool = False


@dataclass
class RequestOutcome:
    success: bool
    doc_id: str = ""
    status_code: int | None = None
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_backend_path(value: str | Path, backend_dir: Path = BACKEND_DIR) -> Path:
    """把相对路径按后端目录解释，再得到规范绝对路径。"""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = backend_dir / candidate
    return candidate.resolve()


def state_key(kb_id: str, path: Path) -> str:
    """按知识库和规范绝对路径生成稳定清单键。"""
    normalized = os.path.normcase(str(path.resolve()))
    return f"{kb_id}|{normalized}"


def _source_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "source", "file", "filename"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def collect_documents(
    *,
    directory: str | Path | None = None,
    testset: str | Path | None = None,
    backend_dir: Path = BACKEND_DIR,
) -> list[DocumentInput]:
    """从目录递归扫描，或从题集 expected_sources 去重收集 PDF。"""
    if directory is not None and testset is not None:
        raise CorpusError("--directory 和 --testset 只能二选一")
    if directory is not None:
        root = resolve_backend_path(directory, backend_dir)
        if not root.is_dir():
            raise CorpusError(f"PDF 目录不存在或不是目录：{root}")
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: str(path).casefold(),
        )
    else:
        testset_path = resolve_backend_path(testset or DEFAULT_TESTSET, backend_dir)
        if not testset_path.is_file():
            raise CorpusError(f"题集文件不存在：{testset_path}")
        try:
            payload = yaml.safe_load(testset_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise CorpusError(f"题集读取失败：{testset_path}；{exc}") from exc
        if not isinstance(payload, Mapping):
            raise CorpusError(f"题集格式不是对象：{testset_path}")
        paths = []
        for case in payload.get("tests", []):
            if not isinstance(case, Mapping):
                continue
            for raw_source in case.get("expected_sources", []) or []:
                source = _source_text(raw_source)
                if source:
                    paths.append(resolve_backend_path(source, backend_dir))
        paths = _dedupe_paths(paths)
    return [DocumentInput(path=path, title=path.stem) for path in paths]


def snapshot_document(document: DocumentInput) -> DocumentInput:
    """流式计算 PDF SHA-256，同时记录大小和修改时间。"""
    try:
        digest = hashlib.sha256()
        with document.path.open("rb") as stream:
            while True:
                block = stream.read(HASH_CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
        stat = document.path.stat()
        document.sha256 = digest.hexdigest()
        document.size = stat.st_size
        document.mtime = stat.st_mtime_ns
    except Exception as exc:
        document.error = f"文件读取或 SHA-256 计算失败：{type(exc).__name__}: {exc}"
    return document


def snapshot_documents(documents: list[DocumentInput]) -> list[DocumentInput]:
    return [snapshot_document(document) for document in documents]


def _fingerprint_file(hasher: Any, path: Path, label: str) -> None:
    hasher.update(f"FILE:{label}\0".encode("utf-8"))
    if not path.is_file():
        hasher.update(b"MISSING\0")
        return
    with path.open("rb") as stream:
        while True:
            block = stream.read(HASH_CHUNK_SIZE)
            if not block:
                break
            hasher.update(block)


def compute_pipeline_fingerprint(
    backend_dir: Path = BACKEND_DIR,
    *,
    ocr_mode: str = "local",
) -> str:
    """覆盖解析器、OCR、脚本版本和关键分块配置的指纹。"""
    hasher = hashlib.sha256()
    dot_env = dotenv_values(backend_dir / ".env") if (backend_dir / ".env").is_file() else {}
    config = {
        key: os.getenv(key) if os.getenv(key) is not None else str(dot_env.get(key) or "")
        for key in FINGERPRINT_ENV_KEYS
    }
    config.update({"ocr_mode": ocr_mode, "script_version": SCRIPT_VERSION})
    hasher.update(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for relative in (
        "src/services/kb/ingest.py",
        "src/services/kb/ocr.py",
        "scripts/ingest_corpus.py",
    ):
        _fingerprint_file(hasher, backend_dir / relative, relative)
    return hasher.hexdigest()


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "updated_at": None,
        "last_run": None,
        "entries": {},
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorpusError(f"状态清单读取失败，已保护原文件：{path}；{exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        raise CorpusError(f"状态清单格式不受支持，拒绝覆盖：{path}")
    payload.setdefault("schema_version", 1)
    payload.setdefault("script_version", SCRIPT_VERSION)
    payload.setdefault("updated_at", None)
    payload.setdefault("last_run", None)
    return payload


def save_state_atomic(path: Path, state: dict[str, Any]) -> bool:
    """同目录临时文件 + fsync + os.replace，失败时保留旧清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        return True
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _base_entry(document: DocumentInput, kb_id: str) -> dict[str, Any]:
    return {
        "kb_id": kb_id,
        "path": str(document.path.resolve()),
        "sha256": document.sha256,
        "size": document.size,
        "mtime": document.mtime,
        "title": document.title,
        "doc_id": "",
        "last_status": "failed",
        "last_action": "FAIL",
        "indexed_at": None,
        "pipeline_fingerprint": "",
        "last_error": "",
        "last_http_status": None,
    }


def plan_documents(
    documents: list[DocumentInput],
    state: dict[str, Any],
    *,
    kb_id: str,
    pipeline_fingerprint: str,
    force: bool = False,
) -> list[PlannedDocument]:
    plans: list[PlannedDocument] = []
    entries = state.setdefault("entries", {})
    for document in documents:
        key = state_key(kb_id, document.path)
        previous = dict(entries.get(key) or {})
        if document.error:
            plans.append(PlannedDocument(document, "FAIL", "input-error", previous))
            continue
        doc_id = str(previous.get("doc_id") or "")
        same_hash = bool(previous.get("sha256")) and previous.get("sha256") == document.sha256
        successful = previous.get("last_status") in SUCCESS_STATUSES
        pipeline_changed = successful and same_hash and (
            previous.get("pipeline_fingerprint") != pipeline_fingerprint
        )
        if force:
            action = "UPDATE" if doc_id else "NEW"
            reason = "force"
        elif successful and same_hash and doc_id:
            action = "SKIP"
            reason = "sha256-unchanged"
        elif doc_id:
            action = "UPDATE"
            reason = "previous-failure" if not successful else "content-changed"
        else:
            action = "NEW"
            reason = "previous-failure" if previous else "not-indexed"
        plans.append(PlannedDocument(document, action, reason, previous, pipeline_changed))
    return plans


def _safe_response_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        detail = payload.get("detail") or payload.get("error") or payload.get("message")
        if detail:
            return str(detail)[:500]
    text = str(getattr(response, "text", "") or "").strip()
    return text[:500] or "服务端没有返回错误详情"


def check_service(
    session: Any,
    base_url: str,
    *,
    timeout: float,
) -> None:
    endpoint = f"{base_url.rstrip('/')}/readyz"
    try:
        response = session.get(endpoint, timeout=(10.0, min(timeout, 30.0)))
    except Exception as exc:
        raise ServiceError(
            f"服务健康检查失败：无法连接 {endpoint}。请先运行现有“一键启动前后端.cmd”，"
            f"确认后端已启动；原始错误：{type(exc).__name__}: {exc}"
        ) from exc
    if not 200 <= int(response.status_code) < 300:
        detail = _safe_response_detail(response)
        raise ServiceError(
            f"服务健康检查失败：{endpoint} 返回 HTTP {response.status_code}，{detail}。"
            "请先运行现有“一键启动前后端.cmd”或检查后端日志。"
        )


def _request_headers() -> dict[str, str]:
    token = os.getenv("KB_INGEST_TOKEN") or os.getenv("KB_API_TOKEN")
    if token:
        return {"X-Api-Token": token}
    return {}


def fetch_remote_documents(
    session: Any,
    base_url: str,
    *,
    kb_id: str,
    user_id: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    """读取目标知识库的完整文档清单，失败时拒绝继续写入。

    该清单是本次运行的服务端事实来源。不能只读取第一页，也不能把一个
    失败的 GET 当成空库，否则状态清单中的旧 doc_id 可能被误用为 PUT。
    """
    endpoint = f"{base_url.rstrip('/')}/kb/docs"
    documents: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    offset = 0
    expected_total: int | None = None
    headers = _request_headers()
    while True:
        params: dict[str, Any] = {
            "kb_id": kb_id,
            "limit": REMOTE_PAGE_SIZE,
            "offset": offset,
        }
        if user_id:
            params["user_id"] = user_id
        try:
            response = session.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=(10.0, min(timeout, 30.0)),
            )
        except Exception as exc:
            raise ServiceError(
                f"远端文档对账失败：无法读取 {endpoint}（offset={offset}）；"
                f"{type(exc).__name__}: {exc}。已终止，未执行任何入库请求。"
            ) from exc
        status_code = int(response.status_code)
        if not 200 <= status_code < 300:
            detail = _safe_response_detail(response)
            raise ServiceError(
                f"远端文档对账失败：{endpoint}?kb_id={kb_id}&offset={offset} "
                f"返回 HTTP {status_code}：{detail}。已终止，未执行任何入库请求。"
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise ServiceError(
                f"远端文档对账失败：offset={offset} 的响应不是有效 JSON；{exc}。"
                "已终止，未执行任何入库请求。"
            ) from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("docs"), list):
            raise ServiceError(
                f"远端文档对账失败：offset={offset} 的响应缺少 docs 列表。"
                "已终止，未执行任何入库请求。"
            )
        raw_total = payload.get("total")
        if raw_total is not None:
            if isinstance(raw_total, bool):
                raise ServiceError("远端文档对账失败：total 不是有效整数。")
            try:
                page_total = int(raw_total)
            except (TypeError, ValueError) as exc:
                raise ServiceError("远端文档对账失败：total 不是有效整数。") from exc
            if page_total < 0:
                raise ServiceError("远端文档对账失败：total 不能为负数。")
            if expected_total is None:
                expected_total = page_total
            elif expected_total != page_total:
                raise ServiceError("远端文档对账失败：分页返回的 total 不一致。")

        page = payload["docs"]
        for item in page:
            if not isinstance(item, Mapping):
                raise ServiceError("远端文档对账失败：docs 含有非对象项。")
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                raise ServiceError("远端文档对账失败：文档项缺少 doc_id。")
            if doc_id in seen_doc_ids:
                raise ServiceError(f"远端文档对账失败：doc_id 重复返回：{doc_id}。")
            returned_kb_id = str(item.get("kb_id") or "").strip()
            if returned_kb_id and returned_kb_id != kb_id:
                raise ServiceError(
                    f"远端文档对账失败：返回了不属于目标知识库的 doc_id={doc_id}。"
                )
            try:
                chunks = int(item.get("chunks"))
            except (TypeError, ValueError) as exc:
                raise ServiceError(
                    f"远端文档对账失败：doc_id={doc_id} 的 chunks 无效。"
                ) from exc
            normalized = dict(item)
            normalized["doc_id"] = doc_id
            normalized["title"] = str(item.get("title") or "").strip()
            normalized["chunks"] = chunks
            normalized["kb_id"] = returned_kb_id or kb_id
            documents.append(normalized)
            seen_doc_ids.add(doc_id)

        offset += len(page)
        if not page:
            if expected_total is not None and offset < expected_total:
                raise ServiceError(
                    "远端文档对账失败：分页提前结束，未取得目标库完整清单。"
                )
            break
        if expected_total is not None and offset >= expected_total:
            if offset > expected_total:
                raise ServiceError("远端文档对账失败：分页数量超过 total。")
            break
        if expected_total is None and len(page) < REMOTE_PAGE_SIZE:
            break
    if expected_total is not None and len(documents) != expected_total:
        raise ServiceError(
            f"远端文档对账失败：期望 {expected_total} 个文档，实际取得 {len(documents)} 个。"
        )
    return documents


def _title_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def reconcile_plans(
    plans: list[PlannedDocument],
    *,
    state_exists: bool,
    remote_documents: list[dict[str, Any]],
    kb_id: str,
    force: bool = False,
) -> list[PlannedDocument]:
    """把本地计划与目标知识库事实对账，阻断不安全的 PUT/SKIP。"""
    by_id = {str(item.get("doc_id")): item for item in remote_documents}
    by_title: dict[str, list[dict[str, Any]]] = {}
    for item in remote_documents:
        by_title.setdefault(_title_key(item.get("title")), []).append(item)

    for plan in plans:
        if plan.action == "FAIL":
            continue
        document = plan.document
        previous_id = str(plan.previous.get("doc_id") or "").strip()
        title_matches = by_title.get(_title_key(document.title), [])
        distinct_title_ids = {str(item.get("doc_id")) for item in title_matches}
        if len(distinct_title_ids) > 1:
            plan.action = "FAIL"
            plan.reason = (
                f"目标库中标题“{document.title}”对应多个 doc_id，拒绝自动认领或覆盖："
                + ", ".join(sorted(distinct_title_ids))
            )
            continue
        if any(int(item.get("chunks", 0)) <= 0 for item in title_matches):
            plan.action = "FAIL"
            plan.reason = (
                f"目标库中文档“{document.title}”的 chunks<=0，远端文档不完整；"
                "拒绝 SKIP/PUT/自动覆盖，请先恢复或人工核对。"
            )
            continue

        remote_by_id = by_id.get(previous_id) if previous_id else None
        if remote_by_id is not None:
            if int(remote_by_id.get("chunks", 0)) <= 0:
                plan.action = "FAIL"
                plan.reason = (
                    f"目标库 doc_id={previous_id} 的 chunks<=0，远端文档不完整；"
                    "拒绝 SKIP/PUT，请先恢复或人工核对。"
                )
                continue
            plan.remote_verified = True
            same_hash = (
                bool(plan.previous.get("sha256"))
                and plan.previous.get("sha256") == document.sha256
            )
            successful = plan.previous.get("last_status") in SUCCESS_STATUSES
            if successful and same_hash and not force:
                plan.action = "SKIP"
                plan.reason = "sha256-unchanged-and-remote-doc-present"
            else:
                plan.action = "UPDATE"
                plan.reason = "remote-doc-present-content-changed"
            continue

        if previous_id:
            if title_matches:
                plan.action = "FAIL"
                plan.reason = (
                    f"状态中的 doc_id={previous_id} 不在目标知识库“{kb_id}”清单，"
                    f"但目标库已有同名文档 doc_id={title_matches[0]['doc_id']}；"
                    "拒绝凭标题覆盖，请恢复正确状态清单或人工核对。"
                )
            else:
                plan.action = "NEW"
                plan.reason = "state-doc-missing-from-target"
            continue

        if title_matches:
            plan.action = "FAIL"
            if state_exists:
                plan.reason = (
                    f"状态清单没有该路径的 doc_id，但目标库已有同名文档 "
                    f"doc_id={title_matches[0]['doc_id']}；拒绝自动认领，请恢复状态清单或人工核对。"
                )
            else:
                plan.reason = (
                    f"状态清单丢失，目标库已有同名文档 doc_id={title_matches[0]['doc_id']}；"
                    "拒绝自动认领，请恢复状态清单或人工核对。"
                )
        else:
            plan.action = "NEW"
            plan.reason = "not-indexed-and-no-remote-title-conflict"
    return plans


def send_document(
    session: Any,
    plan: PlannedDocument,
    *,
    base_url: str,
    kb_id: str,
    user_id: str | None,
    ocr_mode: str,
    timeout: float,
) -> RequestOutcome:
    document = plan.document
    if plan.action not in {"NEW", "UPDATE"}:
        return RequestOutcome(False, error=f"不允许为动作 {plan.action} 发送入库请求。")
    if plan.action == "UPDATE" and not plan.remote_verified:
        return RequestOutcome(
            False,
            error=(
                f"拒绝 PUT：doc_id={plan.previous.get('doc_id') or '(空)'} 未经目标知识库对账确认；"
                "不会把不存在或其他知识库的 doc_id 交给服务端。"
            ),
        )
    data = {"title": document.title, "kb_id": kb_id, "ocr_mode": ocr_mode}
    if user_id:
        data["user_id"] = user_id
    headers = _request_headers()
    try:
        with document.path.open("rb") as stream:
            files = {"file": (document.path.name, stream, "application/pdf")}
            if plan.action == "NEW":
                response = session.post(
                    f"{base_url.rstrip('/')}/kb/ingest",
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=(10.0, timeout),
                )
            else:
                response = session.put(
                    f"{base_url.rstrip('/')}/kb/docs/{plan.previous.get('doc_id')}",
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=(10.0, timeout),
                )
    except Exception as exc:
        return RequestOutcome(
            False,
            status_code=None,
            error=(
                f"请求失败：{type(exc).__name__}: {exc}。本条会保留为失败，"
                "下次运行会自动重试。"
            ),
        )
    status_code = int(response.status_code)
    if status_code == 409:
        return RequestOutcome(
            False,
            status_code=status_code,
            error=(
                "服务端返回409重复内容，未算成功；请核对状态清单中的 doc_id 和服务端文档，"
                "确认后再续跑或用 --state 指向正确清单。"
            ),
        )
    if not 200 <= status_code < 300:
        return RequestOutcome(
            False,
            status_code=status_code,
            error=f"服务端返回 HTTP {status_code}：{_safe_response_detail(response)}；下次运行会自动重试。",
        )
    try:
        payload = response.json()
    except Exception as exc:
        return RequestOutcome(False, status_code=status_code, error=f"成功响应不是有效 JSON：{exc}")
    if not isinstance(payload, Mapping):
        return RequestOutcome(False, status_code=status_code, error="成功响应不是对象，未保存为成功。")
    doc_id = str(payload.get("doc_id") or plan.previous.get("doc_id") or "")
    if not doc_id:
        return RequestOutcome(False, status_code=status_code, error="响应缺少 doc_id，未保存为成功。")
    return RequestOutcome(True, doc_id=doc_id, status_code=status_code)


def _record_success(
    state: dict[str, Any],
    plan: PlannedDocument,
    *,
    kb_id: str,
    pipeline_fingerprint: str,
    doc_id: str,
    status_code: int | None,
) -> None:
    entry = _base_entry(plan.document, kb_id)
    entry.update(plan.previous)
    entry.update(
        {
            "kb_id": kb_id,
            "path": str(plan.document.path.resolve()),
            "sha256": plan.document.sha256,
            "size": plan.document.size,
            "mtime": plan.document.mtime,
            "title": plan.document.title,
            "doc_id": doc_id,
            "last_status": "success",
            "last_action": plan.action,
            "indexed_at": utc_now(),
            "pipeline_fingerprint": pipeline_fingerprint,
            "last_error": "",
            "last_http_status": status_code,
        }
    )
    state["entries"][state_key(kb_id, plan.document.path)] = entry


def _record_skip(
    state: dict[str, Any], plan: PlannedDocument, *, kb_id: str
) -> None:
    entry = _base_entry(plan.document, kb_id)
    entry.update(plan.previous)
    entry.update(
        {
            "kb_id": kb_id,
            "path": str(plan.document.path.resolve()),
            "sha256": plan.document.sha256,
            "size": plan.document.size,
            "mtime": plan.document.mtime,
            "title": plan.document.title,
            "last_status": "success",
            "last_action": "SKIP",
            "last_error": "",
        }
    )
    state["entries"][state_key(kb_id, plan.document.path)] = entry


def _record_failure(
    state: dict[str, Any],
    plan: PlannedDocument,
    *,
    kb_id: str,
    pipeline_fingerprint: str,
    error: str,
    status_code: int | None,
) -> None:
    entry = _base_entry(plan.document, kb_id)
    entry.update(plan.previous)
    entry.update(
        {
            "kb_id": kb_id,
            "path": str(plan.document.path.resolve()),
            "sha256": plan.document.sha256,
            "size": plan.document.size,
            "mtime": plan.document.mtime,
            "title": plan.document.title,
            "last_status": "failed",
            "last_action": "FAIL",
            "last_error": error,
            "last_http_status": status_code,
            "last_attempted_at": utc_now(),
        }
    )
    if not entry.get("pipeline_fingerprint"):
        entry["pipeline_fingerprint"] = pipeline_fingerprint
    state["entries"][state_key(kb_id, plan.document.path)] = entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 PDF 并按 SHA-256 对企业知识库执行可续跑的增量入库。"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--directory", help="递归扫描 PDF 的目录，相对路径按后端目录解释")
    source.add_argument("--testset", help="从 YAML 的 expected_sources 提取 PDF")
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--state", default=str(DEFAULT_STATE), help="持久状态清单路径")
    parser.add_argument("--force", action="store_true", help="强制处理所有文件；已有 doc_id 使用更新")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不检查服务、不请求、不写状态")
    parser.add_argument("--user-id", help="写入接口使用的用户 ID；也可用 KB_INGEST_USER_ID 环境变量")
    parser.add_argument("--ocr-mode", choices=("local", "baidu", "auto"), default="local")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="每份 PDF HTTP 请求的读取超时秒数，默认1800",
    )
    args = parser.parse_args(argv)
    if args.directory is None and args.testset is None:
        args.testset = str(DEFAULT_TESTSET)
    if args.timeout <= 0:
        parser.error("--timeout 必须大于0")
    return args


def _effective_user_id(args: argparse.Namespace) -> str | None:
    return args.user_id or os.getenv("KB_INGEST_USER_ID") or os.getenv("KB_USER_ID")


def _print_plan(plans: list[PlannedDocument], *, dry_run: bool) -> None:
    prefix = "DRY-RUN " if dry_run else ""
    for plan in plans:
        suffix = f" ({plan.reason})"
        print(f"{prefix}{plan.action} {plan.document.path}{suffix}")


def _set_last_run(
    state: dict[str, Any],
    *,
    status: str,
    planned_total: int,
    completed_total: int,
    stats: dict[str, int],
    started_at: str,
    finished_at: str | None,
    state_write_failures: int,
    error: str = "",
) -> None:
    state["last_run"] = {
        "run_status": status,
        "planned_total": planned_total,
        "completed_total": completed_total,
        "new": stats["NEW"],
        "updated": stats["UPDATE"],
        "skipped": stats["SKIP"],
        "failed": stats["FAIL"],
        "started_at": started_at,
        "finished_at": finished_at,
        "state_write_failures": state_write_failures,
        "last_error": error,
    }


def _save_run_snapshot(
    state_path: Path,
    state: dict[str, Any],
    *,
    status: str,
    planned_total: int,
    completed_total: int,
    stats: dict[str, int],
    started_at: str,
    finished_at: str | None,
    state_write_failures: int,
    error: str = "",
) -> str | None:
    _set_last_run(
        state,
        status=status,
        planned_total=planned_total,
        completed_total=completed_total,
        stats=stats,
        started_at=started_at,
        finished_at=finished_at,
        state_write_failures=state_write_failures,
        error=error,
    )
    try:
        saved = save_state_atomic(state_path, state)
        if saved is False:
            return "save_state_atomic 返回 False"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def run(args: argparse.Namespace, *, session: Any | None = None) -> int:
    client = session or requests.Session()
    owns_session = session is None
    stats = {"NEW": 0, "UPDATE": 0, "SKIP": 0, "FAIL": 0}
    state_save_failures = 0
    state: dict[str, Any] | None = None
    state_path: Path | None = None
    run_started_at = ""
    planned_total = 0
    completed_total = 0
    state_ready = False
    try:
        documents = collect_documents(directory=args.directory, testset=args.testset)
        if not documents:
            raise CorpusError("没有找到 PDF：请检查 --directory 或题集 expected_sources")
        state_path = resolve_backend_path(args.state)
        state_exists = state_path.is_file()
        state = load_state(state_path)
        pipeline_fingerprint = compute_pipeline_fingerprint(ocr_mode=args.ocr_mode)
        if not args.dry_run:
            check_service(client, args.base_url, timeout=args.timeout)
        documents = snapshot_documents(documents)
        plans = plan_documents(
            documents,
            state,
            kb_id=args.kb_id,
            pipeline_fingerprint=pipeline_fingerprint,
            force=args.force,
        )
        planned_total = len(plans)
        changed = [plan.document.path for plan in plans if plan.pipeline_changed]
        if changed:
            if args.force:
                print(
                    f"WARNING 规则已变化，已使用 --force 重建 {len(changed)} 份；"
                    "不使用 --force 时默认不会自动全量重建。"
                )
            else:
                print(
                    f"WARNING 规则已变化，是否用 --force 重建：{len(changed)} 份；"
                    "本次默认跳过，避免擅自全量重建。"
                )
        if args.dry_run:
            _print_plan(plans, dry_run=True)
            for plan in plans:
                stats[plan.action] += 1
            print(
                "SUMMARY "
                + " ".join(f"{key}={stats[key]}" for key in ("NEW", "UPDATE", "SKIP", "FAIL"))
            )
            return 0

        user_id = _effective_user_id(args)
        run_started_at = utc_now()
        initial_error = _save_run_snapshot(
            state_path,
            state,
            status="in_progress",
            planned_total=planned_total,
            completed_total=0,
            stats=stats,
            started_at=run_started_at,
            finished_at=None,
            state_write_failures=state_save_failures,
        )
        if initial_error:
            print(
                f"ERROR 无法保存 in_progress 状态清单 {state_path}：{initial_error}；"
                "为避免状态不明，未执行任何入库请求。",
                file=sys.stderr,
            )
            return 1
        state_ready = True

        remote_documents = fetch_remote_documents(
            client,
            args.base_url,
            kb_id=args.kb_id,
            user_id=user_id,
            timeout=args.timeout,
        )
        plans = reconcile_plans(
            plans,
            state_exists=state_exists,
            remote_documents=remote_documents,
            kb_id=args.kb_id,
            force=args.force,
        )
        _print_plan(plans, dry_run=False)

        for plan in plans:
            if plan.action == "SKIP":
                _record_skip(state, plan, kb_id=args.kb_id)
                stats["SKIP"] += 1
                message = "内容未变化，未调用入库接口"
            elif plan.action == "FAIL":
                _record_failure(
                    state,
                    plan,
                    kb_id=args.kb_id,
                    pipeline_fingerprint=pipeline_fingerprint,
                    error=plan.document.error or plan.reason,
                    status_code=None,
                )
                stats["FAIL"] += 1
                message = plan.document.error or plan.reason
            else:
                outcome = send_document(
                    client,
                    plan,
                    base_url=args.base_url,
                    kb_id=args.kb_id,
                    user_id=user_id,
                    ocr_mode=args.ocr_mode,
                    timeout=args.timeout,
                )
                if outcome.success:
                    _record_success(
                        state,
                        plan,
                        kb_id=args.kb_id,
                        pipeline_fingerprint=pipeline_fingerprint,
                        doc_id=outcome.doc_id,
                        status_code=outcome.status_code,
                    )
                    stats[plan.action] += 1
                    message = f"doc_id={outcome.doc_id} HTTP={outcome.status_code}"
                else:
                    _record_failure(
                        state,
                        plan,
                        kb_id=args.kb_id,
                        pipeline_fingerprint=pipeline_fingerprint,
                        error=outcome.error,
                        status_code=outcome.status_code,
                    )
                    stats["FAIL"] += 1
                    message = outcome.error
            completed_total += 1
            snapshot_error = _save_run_snapshot(
                state_path,
                state,
                status="in_progress",
                planned_total=planned_total,
                completed_total=completed_total,
                stats=stats,
                started_at=run_started_at,
                finished_at=None,
                state_write_failures=state_save_failures,
            )
            if snapshot_error:
                state_save_failures += 1
                failure_reason = (
                    "进度快照写入失败，已停止后续计划；"
                    f"已处理 {completed_total}/{planned_total} 项，{snapshot_error}"
                )
                failure_snapshot_error = _save_run_snapshot(
                    state_path,
                    state,
                    status="failed",
                    planned_total=planned_total,
                    completed_total=completed_total,
                    stats=stats,
                    started_at=run_started_at,
                    finished_at=utc_now(),
                    state_write_failures=state_save_failures,
                    error=failure_reason,
                )
                print(
                    f"FAIL STATE {state_path}：进度快照写入失败，已完成请求不计为业务失败；"
                    f"已停止后续计划；旧清单/上一个有效快照保留；{snapshot_error}"
                )
                print(f"{plan.action if plan.action != 'FAIL' else 'FAIL'} {plan.document.path}：{message}")
                print(
                    "SUMMARY "
                    + " ".join(
                        f"{key}={stats[key]}" for key in ("NEW", "UPDATE", "SKIP", "FAIL")
                    )
                    + f" total={planned_total} completed={completed_total}"
                )
                if failure_snapshot_error:
                    state_save_failures += 1
                    print(
                        f"ERROR 无法落盘 last_run.run_status=failed：{state_path}；"
                        "旧清单/上一个有效快照保留；"
                        f"{failure_snapshot_error}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"ERROR 已停止后续计划，last_run 已标记 failed：{state_path}。",
                        file=sys.stderr,
                    )
                return 1
            print(f"{plan.action if plan.action != 'FAIL' else 'FAIL'} {plan.document.path}：{message}")

        final_status = "completed" if stats["FAIL"] == 0 and state_save_failures == 0 else "failed"
        final_reason = "" if final_status == "completed" else "本次运行存在业务失败或状态清单写盘失败"
        final_error = _save_run_snapshot(
            state_path,
            state,
            status=final_status,
            planned_total=planned_total,
            completed_total=completed_total,
            stats=stats,
            started_at=run_started_at,
            finished_at=utc_now(),
            state_write_failures=state_save_failures,
            error=final_reason,
        )
        print(
            "SUMMARY "
            + " ".join(f"{key}={stats[key]}" for key in ("NEW", "UPDATE", "SKIP", "FAIL"))
            + f" total={len(plans)}"
        )
        if final_error:
            print(
                f"ERROR 最终状态清单写入失败：{state_path}；旧清单/上一个有效快照保留；"
                f"{final_error}",
                file=sys.stderr,
            )
            return 1
        if state_save_failures:
            print(
                "ERROR 状态清单曾写入失败；请检查磁盘权限后重新运行，业务请求不计为失败。",
                file=sys.stderr,
            )
        return 1 if final_status != "completed" else 0
    except CorpusError as exc:
        if state_ready and state is not None and state_path is not None:
            snapshot_error = _save_run_snapshot(
                state_path,
                state,
                status="failed",
                planned_total=planned_total,
                completed_total=completed_total,
                stats=stats,
                started_at=run_started_at,
                finished_at=utc_now(),
                state_write_failures=state_save_failures,
                error=str(exc),
            )
            if snapshot_error:
                print(
                    f"ERROR 失败状态也无法写入 {state_path}：{snapshot_error}；"
                    "旧清单/上一个有效快照保留。",
                    file=sys.stderr,
                )
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if state_ready and state is not None and state_path is not None:
            snapshot_error = _save_run_snapshot(
                state_path,
                state,
                status="interrupted",
                planned_total=planned_total,
                completed_total=completed_total,
                stats=stats,
                started_at=run_started_at,
                finished_at=utc_now(),
                state_write_failures=state_save_failures,
                error="KeyboardInterrupt",
            )
            if snapshot_error:
                print(
                    f"ERROR 中断状态无法写入 {state_path}：{snapshot_error}；"
                    "旧清单/上一个有效快照保留。",
                    file=sys.stderr,
                )
        print("ERROR 任务被中断；已完成案例保留在最近一次有效快照中。", file=sys.stderr)
        return 130
    except Exception as exc:
        if state_ready and state is not None and state_path is not None:
            snapshot_error = _save_run_snapshot(
                state_path,
                state,
                status="failed",
                planned_total=planned_total,
                completed_total=completed_total,
                stats=stats,
                started_at=run_started_at,
                finished_at=utc_now(),
                state_write_failures=state_save_failures,
                error=f"{type(exc).__name__}: {exc}",
            )
            if snapshot_error:
                print(
                    f"ERROR 失败状态也无法写入 {state_path}：{snapshot_error}；"
                    "旧清单/上一个有效快照保留。",
                    file=sys.stderr,
                )
        print(f"ERROR 未预期错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if owns_session:
            client.close()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
