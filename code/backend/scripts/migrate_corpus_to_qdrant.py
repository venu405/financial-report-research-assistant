"""可恢复、幂等、隔离的冻结语料 Qdrant staging 入库工具。

本脚本只从冻结源文件重新解析、分块和 embedding，不读取 Chroma，也不调用
backend 的上传接口。默认没有执行模式时只打印帮助并返回非零；``--dry-run``
只做本地 manifest 校验，不连接或写入 Qdrant。

正式全量需要显式 ``--full-authorized``，并且根目录的
``migration-record/2026-08-29-qdrant-full-ingest/full-run-gate.json`` 必须绑定
当前脚本、冻结 manifest、loopback Qdrant 和 T1 快照。脚本不会创建 alias，也
不会修改默认应用配置。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4, uuid5

import httpx

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.kb.embeddings import EmbeddingClient  # noqa: E402
from services.kb.ingest import DocumentChunk, build_chunks  # noqa: E402
from services.kb.qdrant_vector_store import (  # noqa: E402
    CHUNK_POINT_NAMESPACE,
    QdrantVectorStore,
    point_id_for_chunk,
)

SCRIPT_VERSION = "migrate-corpus-to-qdrant-v1"
PAYLOAD_VERSION = 1
VECTOR_SIZE = 1024
VECTOR_DISTANCE = "Cosine"
EMBEDDING_MODEL = "bge-m3"
CHUNK_PROFILE = "structured"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EXPECTED_SOURCE_COUNT = 185
EXPECTED_EXTENSIONS = {".pdf": 175, ".docx": 10}
HASH_BLOCK_SIZE = 1024 * 1024
DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
T1_COLLECTION = "qdrant_preflight_20260828_141124_ca47dc59"
T1_ALIAS = "qdrant_preflight_current_20260828_141124_ca47dc59"
T1_CONTAINER = "kb-qdrant-preflight-20260828-141124-ca47dc59"
FULL_INGEST_ROOT = PROJECT_ROOT / "migration-record" / "2026-08-29-qdrant-full-ingest"
FULL_RUN_GATE = FULL_INGEST_ROOT / "full-run-gate.json"
IMPLEMENTATION_TEST_PATH = BACKEND_DIR / "tests/unit/test_migrate_corpus_to_qdrant_impl.py"
REVIEWER_TEST_PATH = BACKEND_DIR / "tests/unit/test_migrate_corpus_to_qdrant.py"
PROTECTED_COLLECTIONS = {
    T1_COLLECTION,
    "enterprise_kb",
    "enterprise_kb_current",
}
PROTECTED_WORDS = {"latest", "default", "production", "prod", "current"}
DOC_ID_NAMESPACE = UUID("c0ce5d9e-aee8-5e95-9b4d-25c70c83fbb8")


class MigrationError(RuntimeError):
    """输入、门禁、远端或 checkpoint 错误；不会被静默吞掉。"""


class FingerprintMismatch(MigrationError):
    """当前现场与 run manifest/checkpoint 不一致。"""


@dataclass(frozen=True)
class SourceFile:
    manifest_path: str
    relative_path: str
    path: Path
    size: int
    sha256: str
    extension: str

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name


@dataclass(frozen=True)
class ManifestValidation:
    manifest_path: Path
    source_dir: Path
    source_root: str
    manifest_sha256: str
    files: tuple[SourceFile, ...]
    extension_counts: dict[str, int]


@dataclass(frozen=True)
class PipelineIdentity:
    script_sha256: str
    code_sha256: str
    config_sha256: str
    config: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(HASH_BLOCK_SIZE)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """UTF-8 JSON + flush/fsync + same-directory atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MigrationError(f"JSON 读取失败：{path.name}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise MigrationError(f"JSON 根对象格式错误：{path.name}")
    return payload


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _safe_error(exc: BaseException) -> str:
    """只保留脱敏、有限长度的错误摘要，不输出全文、向量或密钥。"""

    detail = re.sub(r"\s+", " ", str(exc or "")).strip()
    detail = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*bearer\s+\S+",
        "authorization=[redacted]",
        detail,
    )
    detail = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*\S+",
        "authorization=[redacted]",
        detail,
    )
    detail = re.sub(
        r"(?i)\bapi[-_ ]?key\s*[:=]\s*\S+",
        "api-key=[redacted]",
        detail,
    )
    detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", detail)
    detail = detail.replace("\x00", "")
    if len(detail) > 320:
        detail = detail[:317] + "..."
    return f"{type(exc).__name__}: {detail or '无详细信息'}"


def _log(run_dir: Path, message: str) -> None:
    log_path = run_dir / "migration.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{utc_now()} {message}\n")


def _normal_posix(value: str) -> str:
    """Normalize only after rejecting absolute and traversal-shaped input."""

    if not isinstance(value, str):
        raise MigrationError("路径必须是字符串")
    raw = value.strip()
    if raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
        raise MigrationError("路径必须是相对路径，拒绝 Unix/Windows 绝对路径")
    raw_parts = raw.replace("\\", "/").split("/")
    if any(part in {".", ".."} for part in raw_parts):
        raise MigrationError("路径含当前目录或穿越段，拒绝继续解析")
    return raw.replace("\\", "/").strip("/")


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _parse_manifest_rows(raw: str) -> tuple[str, list[tuple[str, int, str]]]:
    lines = raw.splitlines()
    header = next((line for line in lines if line.startswith("# source_root=")), "")
    match = re.match(r"# source_root=([^;]+);\s*scan_count=(\d+);", header)
    if not match:
        raise MigrationError("source manifest 缺少可识别的 source_root/scan_count 头")
    source_root = _normal_posix(match.group(1))
    declared_count = int(match.group(2))
    rows: list[tuple[str, int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise MigrationError(f"source manifest 第 {line_number} 行列数错误")
        relative_path, raw_size, digest = parts
        relative_path = _normal_posix(relative_path)
        if not relative_path or PurePosixPath(relative_path).is_absolute():
            raise MigrationError(f"source manifest 第 {line_number} 行路径不是相对路径")
        pure = PurePosixPath(relative_path)
        if any(part in {"", ".", ".."} for part in pure.parts):
            raise MigrationError(f"source manifest 第 {line_number} 行路径含穿越或空段")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise MigrationError(f"source manifest 第 {line_number} 行 size 非整数") from exc
        if size < 0 or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise MigrationError(f"source manifest 第 {line_number} 行 size/hash 格式错误")
        rows.append((relative_path, size, digest.lower()))
    if declared_count != len(rows):
        raise MigrationError(
            f"source manifest 声明 {declared_count} 行，实际 {len(rows)} 行"
        )
    return source_root, rows


def validate_source_manifest(
    source_dir: str | Path,
    source_manifest: str | Path,
    *,
    expected_count: int = EXPECTED_SOURCE_COUNT,
    expected_extensions: Mapping[str, int] = EXPECTED_EXTENSIONS,
) -> ManifestValidation:
    """逐文件验证冻结 manifest；任何 missing/extra/hash/size 都失败。"""

    source_path = _resolve_path(source_dir)
    manifest_path = _resolve_path(source_manifest)
    if not source_path.is_dir():
        raise MigrationError(f"source-dir 不存在或不是目录：{source_path}")
    if not manifest_path.is_file():
        raise MigrationError(f"source-manifest 不存在：{manifest_path}")
    raw_bytes = manifest_path.read_bytes()
    try:
        raw = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MigrationError("source manifest 不是 UTF-8") from exc
    source_root, rows = _parse_manifest_rows(raw)
    if len(rows) != expected_count:
        raise MigrationError(f"source manifest 必须恰好 {expected_count} 行，实际 {len(rows)}")

    declared_root = _resolve_path(source_root)
    if _path_key(declared_root) != _path_key(source_path):
        raise MigrationError(
            f"source-dir 与 manifest source_root 不一致：{source_path} != {declared_root}"
        )

    prefix = source_root.rstrip("/") + "/"
    seen: set[str] = set()
    case_seen: dict[str, str] = {}
    files: list[SourceFile] = []
    manifest_ext = Counter[str]()
    for relative_path, expected_size, expected_hash in rows:
        if relative_path in seen:
            raise MigrationError(f"source manifest 有重复路径：{relative_path}")
        seen.add(relative_path)
        case_key = relative_path.casefold()
        if case_key in case_seen:
            raise MigrationError(
                f"source manifest 有大小写碰撞：{case_seen[case_key]} / {relative_path}"
            )
        case_seen[case_key] = relative_path
        if not relative_path.startswith(prefix):
            raise MigrationError(f"manifest 路径不在 source_root 下：{relative_path}")
        relative_to_source = relative_path[len(prefix) :]
        if not relative_to_source:
            raise MigrationError("manifest 不能把 source_root 本身作为文件")
        candidate = (source_path / Path(*PurePosixPath(relative_to_source).parts)).resolve()
        try:
            candidate.relative_to(source_path)
        except ValueError as exc:
            raise MigrationError(f"manifest 路径解析越界：{relative_path}") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise MigrationError(f"manifest 文件缺失或不是普通文件：{relative_path}")
        actual_size, actual_hash = _hash_file(candidate)
        extension = candidate.suffix.lower()
        manifest_ext[extension] += 1
        if actual_size != expected_size:
            raise MigrationError(
                f"source size mismatch：{relative_path} expected={expected_size} actual={actual_size}"
            )
        if actual_hash != expected_hash:
            raise MigrationError(f"source hash mismatch：{relative_path}")
        files.append(
            SourceFile(
                manifest_path=relative_path,
                relative_path=relative_to_source,
                path=candidate,
                size=actual_size,
                sha256=actual_hash,
                extension=extension,
            )
        )

    actual_files = [path for path in source_path.rglob("*") if path.is_file()]
    actual_rel = {path.relative_to(source_path).as_posix(): path for path in actual_files}
    expected_rel = {item.relative_path for item in files}
    extra = sorted(set(actual_rel) - expected_rel)
    missing = sorted(expected_rel - set(actual_rel))
    if extra or missing:
        raise MigrationError(
            f"source 文件集合不一致：missing={len(missing)} extra={len(extra)}"
        )
    actual_case: dict[str, str] = {}
    for relative in actual_rel:
        key = relative.casefold()
        if key in actual_case and actual_case[key] != relative:
            raise MigrationError(
                f"source 目录有大小写碰撞：{actual_case[key]} / {relative}"
            )
        actual_case[key] = relative
    actual_ext = Counter(path.suffix.lower() for path in actual_files)
    normalized_expected_ext = {key.lower(): int(value) for key, value in expected_extensions.items()}
    if dict(actual_ext) != normalized_expected_ext or dict(manifest_ext) != normalized_expected_ext:
        raise MigrationError(
            f"扩展名数量不匹配：manifest={dict(manifest_ext)} actual={dict(actual_ext)}"
        )
    files.sort(key=lambda item: item.manifest_path.casefold())
    return ManifestValidation(
        manifest_path=manifest_path,
        source_dir=source_path,
        source_root=source_root,
        manifest_sha256=_sha256_bytes(raw_bytes),
        files=tuple(files),
        extension_counts=dict(manifest_ext),
    )


def _load_mapping_file(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    mapping = payload.get("mapping", payload)
    if not isinstance(mapping, Mapping):
        raise MigrationError("kb mapping JSON 必须是对象")
    result: dict[str, str] = {}
    for raw_prefix, raw_kb_id in mapping.items():
        prefix = _normal_posix(str(raw_prefix))
        kb_id = str(raw_kb_id).strip()
        if not prefix or not kb_id:
            raise MigrationError("kb mapping 含空值或路径穿越")
        result[prefix] = kb_id
    return result


def parse_kb_mapping(args: argparse.Namespace) -> dict[str, str]:
    if args.kb_id and (args.kb_map or args.kb_map_file):
        raise MigrationError("--kb-id 与 --kb-map/--kb-map-file 互斥")
    if args.kb_id:
        if not args.kb_id.strip():
            raise MigrationError("--kb-id 不能为空")
        return {"*": args.kb_id.strip()}
    mapping: dict[str, str] = {}
    if args.kb_map_file:
        mapping.update(_load_mapping_file(_resolve_path(args.kb_map_file)))
    for item in args.kb_map or []:
        if "=" not in item:
            raise MigrationError(f"--kb-map 必须是 prefix=kb_id：{item!r}")
        prefix, kb_id = item.split("=", 1)
        prefix = _normal_posix(prefix)
        kb_id = kb_id.strip()
        if not prefix or not kb_id:
            raise MigrationError(f"--kb-map 不能为空：{item!r}")
        if prefix in mapping and mapping[prefix] != kb_id:
            raise MigrationError(f"kb mapping 重复且值冲突：{prefix}")
        mapping[prefix] = kb_id
    if not mapping:
        raise MigrationError("必须显式提供 --kb-id、--kb-map 或 --kb-map-file")
    return dict(sorted(mapping.items()))


def kb_id_for_file(source: SourceFile, mapping: Mapping[str, str]) -> str:
    parts = PurePosixPath(source.relative_path).parts
    first = parts[0] if parts else ""
    candidates = ["/".join(parts[:index]) for index in range(1, len(parts) + 1)]
    candidates.reverse()
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    if first in mapping:
        return mapping[first]
    if "*" in mapping:
        return mapping["*"]
    raise MigrationError(f"没有为冻结源路径配置 KB 映射：{source.relative_path}")


def doc_id_for_source(source: SourceFile) -> str:
    """文档 ID 由冻结源相对路径稳定生成，内容变化在新 run 中隔离。"""

    return f"doc-{uuid5(DOC_ID_NAMESPACE, source.manifest_path).hex}"


def _hash_labeled_files(paths: Sequence[Path]) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    hasher = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise MigrationError(f"fingerprint 文件缺失：{path}")
        _, digest = _hash_file(path)
        label = path.relative_to(PROJECT_ROOT).as_posix()
        hashes[label] = digest
        hasher.update(f"FILE:{label}\0{digest}\0".encode("utf-8"))
    return hasher.hexdigest(), hashes


def _execution_scope(args: argparse.Namespace, override: str | None = None) -> str:
    if override is not None:
        scope = str(override).strip()
        if not scope:
            raise MigrationError("execution scope 不能为空")
        return scope
    if bool(getattr(args, "full_authorized", False)):
        return f"full:{EXPECTED_SOURCE_COUNT}"
    sample_limit = getattr(args, "sample_limit", None)
    if sample_limit is not None:
        return f"sample:{int(sample_limit)}"
    return "unspecified"


def pipeline_identity(
    args: argparse.Namespace,
    manifest: ManifestValidation,
    mapping: Mapping[str, str],
    *,
    execution_scope: str | None = None,
) -> PipelineIdentity:
    script_sha256 = _hash_file(SCRIPT_PATH)[1]
    code_sha256, code_files = _hash_labeled_files(
        [
            BACKEND_DIR / "src/services/kb/ingest.py",
            BACKEND_DIR / "src/services/kb/embeddings.py",
            BACKEND_DIR / "src/services/kb/qdrant_vector_store.py",
            BACKEND_DIR / "src/services/kb/vector_store_contract.py",
        ]
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_root": manifest.source_root,
        "kb_mapping": dict(mapping),
        "qdrant_url": args.qdrant_url.rstrip("/"),
        "collection": str(getattr(args, "collection", "")),
        "batch_size": int(getattr(args, "batch_size", 64)),
        "execution_scope": _execution_scope(args, execution_scope),
        "embedding_url": args.ollama_url.rstrip("/"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": VECTOR_SIZE,
        "vector_distance": VECTOR_DISTANCE,
        "chunk_profile": CHUNK_PROFILE,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "ocr_mode": args.ocr_mode,
        "parent_chunks": False,
        "payload_version": PAYLOAD_VERSION,
        "point_id_namespace": str(CHUNK_POINT_NAMESPACE),
        "code_files": code_files,
    }
    return PipelineIdentity(
        script_sha256=script_sha256,
        code_sha256=code_sha256,
        config_sha256=_sha256_json(config),
        config=config,
    )


def validate_loopback_url(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{field} 不能为空")
    raw = value.strip()
    parsed = urlsplit(raw)
    try:
        port = parsed.port
    except ValueError as exc:
        raise MigrationError(f"{field} 端口格式错误") from exc
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None:
        raise MigrationError(f"{field} 必须显式使用批准的 http://127.0.0.1 端点")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MigrationError(f"{field} 不得含凭证、query 或 fragment")
    if parsed.path not in {"", "/"}:
        raise MigrationError(f"{field} 不得包含资源路径")
    normalized = raw.rstrip("/")
    approved = {
        "qdrant-url": DEFAULT_QDRANT_URL,
        "ollama-url": DEFAULT_OLLAMA_URL,
    }.get(field)
    if approved is not None and normalized != approved:
        raise MigrationError(f"{field} 必须是批准端点：{approved}")
    return normalized


def validate_collection_request(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise MigrationError("--collection 不能为空")
    lowered = name.strip().lower()
    if lowered in {item.lower() for item in PROTECTED_COLLECTIONS}:
        raise MigrationError(f"拒绝受保护 collection：{name}")
    if any(word in lowered for word in PROTECTED_WORDS):
        raise MigrationError(f"拒绝保留/默认 collection 名：{name}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}", name):
        raise MigrationError("collection 名必须是 2-128 位 ASCII 字母、数字、_ 或 -")


def staging_collection_name(requested: str, *, now: datetime | None = None, token: str | None = None) -> str:
    validate_collection_request(requested)
    date = (now or datetime.now()).strftime("%Y%m%d")
    suffix = (token or uuid4().hex[:8]).lower()
    if not re.fullmatch(r"[0-9a-f]{8,32}", suffix):
        raise MigrationError("staging collection 随机后缀格式错误")
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", requested).strip("-_")[:72]
    candidate = f"{base}_{date}_{suffix}"
    validate_collection_request(candidate)
    return candidate


class RunLock:
    """跨进程独占锁；不自动清理疑似 stale lock，避免双写。"""

    def __init__(self, path: Path, *, run_id: str):
        self.path = path
        self.run_id = run_id
        self._stream: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise MigrationError(f"run lock 已存在，拒绝并发运行：{self.path.name}") from exc
        self._stream = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        self._stream.write(json.dumps({"run_id": self.run_id, "pid": os.getpid(), "started_at": utc_now()}, ensure_ascii=False))
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def release(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()


def _docker_t1_identity() -> dict[str, Any]:
    """Read-only inspect of the frozen Qdrant container identity."""

    try:
        result = subprocess.run(
            ["docker", "inspect", T1_CONTAINER],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigrationError(f"Docker inspect 失败：{type(exc).__name__}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无 stderr"
        raise MigrationError(f"Docker inspect container 失败：{detail[:200]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MigrationError("Docker inspect 返回非 JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise MigrationError("Docker inspect container 身份数量不唯一")
    item = payload[0]
    state = item.get("State") or {}
    if not isinstance(state, Mapping) or state.get("Status") != "running":
        raise MigrationError("冻结 Qdrant container 未处于 running")
    health = state.get("Health") or {}
    if isinstance(health, Mapping) and health.get("Status") not in {None, "healthy"}:
        raise MigrationError("冻结 Qdrant container health 不是 healthy")

    config = item.get("Config") or {}
    if not isinstance(config, Mapping):
        raise MigrationError("Docker inspect 缺少 Config")
    image = str(config.get("Image") or "")
    image_digest = str(item.get("Image") or "")
    if not image or not image_digest.startswith("sha256:"):
        raise MigrationError("Docker inspect 缺少 image 或 image digest")

    mounts = item.get("Mounts") or []
    if not isinstance(mounts, list):
        raise MigrationError("Docker inspect Mounts 格式错误")
    storage_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, Mapping)
        and mount.get("Destination") == "/qdrant/storage"
        and mount.get("Type") == "volume"
    ]
    if len(storage_mounts) != 1 or not storage_mounts[0].get("Name"):
        raise MigrationError("Docker inspect 缺少唯一冻结 storage volume")
    volume = str(storage_mounts[0]["Name"])

    port_bindings = (item.get("HostConfig") or {}).get("PortBindings")
    if not isinstance(port_bindings, Mapping):
        port_bindings = (item.get("NetworkSettings") or {}).get("Ports")
    if not isinstance(port_bindings, Mapping):
        raise MigrationError("Docker inspect 缺少端口绑定")
    ports: list[str] = []
    for container_port in ("6333/tcp", "6334/tcp"):
        bindings = port_bindings.get(container_port)
        if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], Mapping):
            raise MigrationError(f"Docker inspect 端口绑定不唯一：{container_port}")
        binding = bindings[0]
        if binding.get("HostIp") != "127.0.0.1" or not binding.get("HostPort"):
            raise MigrationError(f"Docker inspect 非批准 loopback 端口：{container_port}")
        ports.append(f"127.0.0.1:{binding['HostPort']}")

    container_id = str(item.get("Id") or "")
    if not container_id:
        raise MigrationError("Docker inspect 缺少 container id")
    return {
        "container_id": container_id,
        "image": image,
        "image_digest": image_digest,
        "volume": volume,
        "ports": ports,
        "status": str(state.get("Status")),
        "health_status": str(health.get("Status")) if isinstance(health, Mapping) else None,
    }


class QdrantApi:
    """迁移脚本需要的最小 REST 读/删/scroll 边界。"""

    def __init__(self, url: str, *, timeout: float = 30.0, api_key: str | None = None):
        self.url = validate_loopback_url(url, field="qdrant-url")
        self._client = httpx.Client(timeout=timeout, headers={"api-key": api_key} if api_key else {})

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, *, json_body: Any | None = None, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self.url}{path}", json=json_body, params=params)
        except httpx.HTTPError as exc:
            raise MigrationError(f"Qdrant {method} {path} transport error: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise MigrationError(f"Qdrant {method} {path} HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise MigrationError(f"Qdrant {method} {path} 返回非 JSON") from exc
        if not isinstance(payload, dict):
            raise MigrationError(f"Qdrant {method} {path} 返回对象格式错误")
        return payload

    @staticmethod
    def _result(payload: Mapping[str, Any]) -> Any:
        if "result" not in payload:
            raise MigrationError("Qdrant response 缺少 result")
        return payload["result"]

    @staticmethod
    def _collection_path(collection: str, suffix: str = "") -> str:
        return f"/collections/{quote(collection, safe='')}{suffix}"

    def collection_names(self) -> list[str]:
        result = self._result(self._request("GET", "/collections"))
        if not isinstance(result, Mapping) or not isinstance(result.get("collections"), list):
            raise MigrationError("Qdrant collections 返回格式错误")
        names: list[str] = []
        for item in result["collections"]:
            if isinstance(item, Mapping) and item.get("name"):
                names.append(str(item["name"]))
        return sorted(names)

    def aliases(self) -> dict[str, str]:
        result = self._result(self._request("GET", "/aliases"))
        if not isinstance(result, Mapping) or not isinstance(result.get("aliases"), list):
            raise MigrationError("Qdrant aliases 返回格式错误")
        aliases: dict[str, str] = {}
        for item in result["aliases"]:
            if isinstance(item, Mapping) and item.get("alias_name") and item.get("collection_name"):
                aliases[str(item["alias_name"])] = str(item["collection_name"])
        return aliases

    def collection_info(self, collection: str) -> dict[str, Any] | None:
        path = self._collection_path(collection)
        try:
            return self._result(self._request("GET", path))
        except MigrationError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def count(self, collection: str, *, filter_body: Mapping[str, Any] | None = None) -> int:
        body: dict[str, Any] = {"exact": True}
        if filter_body:
            body["filter"] = dict(filter_body)
        result = self._result(self._request("POST", self._collection_path(collection, "/points/count"), json_body=body))
        if not isinstance(result, Mapping) or not isinstance(result.get("count"), int):
            raise MigrationError("Qdrant count 返回格式错误")
        return int(result["count"])

    def scroll_page(self, collection: str, *, offset: Any = None, limit: int = 128, with_vector: bool = False, filter_body: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], Any]:
        body: dict[str, Any] = {"limit": limit, "with_payload": True, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        if filter_body:
            body["filter"] = dict(filter_body)
        result = self._result(self._request("POST", self._collection_path(collection, "/points/scroll"), json_body=body))
        if not isinstance(result, Mapping) or not isinstance(result.get("points"), list):
            raise MigrationError("Qdrant scroll 返回格式错误")
        points = [dict(point) for point in result["points"] if isinstance(point, Mapping)]
        return points, result.get("next_page_offset")

    def scroll_all(self, collection: str, *, page_size: int = 128, with_vector: bool = False, filter_body: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            page, next_offset = self.scroll_page(collection, offset=offset, limit=page_size, with_vector=with_vector, filter_body=filter_body)
            points.extend(page)
            if next_offset is None:
                return points
            if next_offset == offset:
                raise MigrationError("Qdrant scroll offset 未前进")
            offset = next_offset

    def delete_collection(self, collection: str) -> int:
        path = self._collection_path(collection)
        try:
            response = self._client.request("DELETE", f"{self.url}{path}")
        except httpx.HTTPError as exc:
            raise MigrationError(f"Qdrant DELETE transport error: {type(exc).__name__}") from exc
        return int(response.status_code)

    def create_snapshot(self, collection: str) -> dict[str, Any]:
        """请求当前 collection 的 Qdrant snapshot；不做恢复、alias 或默认 wiring。"""

        result = self._result(self._request("POST", self._collection_path(collection, "/snapshots")))
        if not isinstance(result, Mapping) or not isinstance(result.get("name"), str) or not result["name"]:
            raise MigrationError("Qdrant snapshot response 缺少 name")
        return {"name": result["name"]}


def _vector_config(info: Mapping[str, Any]) -> tuple[int, str]:
    try:
        vectors = info["config"]["params"]["vectors"]
    except (KeyError, TypeError) as exc:
        raise MigrationError("Qdrant collection 缺少 unnamed vector config") from exc
    if not isinstance(vectors, Mapping):
        raise MigrationError("Qdrant collection 不得使用 named vector")
    try:
        return int(vectors["size"]), str(vectors["distance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError("Qdrant collection vector config 无效") from exc


def _assert_collection_contract(info: Mapping[str, Any], *, collection: str) -> None:
    size, distance = _vector_config(info)
    if size != VECTOR_SIZE or distance.lower() != VECTOR_DISTANCE.lower():
        raise MigrationError(
            f"collection {collection} vector contract 不符：expected 1024/Cosine got {size}/{distance}"
        )


def _t1_snapshot(api: QdrantApi) -> dict[str, Any]:
    info = api.collection_info(T1_COLLECTION)
    if info is None:
        raise MigrationError("T1 collection 不存在")
    _assert_collection_contract(info, collection=T1_COLLECTION)
    points = api.count(T1_COLLECTION)
    status = str(info.get("status", ""))
    optimizer = str(info.get("optimizer_status", ""))
    if points != 100 or status != "green" or optimizer not in {"ok", "optimizing"}:
        raise MigrationError(f"T1 collection 门槛失败：status={status} optimizer={optimizer} points={points}")
    aliases = api.aliases()
    if aliases.get(T1_ALIAS) != T1_COLLECTION:
        raise MigrationError("T1 collection 缺少冻结 alias 绑定")
    return {
        "alias": T1_ALIAS,
        "collection": T1_COLLECTION,
        "points": points,
        "status": status,
        "optimizer_status": optimizer,
        "vector_size": VECTOR_SIZE,
        "distance": VECTOR_DISTANCE,
        "aliases": aliases,
    }


def _source_records(validation: ManifestValidation, mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in validation.files:
        records.append(
            {
                "manifest_path": source.manifest_path,
                "relative_path": source.relative_path,
                "size": source.size,
                "source_hash": source.sha256,
                "extension": source.extension,
                "kb_id": kb_id_for_file(source, mapping),
                "doc_id": doc_id_for_source(source),
            }
        )
    return records


def _sample_order(files: Sequence[SourceFile]) -> list[SourceFile]:
    pdfs = [item for item in files if item.extension == ".pdf"]
    docxs = [item for item in files if item.extension == ".docx"]
    ordered: list[SourceFile] = []
    if pdfs:
        ordered.append(min(pdfs, key=lambda item: (item.size, item.manifest_path.casefold())))
    if pdfs:
        largest = max(pdfs, key=lambda item: (item.size, item.manifest_path.casefold()))
        if largest not in ordered:
            ordered.append(largest)
    if docxs:
        ordered.append(sorted(docxs, key=lambda item: item.manifest_path.casefold())[0])
    for item in files:
        if item not in ordered:
            ordered.append(item)
    return ordered


def select_files(files: Sequence[SourceFile], sample_limit: int | None) -> list[SourceFile]:
    if sample_limit is None:
        return list(files)
    if sample_limit < 1 or sample_limit > 3:
        raise MigrationError("--sample-limit 只能是 1–3")
    return _sample_order(files)[:sample_limit]


def _new_run_manifest(
    *,
    args: argparse.Namespace,
    validation: ManifestValidation,
    mapping: Mapping[str, str],
    identity: PipelineIdentity,
    selected: Sequence[SourceFile],
    run_id: str,
    collection: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "planned",
        "run_id": run_id,
        "collection_name": collection,
        "collection_base": args.collection,
        "collection_owned": True,
        "execution_scope": identity.config.get("execution_scope", "unspecified"),
        "qdrant_url": args.qdrant_url.rstrip("/"),
        "source_dir": validation.source_root,
        "source_manifest": str(validation.manifest_path.relative_to(PROJECT_ROOT).as_posix()),
        "source_manifest_sha256": validation.manifest_sha256,
        "source_count": len(validation.files),
        "source_extensions": validation.extension_counts,
        "kb_mapping": dict(mapping),
        "script_sha256": identity.script_sha256,
        "code_sha256": identity.code_sha256,
        "config_sha256": identity.config_sha256,
        "config": identity.config,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": VECTOR_SIZE,
        "vector_distance": VECTOR_DISTANCE,
        "chunk_profile": CHUNK_PROFILE,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "point_id_rule": {
            "function": "services.kb.qdrant_vector_store.point_id_for_chunk",
            "namespace": str(CHUNK_POINT_NAMESPACE),
            "input": "kb_id + chunk_id (adapter scoped UUIDv5)",
        },
        "payload_version": PAYLOAD_VERSION,
        "created_at": utc_now(),
        "selected_files": _source_records(
            ManifestValidation(
                validation.manifest_path,
                validation.source_dir,
                validation.source_root,
                validation.manifest_sha256,
                tuple(selected),
                validation.extension_counts,
            ),
            mapping,
        ),
    }


def _checkpoint_template(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = {}
    for item in run_manifest.get("selected_files", []):
        files[item["manifest_path"]] = {
            "status": "pending",
            "manifest_path": item["manifest_path"],
            "relative_path": item["relative_path"],
            "source_hash": item["source_hash"],
            "size": item["size"],
            "config_sha256": run_manifest["config_sha256"],
            "kb_id": item["kb_id"],
            "doc_id": item["doc_id"],
            "point_ids": [],
        }
    return {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "collection_name": run_manifest["collection_name"],
        "status": "planned",
        "started_at": None,
        "updated_at": utc_now(),
        "completed_files": 0,
        "failed_files": 0,
        "processed_files": 0,
        "remaining_files": len(files),
        "total_chunks": 0,
        "active_file": None,
        "files": files,
    }


def _write_checkpoint(run_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    _atomic_write_json(run_dir / "checkpoint.json", checkpoint)


def _write_run_manifest(run_dir: Path, run_manifest: dict[str, Any]) -> None:
    _atomic_write_json(run_dir / "run-manifest.json", run_manifest)


def _check_loaded_run(
    run_manifest: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    validation: ManifestValidation,
    mapping: Mapping[str, str],
    identity: PipelineIdentity,
) -> None:
    expected = {
        "source_manifest_sha256": validation.manifest_sha256,
        "script_sha256": identity.script_sha256,
        "code_sha256": identity.code_sha256,
        "config_sha256": identity.config_sha256,
        "qdrant_url": args.qdrant_url.rstrip("/"),
        "kb_mapping": dict(mapping),
    }
    for key, value in expected.items():
        if run_manifest.get(key) != value:
            raise FingerprintMismatch(f"run manifest {key} 不一致，拒绝混用")
    stored_config = run_manifest.get("config")
    if stored_config is not None:
        if stored_config != identity.config or _sha256_json(stored_config) != identity.config_sha256:
            raise FingerprintMismatch("run manifest config 内容或 hash 不一致，拒绝混用")
    expected_scope = identity.config.get("execution_scope")
    if run_manifest.get("execution_scope") is not None and run_manifest.get("execution_scope") != expected_scope:
        raise FingerprintMismatch("run manifest execution scope 不一致，拒绝混用")
    if run_manifest.get("embedding_model") != EMBEDDING_MODEL or run_manifest.get("embedding_dimension") != VECTOR_SIZE:
        raise FingerprintMismatch("run manifest embedding contract 不一致")
    if run_manifest.get("vector_distance", "").lower() != VECTOR_DISTANCE.lower():
        raise FingerprintMismatch("run manifest distance 不一致")
    if run_manifest.get("collection_base") != args.collection:
        raise FingerprintMismatch("resume 的 --collection base 与 run manifest 不一致")
    validate_collection_request(str(run_manifest.get("collection_name", "")))
    if str(run_manifest.get("collection_name")) in PROTECTED_COLLECTIONS:
        raise FingerprintMismatch("run manifest 指向受保护 collection")
    selected_items = run_manifest.get("selected_files")
    if not isinstance(selected_items, list) or not selected_items:
        raise FingerprintMismatch("run manifest selected_files 缺失或为空")
    current_by_manifest = {item.manifest_path: item for item in validation.files}
    seen_selected: set[str] = set()
    for item in selected_items:
        if not isinstance(item, Mapping):
            raise FingerprintMismatch("run manifest selected file 记录格式错误")
        manifest_path = item.get("manifest_path")
        if not isinstance(manifest_path, str) or manifest_path in seen_selected:
            raise FingerprintMismatch("run manifest selected file manifest path 重复或无效")
        seen_selected.add(manifest_path)
        current = current_by_manifest.get(manifest_path)
        expected_kb = kb_id_for_file(current, mapping) if current is not None else None
        expected_doc = doc_id_for_source(current) if current is not None else None
        if (
            current is None
            or item.get("source_hash") != current.sha256
            or item.get("size") != current.size
            or item.get("kb_id") != expected_kb
            or item.get("doc_id") != expected_doc
            or ("relative_path" in item and item.get("relative_path") != current.relative_path)
            or ("extension" in item and item.get("extension") != current.extension)
        ):
            raise FingerprintMismatch(f"selected source 与 run manifest 不一致：{item.get('manifest_path')}")


def _loaded_execution_scope(run_manifest: Mapping[str, Any]) -> str:
    scope = run_manifest.get("execution_scope")
    if scope is None and isinstance(run_manifest.get("config"), Mapping):
        scope = run_manifest["config"].get("execution_scope")
    if scope is None and run_manifest.get("full_run_gate"):
        return "full:legacy"
    if not isinstance(scope, str) or not scope.strip():
        raise FingerprintMismatch("run manifest 缺少 execution scope")
    return scope.strip()


def _is_full_run_manifest(run_manifest: Mapping[str, Any]) -> bool:
    if run_manifest.get("full_run_gate"):
        return True
    scope = run_manifest.get("execution_scope")
    return isinstance(scope, str) and scope.startswith("full:")


def _gate_bindings(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    bindings = gate.get("bindings", gate)
    if not isinstance(bindings, Mapping):
        raise MigrationError("full-run-gate bindings 格式错误")
    return bindings


def _gate_uses_strict_bindings(gate: Mapping[str, Any], bindings: Mapping[str, Any]) -> bool:
    return "bindings" in gate or any(
        key in bindings
        for key in (
            "code_sha256",
            "config_sha256",
            "test_sha256",
            "implementation_test_sha256",
            "reviewer_test_sha256",
            "container_id",
            "image_digest",
            "volume",
            "ports",
        )
    )


def _first_gate_value(bindings: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in bindings:
            return bindings[name]
    return None


def _gate_value_matches(key: str, actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if key in {"script_sha256", "code_sha256", "implementation_test_sha256", "test_sha256", "config_sha256", "source_manifest_sha256"}:
        return isinstance(actual, str) and actual.strip().lower() == str(expected).lower()
    if key == "qdrant_url":
        try:
            return validate_loopback_url(str(actual), field="qdrant-url") == str(expected)
        except MigrationError:
            return False
    if key in {"embedding_dimension", "vector_distance"}:
        if key == "vector_distance":
            return str(actual).lower() == str(expected).lower()
        return actual == expected
    return actual == expected


def _full_gate_check(args: argparse.Namespace, validation: ManifestValidation, identity: PipelineIdentity) -> dict[str, Any]:
    if not FULL_RUN_GATE.is_file():
        raise MigrationError(f"正式全量需要 Luna-2 gate：缺少 {FULL_RUN_GATE.name}")
    gate = _read_json(FULL_RUN_GATE)
    if gate.get("decision") != "READY":
        raise MigrationError("full-run-gate decision 不是 READY，禁止全量")
    bindings = _gate_bindings(gate)
    strict = _gate_uses_strict_bindings(gate, bindings)
    expected = {
        "script_sha256": identity.script_sha256,
        "source_manifest_sha256": validation.manifest_sha256,
        "qdrant_url": args.qdrant_url.rstrip("/"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": VECTOR_SIZE,
        "vector_distance": VECTOR_DISTANCE,
    }
    aliases = {
        "source_manifest_sha256": ("source_manifest_sha256", "manifest_sha256"),
        "script_sha256": ("script_sha256",),
        "qdrant_url": ("qdrant_url",),
        "embedding_model": ("embedding_model",),
        "embedding_dimension": ("embedding_dimension", "vector_size"),
        "vector_distance": ("vector_distance", "distance"),
    }
    for key, expected_value in expected.items():
        actual = _first_gate_value(bindings, aliases[key])
        if not _gate_value_matches(key, actual, expected_value):
            raise MigrationError(f"full-run-gate {key} 与现场不一致")
    t1 = bindings.get("t1", bindings.get("t1_snapshot", gate.get("t1_snapshot")))
    if not isinstance(t1, Mapping):
        raise MigrationError("full-run-gate 缺少 T1 快照绑定")
    t1_points = t1.get("points", t1.get("count"))
    if (
        t1_points != 100
        or t1.get("collection") != T1_COLLECTION
        or t1.get("vector_size") != VECTOR_SIZE
        or str(t1.get("distance", "")).lower() != VECTOR_DISTANCE.lower()
        or t1.get("status") not in {None, "green"}
    ):
        raise MigrationError("full-run-gate T1 绑定不一致")
    if strict:
        strict_expected = {
            "code_sha256": identity.code_sha256,
            "implementation_test_sha256": _hash_file(IMPLEMENTATION_TEST_PATH)[1],
            "test_sha256": _hash_file(REVIEWER_TEST_PATH)[1],
            "config_sha256": identity.config_sha256,
        }
        strict_aliases = {
            "code_sha256": ("code_sha256",),
            "implementation_test_sha256": ("implementation_test_sha256", "implementation_unit_sha256", "impl_test_sha256"),
            "test_sha256": ("test_sha256", "reviewer_test_sha256", "independent_test_sha256"),
            "config_sha256": ("config_sha256",),
        }
        for key, expected_value in strict_expected.items():
            actual = _first_gate_value(bindings, strict_aliases[key])
            if not _gate_value_matches(key, actual, expected_value):
                raise MigrationError(f"full-run-gate {key} 与现场不一致")
        for key in ("container_id", "image", "image_digest", "volume", "ports"):
            if key not in t1:
                raise MigrationError(f"full-run-gate T1 缺少 Docker {key} 绑定")
        for key in ("status", "optimizer_status", "alias"):
            if key not in t1:
                raise MigrationError(f"full-run-gate T1 缺少 {key} 绑定")
        docker = _docker_t1_identity()
        for key in ("container_id", "image", "image_digest", "volume", "ports"):
            actual = t1.get(key)
            expected_value = docker[key]
            if key == "ports":
                equal = actual == expected_value
            else:
                equal = str(actual).lower() == str(expected_value).lower()
            if not equal:
                raise MigrationError(f"full-run-gate T1 Docker {key} 与现场不一致")
        gate_aliases = bindings.get("qdrant_aliases")
        if not isinstance(gate_aliases, list) or T1_ALIAS not in gate_aliases:
            raise MigrationError("full-run-gate 缺少冻结 T1 alias 绑定")
        if t1.get("alias") != T1_ALIAS:
            raise MigrationError("full-run-gate T1 alias 绑定不一致")
        if t1.get("status") != "green":
            raise MigrationError("full-run-gate T1 status 绑定不一致")
        if t1.get("optimizer_status") not in {"ok", "optimizing"}:
            raise MigrationError("full-run-gate T1 optimizer 绑定不一致")
    approved_collection = bindings.get("collection_name") or bindings.get("staging_collection") or gate.get("staging_collection")
    if not isinstance(approved_collection, str) or not approved_collection:
        raise MigrationError("full-run-gate 缺少 staging collection 绑定")
    if not re.search(r"_[0-9]{8}_[0-9a-f]{8,32}$", approved_collection):
        raise MigrationError("full-run-gate staging collection 必须带日期和随机后缀")
    if args.collection != approved_collection:
        raise MigrationError("--collection 与 gate 绑定的 staging collection 不一致")
    validate_collection_request(approved_collection)
    return dict(gate)


def _assert_gate_t1_current(gate: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """在任何正式写入前，将 gate 的 T1 绑定与现场只读快照再次核对。"""

    bindings = _gate_bindings(gate)
    strict = _gate_uses_strict_bindings(gate, bindings)
    expected = bindings.get("t1", bindings.get("t1_snapshot", gate.get("t1_snapshot")))
    if not isinstance(expected, Mapping):
        raise MigrationError("full-run-gate 缺少 T1 快照绑定")
    for key in ("collection", "points", "vector_size", "distance", "status", "optimizer_status"):
        expected_value = expected.get(key, expected.get("count") if key == "points" else None)
        if expected_value is None:
            continue
        if key == "distance":
            if str(expected_value).lower() != str(current.get(key)).lower():
                raise MigrationError("full-run-gate T1 快照与现场不一致")
        elif expected_value != current.get(key):
            raise MigrationError("full-run-gate T1 快照与现场不一致")
    if strict:
        gate_aliases = bindings.get("qdrant_aliases")
        current_aliases = current.get("aliases")
        if not isinstance(gate_aliases, list) or not isinstance(current_aliases, Mapping):
            raise MigrationError("full-run-gate T1 alias 与现场格式不一致")
        current_t1_aliases = sorted(
            str(alias) for alias, target in current_aliases.items() if target == T1_COLLECTION
        )
        if sorted(str(alias) for alias in gate_aliases) != current_t1_aliases:
            raise MigrationError("full-run-gate T1 alias 与现场不一致")
        docker_expected = {
            key: expected.get(key)
            for key in ("container_id", "image", "image_digest", "volume", "ports")
        }
        if any(value is None for value in docker_expected.values()):
            raise MigrationError("full-run-gate 缺少 Docker identity")
        docker_current = _docker_t1_identity()
        for key, expected_value in docker_expected.items():
            if key == "ports":
                equal = expected_value == docker_current[key]
            else:
                equal = str(expected_value).lower() == str(docker_current[key]).lower()
            if not equal:
                raise MigrationError(f"full-run-gate Docker {key} 与现场不一致")


def _probe_embedding(args: argparse.Namespace) -> None:
    client = EmbeddingClient(base_url=args.ollama_url, model=EMBEDDING_MODEL)
    try:
        if not client.ping():
            raise MigrationError("Ollama /api/tags 不可用")
        vector = client.embed_query("migration fixed dimension probe")
        if len(vector) != VECTOR_SIZE:
            raise MigrationError(f"Ollama bge-m3 维度不是 {VECTOR_SIZE}")
    except MigrationError:
        raise
    except Exception as exc:
        raise MigrationError(f"Ollama embedding 探针失败：{type(exc).__name__}") from exc
    finally:
        raw_client = getattr(client, "_client", None)
        if raw_client is not None:
            raw_client.close()


def _payload_metadata(
    chunk: DocumentChunk,
    *,
    source: SourceFile,
    kb_id: str,
    run_manifest: Mapping[str, Any],
    parse_result: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "source_path": source.manifest_path,
            "source_hash": source.sha256,
            "migration_run_id": run_manifest["run_id"],
            "run_id": run_manifest["run_id"],
            "source_manifest_hash": run_manifest["source_manifest_sha256"],
            "ingest_config_hash": run_manifest["config_sha256"],
            "code_hash": run_manifest["code_sha256"],
            "script_hash": run_manifest["script_sha256"],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": VECTOR_SIZE,
            "chunk_profile": CHUNK_PROFILE,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "payload_version": PAYLOAD_VERSION,
            "ingest_version": SCRIPT_VERSION,
            "parse_status": parse_result["status"],
            "content_hash": source.sha256,
            "kb_id": kb_id,
        }
    )
    return metadata


def _file_record_base(source: SourceFile, *, kb_id: str, doc_id: str, config_sha256: str) -> dict[str, Any]:
    return {
        "manifest_path": source.manifest_path,
        "relative_path": source.relative_path,
        "source_hash": source.sha256,
        "size": source.size,
        "extension": source.extension,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "status": "in_progress",
        "parse": {"status": "not_started"},
        "chunk_count": 0,
        "point_ids": [],
        "point_ids_sha256": None,
        "elapsed_ms": 0,
        "error_category": None,
        "error_summary": None,
        "config_sha256": config_sha256,
    }


def _point_hash(point_ids: Sequence[str]) -> str:
    return _sha256_json(list(point_ids))


def _verify_file_record(
    api: QdrantApi,
    store: QdrantVectorStore,
    *,
    collection: str,
    item: Mapping[str, Any],
    page_size: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    expected_ids = [str(value) for value in item.get("point_ids", [])]
    if not expected_ids or item.get("status") != "completed":
        raise MigrationError(f"completed checkpoint 缺少 point ids：{item.get('manifest_path')}")
    if len(expected_ids) != len(set(expected_ids)):
        raise MigrationError(f"checkpoint point ids 重复：{item.get('manifest_path')}")
    filters = {"must": [{"key": "doc_id", "match": {"value": item["doc_id"]}}, {"key": "kb_id", "match": {"value": item["kb_id"]}}]}
    points = api.scroll_all(collection, page_size=page_size, with_vector=False, filter_body=filters)
    actual_ids = sorted(str(point.get("id")) for point in points)
    expected_sorted = sorted(expected_ids)
    if actual_ids != expected_sorted:
        raise FingerprintMismatch(f"Qdrant points 与 checkpoint 不一致：{item.get('manifest_path')}")
    for point in points:
        payload = point.get("payload") or {}
        if run_id is not None and payload.get("migration_run_id") != run_id:
            raise FingerprintMismatch(f"payload migration_run_id 不属于当前 run：{item.get('manifest_path')}")
        for key, expected in {
            "kb_id": item["kb_id"],
            "doc_id": item["doc_id"],
            "source_hash": item["source_hash"],
            "content_hash": item["source_hash"],
            "payload_version": PAYLOAD_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": VECTOR_SIZE,
        }.items():
            if payload.get(key) != expected:
                raise FingerprintMismatch(f"payload {key} 校验失败：{item.get('manifest_path')}")
    count = api.count(collection, filter_body=filters)
    if count != len(expected_ids):
        raise FingerprintMismatch(f"file count 不一致：{item.get('manifest_path')}")
    logical_ids = sorted(str((point.get("payload") or {}).get("chunk_id", "")) for point in points)
    if any(not value for value in logical_ids):
        raise MigrationError(f"payload 缺少 chunk_id：{item.get('manifest_path')}")
    expected_physical = sorted(point_id_for_chunk(logical_id, str(item["kb_id"])) for logical_id in logical_ids)
    if expected_physical != expected_sorted:
        raise FingerprintMismatch(f"scoped point id 规则校验失败：{item.get('manifest_path')}")
    random_logical = logical_ids[0]
    got = store.get_chunk_by_id(random_logical, kb_id=str(item["kb_id"]))
    if not got or got.get("payload", {}).get("source_hash") != item["source_hash"]:
        raise FingerprintMismatch(f"随机 get/source_hash 失败：{item.get('manifest_path')}")
    snapshot = store.snapshot_doc(str(item["doc_id"]))
    if len(snapshot.get("ids", [])) != len(expected_ids):
        raise FingerprintMismatch(f"snapshot/get 汇总失败：{item.get('manifest_path')}")
    vector = snapshot["embeddings"][0]
    hits = store.search(vector, top_k=1, where={"doc_id": item["doc_id"]}, kb_id=str(item["kb_id"]))
    if not hits or hits[0].get("payload", {}).get("source_hash") != item["source_hash"]:
        raise FingerprintMismatch(f"随机 search/source_hash 失败：{item.get('manifest_path')}")
    return {
        "manifest_path": item["manifest_path"],
        "kb_id": item["kb_id"],
        "doc_id": item["doc_id"],
        "expected_count": len(expected_ids),
        "actual_count": count,
        "point_ids_sha256": _point_hash(expected_ids),
        "random_get": True,
        "random_search": True,
        "source_hash_ok": True,
    }


def verify_run(
    api: QdrantApi,
    store: QdrantVectorStore,
    *,
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    page_size: int,
) -> dict[str, Any]:
    collection = str(run_manifest["collection_name"])
    info = api.collection_info(collection)
    if info is None:
        raise MigrationError(f"staging collection 不存在：{collection}")
    _assert_collection_contract(info, collection=collection)
    all_points = api.scroll_all(collection, page_size=page_size, with_vector=False)
    physical_ids = [str(point.get("id")) for point in all_points]
    if len(physical_ids) != len(set(physical_ids)):
        raise MigrationError("完整 scroll 发现重复物理 point ID")
    logical_pairs = [(str((point.get("payload") or {}).get("kb_id", "")), str((point.get("payload") or {}).get("chunk_id", ""))) for point in all_points]
    if any(not kb or not chunk for kb, chunk in logical_pairs) or len(logical_pairs) != len(set(logical_pairs)):
        raise MigrationError("完整 scroll 发现重复/缺失逻辑 (kb_id, chunk_id)")
    collection_count = api.count(collection)
    if collection_count != len(all_points):
        raise MigrationError(f"collection count 与 scroll 不一致：{collection_count} != {len(all_points)}")
    file_results: list[dict[str, Any]] = []
    expected_by_kb: Counter[str] = Counter()
    run_id = str(run_manifest.get("run_id", ""))
    if not run_id:
        raise FingerprintMismatch("run manifest 缺少 run_id")
    for point in all_points:
        payload = point.get("payload") or {}
        if payload.get("migration_run_id") != run_id:
            raise FingerprintMismatch("payload migration_run_id 不属于当前 run")
    for item in run_manifest.get("selected_files", []):
        entry = checkpoint.get("files", {}).get(item["manifest_path"])
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            raise MigrationError(f"verify 发现未完成文件：{item['manifest_path']}")
        result = _verify_file_record(
            api,
            store,
            collection=collection,
            item=entry,
            page_size=page_size,
            run_id=run_id,
        )
        file_results.append(result)
        expected_by_kb[str(item["kb_id"])] += int(item.get("chunk_count", entry.get("chunk_count", len(entry.get("point_ids", [])))))
    if sum(int(entry.get("actual_count", 0)) for entry in file_results) != collection_count:
        raise MigrationError("逐文件 count 汇总与 collection count 不一致")
    kb_results: dict[str, Any] = {}
    for kb_id, expected in sorted(expected_by_kb.items()):
        actual = api.count(collection, filter_body={"must": [{"key": "kb_id", "match": {"value": kb_id}}]})
        if actual != expected:
            raise MigrationError(f"KB count 不一致：{kb_id} expected={expected} actual={actual}")
        kb_results[kb_id] = {"expected_count": expected, "actual_count": actual}
    for point in all_points:
        payload = point.get("payload") or {}
        required = {
            "kb_id", "doc_id", "chunk_id", "chunk_index", "source_path", "source_hash",
            "migration_run_id", "payload_version", "embedding_model", "embedding_dimension",
            "ingest_config_hash", "source_manifest_hash",
        }
        missing = sorted(key for key in required if key not in payload)
        if missing:
            raise MigrationError(f"payload 缺字段：{','.join(missing)}")
    result = {
        "schema_version": 1,
        "status": "verified",
        "run_id": run_manifest["run_id"],
        "collection": collection,
        "collection_count": collection_count,
        "vector_size": VECTOR_SIZE,
        "distance": VECTOR_DISTANCE,
        "physical_id_count": len(physical_ids),
        "logical_pair_count": len(logical_pairs),
        "pagination": {"complete_scroll": True, "page_size": page_size, "duplicate_physical_ids": 0, "duplicate_logical_pairs": 0},
        "files": file_results,
        "kb_counts": kb_results,
        "payload_source_hash": {"checked_points": len(all_points), "valid": True},
        "verified_at": utc_now(),
    }
    _atomic_write_json(run_dir / "verify-report.json", result)
    return result


def _prepare_remote_target(
    api: QdrantApi,
    *,
    collection: str,
    run_manifest: Mapping[str, Any] | None,
    creating: bool,
) -> dict[str, Any] | None:
    validate_collection_request(collection)
    if collection in PROTECTED_COLLECTIONS or collection == T1_COLLECTION:
        raise MigrationError("目标 collection 是受保护资源")
    names = set(api.collection_names())
    aliases = api.aliases()
    if collection in aliases:
        raise MigrationError(f"目标名称是现有 alias：{collection}")
    if creating and collection in names:
        raise MigrationError("目标 collection 已存在且不属于本 run，拒绝接管")
    if not creating and collection not in names:
        raise MigrationError("resume collection 不存在，拒绝静默重建")
    if collection in names and run_manifest is None:
        raise MigrationError("已有未知 collection，拒绝接管")
    if not creating:
        if not isinstance(run_manifest, Mapping) or not run_manifest.get("run_id"):
            raise FingerprintMismatch("远端 collection ownership 缺少 run_id")
        return _assert_remote_collection_owned(
            api,
            collection=collection,
            run_id=str(run_manifest["run_id"]),
        )
    return None


def _assert_remote_collection_owned(
    api: QdrantApi,
    *,
    collection: str,
    run_id: str,
) -> dict[str, Any]:
    """Scroll the remote collection and reject points owned by another run."""

    points = api.scroll_all(collection, page_size=256, with_vector=False)
    for point in points:
        payload = point.get("payload")
        if not isinstance(payload, Mapping) or payload.get("migration_run_id") != run_id:
            raise FingerprintMismatch(
                f"远端 collection ownership 不属于当前 run：{collection}"
            )
    return {"collection": collection, "run_id": run_id, "points_checked": len(points)}


def _process_file(
    *,
    source: SourceFile,
    kb_id: str,
    run_manifest: Mapping[str, Any],
    checkpoint: dict[str, Any],
    run_dir: Path,
    store: QdrantVectorStore,
    embedder: EmbeddingClient,
    batch_size: int,
    ocr_mode: str,
) -> dict[str, Any]:
    doc_id = doc_id_for_source(source)
    record = _file_record_base(source, kb_id=kb_id, doc_id=doc_id, config_sha256=str(run_manifest["config_sha256"]))
    started = time.perf_counter()
    checkpoint["active_file"] = {
        "manifest_path": source.manifest_path,
        "relative_path": source.relative_path,
        "size": source.size,
        "source_hash": source.sha256,
        "config_sha256": run_manifest["config_sha256"],
        "kb_id": kb_id,
        "doc_id": doc_id,
        "batches_completed": 0,
        "point_ids": [],
    }
    _write_checkpoint(run_dir, checkpoint)
    try:
        chunks = build_chunks(
            source.path,
            doc_id=doc_id,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            kb_id=kb_id,
            ocr_mode=ocr_mode,
            document_title=source.path.stem,
            include_parent_chunks=False,
        )
        if not chunks:
            raise MigrationError("解析成功但没有生成 chunk")
        type_counts = Counter(chunk.chunk_type for chunk in chunks)
        record["parse"] = {"status": "ok", "parser": "services.kb.ingest.build_chunks", "chunk_types": dict(type_counts), "text_chars": sum(len(chunk.text) for chunk in chunks)}
        record["chunk_count"] = len(chunks)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(texts, batch_size=batch_size)
        if len(embeddings) != len(chunks):
            raise MigrationError(f"embedding 数量不匹配：chunks={len(chunks)} vectors={len(embeddings)}")
        if any(len(vector) != VECTOR_SIZE for vector in embeddings):
            raise MigrationError("embedding 维度不匹配")
        physical_ids: list[str] = []
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vector_batch = embeddings[start : start + batch_size]
            logical_ids = store.add_chunks(
                embeddings=vector_batch,
                texts=[chunk.text for chunk in chunk_batch],
                doc_id=doc_id,
                doc_title=source.path.stem,
                source_type=source.extension.lstrip("."),
                chunk_indices=[chunk.chunk_index for chunk in chunk_batch],
                kb_id=kb_id,
                content_hash=source.sha256,
                extra_metadata=[
                    _payload_metadata(chunk, source=source, kb_id=kb_id, run_manifest=run_manifest, parse_result=record["parse"])
                    for chunk in chunk_batch
                ],
            )
            physical_ids.extend(point_id_for_chunk(logical_id, kb_id) for logical_id in logical_ids)
            checkpoint["active_file"]["batches_completed"] += 1
            checkpoint["active_file"]["point_ids"] = physical_ids
            _write_checkpoint(run_dir, checkpoint)
        if len(physical_ids) != len(chunks) or len(set(physical_ids)) != len(physical_ids):
            raise MigrationError("本文件 point ID 数量或唯一性错误")
        record.update({
            "status": "completed",
            "point_ids": physical_ids,
            "point_ids_sha256": _point_hash(physical_ids),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        active = checkpoint.get("active_file") or {}
        record.update({
            "status": "failed",
            "point_ids": list(active.get("point_ids", [])),
            "point_ids_sha256": _point_hash(active.get("point_ids", [])) if active.get("point_ids") else None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error_category": type(exc).__name__,
            "error_summary": _safe_error(exc),
        })
    return record


def _assert_t1_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    keys = ("alias", "aliases", "collection", "points", "status", "optimizer_status", "vector_size", "distance")
    if any(before.get(key) != after.get(key) for key in keys):
        raise MigrationError("T1 前后快照不一致，停止并保留 checkpoint")


def execute_run(args: argparse.Namespace, *, resume: bool) -> int:
    validation = validate_source_manifest(args.source_dir, args.source_manifest)
    mapping = parse_kb_mapping(args)
    if not resume and args.sample_limit is None and not args.full_authorized:
        raise MigrationError("当前阶段执行必须显式 --sample-limit 1–3；全量需 --full-authorized + READY gate")
    if args.full_authorized and args.sample_limit is not None:
        raise MigrationError("正式全量不得携带 --sample-limit")
    if args.sample_limit is not None:
        select_files(validation.files, args.sample_limit)
    run_dir = _resolve_path(args.run_dir)
    run_manifest_path = run_dir / "run-manifest.json"
    checkpoint_path = run_dir / "checkpoint.json"
    existing_manifest: dict[str, Any] | None = None
    if resume:
        if not run_manifest_path.is_file() or not checkpoint_path.is_file():
            raise MigrationError("--resume 需要同一 run-dir 下已有 run-manifest.json 和 checkpoint.json")
        existing_manifest = _read_json(run_manifest_path)
        scope = _loaded_execution_scope(existing_manifest)
    else:
        scope = None
    identity = pipeline_identity(args, validation, mapping, execution_scope=scope)
    lock_run_id = str(existing_manifest.get("run_id")) if existing_manifest is not None else f"new-{uuid4().hex}"
    lock = RunLock(run_dir / "run.lock", run_id=lock_run_id)
    with lock:
        gate: dict[str, Any] | None = None
        if resume:
            run_manifest = _read_json(run_manifest_path)
            checkpoint = _read_json(checkpoint_path)
            if _loaded_execution_scope(run_manifest) != identity.config.get("execution_scope"):
                raise FingerprintMismatch("resume execution scope 在加锁后发生变化")
            _check_loaded_run(run_manifest, args=args, validation=validation, mapping=mapping, identity=identity)
            if checkpoint.get("run_id") != run_manifest.get("run_id") or checkpoint.get("collection_name") != run_manifest.get("collection_name"):
                raise FingerprintMismatch("checkpoint 与 run manifest 绑定不一致")
            checkpoint_files = checkpoint.get("files")
            if not isinstance(checkpoint_files, Mapping):
                raise FingerprintMismatch("checkpoint files 格式错误")
            selected_paths = {str(item["manifest_path"]) for item in run_manifest["selected_files"]}
            selected = [item for item in validation.files if item.manifest_path in selected_paths]
            if len(selected) != len(selected_paths):
                raise FingerprintMismatch("resume selected files 与当前 manifest 不一致")
            for selected_item in run_manifest["selected_files"]:
                manifest_path = str(selected_item["manifest_path"])
                entry = checkpoint_files.get(manifest_path)
                if not isinstance(entry, Mapping):
                    raise FingerprintMismatch(f"checkpoint 缺少文件：{manifest_path}")
                for key in ("manifest_path", "source_hash", "size", "kb_id", "doc_id", "config_sha256"):
                    if entry.get(key) != selected_item.get(key, identity.config_sha256 if key == "config_sha256" else None):
                        raise FingerprintMismatch(f"checkpoint {key} 与 selected file 不一致：{manifest_path}")
            if _is_full_run_manifest(run_manifest):
                gate = _full_gate_check(args, validation, identity)
        else:
            if run_manifest_path.exists() or checkpoint_path.exists():
                raise MigrationError("新 run-dir 已有 run-manifest/checkpoint，拒绝覆盖")
            selected = select_files(validation.files, args.sample_limit)
            if args.sample_limit is None:
                if not args.full_authorized:
                    raise MigrationError("全量没有显式授权")
                gate = _full_gate_check(args, validation, identity)
            run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            gate_bindings = _gate_bindings(gate) if gate is not None else {}
            approved_collection = _first_gate_value(gate_bindings, ("collection_name", "staging_collection")) if gate is not None else None
            collection = str(approved_collection) if approved_collection else staging_collection_name(args.collection)
            run_manifest = _new_run_manifest(args=args, validation=validation, mapping=mapping, identity=identity, selected=selected, run_id=run_id, collection=collection)
            checkpoint = _checkpoint_template(run_manifest)
            _write_run_manifest(run_dir, run_manifest)
            _write_checkpoint(run_dir, checkpoint)
        collection = str(run_manifest["collection_name"])
        validate_collection_request(collection)
        api = QdrantApi(args.qdrant_url, api_key=os.getenv("KB_QDRANT_API_KEY"))
        store: QdrantVectorStore | None = None
        embedder: EmbeddingClient | None = None
        try:
            t1_before = _t1_snapshot(api)
            if gate:
                latest_gate = _full_gate_check(args, validation, identity)
                latest_bindings = _gate_bindings(latest_gate)
                latest_collection = _first_gate_value(latest_bindings, ("collection_name", "staging_collection"))
                if latest_collection != collection:
                    raise FingerprintMismatch("full-run-gate staging collection 在写入前发生变化")
                gate = latest_gate
                _assert_gate_t1_current(gate, t1_before)
            _prepare_remote_target(api, collection=collection, run_manifest=None if not resume else run_manifest, creating=not resume)
            if not resume:
                _probe_embedding(args)
            else:
                # resume 也要确认当前 embedding 端点和模型没漂移，且探针发生在任何新写入前。
                _probe_embedding(args)
            store = QdrantVectorStore(
                url=args.qdrant_url,
                collection_name=collection,
                vector_size=VECTOR_SIZE,
                api_key=os.getenv("KB_QDRANT_API_KEY"),
                timeout=args.timeout,
                create_if_missing=not resume,
                scroll_page_size=max(args.batch_size, 64),
            )
            info = api.collection_info(collection)
            if info is None:
                raise MigrationError("collection 创建后不可读")
            _assert_collection_contract(info, collection=collection)
            embedder = EmbeddingClient(base_url=args.ollama_url, model=EMBEDDING_MODEL)
            checkpoint["status"] = "in_progress"
            checkpoint["started_at"] = checkpoint.get("started_at") or utc_now()
            run_manifest["status"] = "in_progress"
            if gate:
                run_manifest["full_run_gate"] = {"path": str(FULL_RUN_GATE.relative_to(PROJECT_ROOT).as_posix()), "decision": "READY"}
            _write_run_manifest(run_dir, run_manifest)
            _write_checkpoint(run_dir, checkpoint)
            source_by_manifest = {item.manifest_path: item for item in validation.files}
            for index, selected_item in enumerate(run_manifest.get("selected_files", []), start=1):
                manifest_path = str(selected_item["manifest_path"])
                source = source_by_manifest[manifest_path]
                entry = checkpoint["files"].get(manifest_path)
                if not isinstance(entry, dict):
                    raise FingerprintMismatch(f"checkpoint 缺少文件：{manifest_path}")
                if (
                    entry.get("manifest_path") != manifest_path
                    or entry.get("source_hash") != source.sha256
                    or entry.get("size") != source.size
                    or entry.get("kb_id") != selected_item.get("kb_id")
                    or entry.get("doc_id") != selected_item.get("doc_id")
                    or entry.get("config_sha256") != identity.config_sha256
                ):
                    raise FingerprintMismatch(f"文件 checkpoint fingerprint 不一致：{manifest_path}")
                if entry.get("status") == "completed":
                    verify = _verify_file_record(api, store, collection=collection, item=entry, page_size=max(args.batch_size, 64), run_id=str(run_manifest["run_id"]))
                    checkpoint["files"][manifest_path]["last_skip_verify"] = verify
                    checkpoint["files"][manifest_path]["last_action"] = "SKIP"
                    _write_checkpoint(run_dir, checkpoint)
                    print(f"SKIP {index}/{len(run_manifest['selected_files'])} {source.relative_path}")
                    continue
                kb_id = str(selected_item["kb_id"])
                print(f"START {index}/{len(run_manifest['selected_files'])} {source.relative_path} sha256={source.sha256[:12]}")
                _log(run_dir, f"START {index}/{len(run_manifest['selected_files'])} {source.relative_path} sha256_prefix={source.sha256[:12]}")
                record = _process_file(source=source, kb_id=kb_id, run_manifest=run_manifest, checkpoint=checkpoint, run_dir=run_dir, store=store, embedder=embedder, batch_size=args.batch_size, ocr_mode=args.ocr_mode)
                checkpoint["files"][manifest_path] = record
                checkpoint["active_file"] = None
                checkpoint["processed_files"] = sum(1 for value in checkpoint["files"].values() if value.get("status") in {"completed", "failed"})
                checkpoint["completed_files"] = sum(1 for value in checkpoint["files"].values() if value.get("status") == "completed")
                checkpoint["failed_files"] = sum(1 for value in checkpoint["files"].values() if value.get("status") == "failed")
                checkpoint["remaining_files"] = len(run_manifest["selected_files"]) - checkpoint["processed_files"]
                checkpoint["total_chunks"] = sum(int(value.get("chunk_count", 0)) for value in checkpoint["files"].values() if value.get("status") == "completed")
                _write_checkpoint(run_dir, checkpoint)
                if record["status"] != "completed":
                    print(f"FAIL {index}/{len(run_manifest['selected_files'])} {source.relative_path} {record['error_category']}", file=sys.stderr)
                    run_manifest["status"] = "failed"
                    _write_run_manifest(run_dir, run_manifest)
                    raise MigrationError(f"文件处理失败，停止新增文件：{source.relative_path}")
                print(f"DONE {index}/{len(run_manifest['selected_files'])} chunks={record['chunk_count']} elapsed_ms={record['elapsed_ms']}")
                _log(run_dir, f"DONE {index}/{len(run_manifest['selected_files'])} chunks={record['chunk_count']} elapsed_ms={record['elapsed_ms']}")
            verification = verify_run(api, store, run_dir=run_dir, run_manifest=run_manifest, checkpoint=checkpoint, page_size=max(args.batch_size, 64))
            t1_after = _t1_snapshot(api)
            _assert_t1_unchanged(t1_before, t1_after)
            checkpoint["last_verification"] = verification
            checkpoint["t1_after_verification"] = t1_after
            _write_checkpoint(run_dir, checkpoint)
            if args.create_snapshot:
                snapshot = api.create_snapshot(collection)
                checkpoint["snapshot"] = snapshot
                run_manifest["snapshot"] = snapshot
                _write_checkpoint(run_dir, checkpoint)
                _write_run_manifest(run_dir, run_manifest)
            checkpoint["status"] = "completed"
            checkpoint["remaining_files"] = 0
            checkpoint["verified_at"] = verification["verified_at"]
            run_manifest["status"] = "completed"
            run_manifest["t1_before"] = t1_before
            run_manifest["t1_after"] = t1_after
            run_manifest["collection_count"] = verification["collection_count"]
            _write_checkpoint(run_dir, checkpoint)
            _write_run_manifest(run_dir, run_manifest)
            print(f"SUMMARY status=completed files={checkpoint['completed_files']}/{len(run_manifest['selected_files'])} points={verification['collection_count']} collection={collection}")
            return 0
        except KeyboardInterrupt:
            checkpoint["status"] = "interrupted"
            checkpoint["interrupted_at"] = utc_now()
            _write_checkpoint(run_dir, checkpoint)
            run_manifest["status"] = "interrupted"
            _write_run_manifest(run_dir, run_manifest)
            print("INTERRUPTED checkpoint 已保存，可用同 run-dir --resume 继续", file=sys.stderr)
            return 130
        finally:
            if embedder is not None:
                raw_client = getattr(embedder, "_client", None)
                if raw_client is not None:
                    raw_client.close()
            if store is not None:
                store.close()
            api.close()


def dry_run(args: argparse.Namespace) -> int:
    validation = validate_source_manifest(args.source_dir, args.source_manifest)
    mapping = parse_kb_mapping(args)
    identity = pipeline_identity(args, validation, mapping)
    selected = select_files(validation.files, args.sample_limit)
    run_dir = _resolve_path(args.run_dir)
    if (run_dir / "run-manifest.json").exists() and not args.resume:
        raise MigrationError("dry-run run-dir 已存在，换用新的 evidence 目录")
    run_id = f"dry-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_manifest = _new_run_manifest(args=args, validation=validation, mapping=mapping, identity=identity, selected=selected, run_id=run_id, collection="not-created-dry-run")
    run_manifest.update({"status": "dry_run", "dry_run": True, "selected_count": len(selected), "qdrant_writes": 0, "ollama_calls": 0})
    result = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "dry-run",
        "run_id": run_id,
        "source_manifest_sha256": validation.manifest_sha256,
        "entries": len(validation.files),
        "actual_files": len(validation.files),
        "extensions": validation.extension_counts,
        "missing": 0,
        "extra": 0,
        "duplicate": 0,
        "case_collision": 0,
        "hash_mismatch": 0,
        "size_mismatch": 0,
        "selected_count": len(selected),
        "qdrant_writes": 0,
        "qdrant_collection_created": False,
        "script_sha256": identity.script_sha256,
        "code_sha256": identity.code_sha256,
        "config_sha256": identity.config_sha256,
        "completed_at": utc_now(),
    }
    with RunLock(run_dir / "run.lock", run_id=run_id):
        if (run_dir / "run-manifest.json").exists() or (run_dir / "dry-run.json").exists():
            raise MigrationError("dry-run run-dir 在加锁后已存在产物，拒绝覆盖")
        _write_run_manifest(run_dir, run_manifest)
        _atomic_write_json(run_dir / "dry-run.json", result)
        _log(run_dir, "DRY-RUN PASS entries=185 qdrant_writes=0")
    print(f"DRY-RUN PASS files={len(validation.files)} pdf={validation.extension_counts.get('.pdf', 0)} docx={validation.extension_counts.get('.docx', 0)} qdrant_writes=0")
    return 0


def verify_only(args: argparse.Namespace) -> int:
    validation = validate_source_manifest(args.source_dir, args.source_manifest)
    mapping = parse_kb_mapping(args)
    run_dir = _resolve_path(args.run_dir)
    run_manifest = _read_json(run_dir / "run-manifest.json")
    identity = pipeline_identity(
        args,
        validation,
        mapping,
        execution_scope=_loaded_execution_scope(run_manifest),
    )
    checkpoint = _read_json(run_dir / "checkpoint.json")
    _check_loaded_run(run_manifest, args=args, validation=validation, mapping=mapping, identity=identity)
    if checkpoint.get("status") not in {"completed", "verified"}:
        raise MigrationError("verify-only 要求 run checkpoint 已完成")
    lock = RunLock(run_dir / "run.lock", run_id=str(run_manifest["run_id"]))
    with lock:
        run_manifest = _read_json(run_dir / "run-manifest.json")
        checkpoint = _read_json(run_dir / "checkpoint.json")
        _check_loaded_run(run_manifest, args=args, validation=validation, mapping=mapping, identity=identity)
        if checkpoint.get("status") not in {"completed", "verified"}:
            raise MigrationError("verify-only 要求 run checkpoint 已完成")
        api = QdrantApi(args.qdrant_url, api_key=os.getenv("KB_QDRANT_API_KEY"))
        store: QdrantVectorStore | None = None
        try:
            t1_before = _t1_snapshot(api)
            _prepare_remote_target(api, collection=str(run_manifest["collection_name"]), run_manifest=run_manifest, creating=False)
            store = QdrantVectorStore(url=args.qdrant_url, collection_name=str(run_manifest["collection_name"]), vector_size=VECTOR_SIZE, api_key=os.getenv("KB_QDRANT_API_KEY"), timeout=args.timeout, create_if_missing=False, scroll_page_size=max(args.batch_size, 64))
            result = verify_run(api, store, run_dir=run_dir, run_manifest=run_manifest, checkpoint=checkpoint, page_size=max(args.batch_size, 64))
            t1_after = _t1_snapshot(api)
            _assert_t1_unchanged(t1_before, t1_after)
            print(f"VERIFY PASS files={len(result['files'])} points={result['collection_count']} collection={run_manifest['collection_name']}")
            return 0
        finally:
            if store is not None:
                store.close()
            api.close()


def cleanup_sample(args: argparse.Namespace) -> int:
    run_dir = _resolve_path(args.run_dir)
    initial_manifest = _read_json(run_dir / "run-manifest.json")
    run_id = str(initial_manifest.get("run_id", ""))
    if not run_id:
        raise FingerprintMismatch("cleanup run manifest 缺少 run_id")
    lock = RunLock(run_dir / "run.lock", run_id=run_id)
    with lock:
        run_manifest = _read_json(run_dir / "run-manifest.json")
        if str(run_manifest.get("run_id", "")) != run_id:
            raise FingerprintMismatch("cleanup run_id 在加锁后发生变化")
        if not run_manifest.get("collection_owned") or len(run_manifest.get("selected_files", [])) > 3:
            raise MigrationError("cleanup 只允许删除本脚本拥有的最多 3 文件 sample collection")
        if not str(run_manifest.get("execution_scope", "")).startswith("sample:"):
            raise MigrationError("cleanup 只允许 sample execution scope")
        collection = str(run_manifest.get("collection_name", ""))
        validate_collection_request(collection)
        if collection in PROTECTED_COLLECTIONS or not collection.startswith(f"{args.collection}_"):
            raise MigrationError("cleanup collection 不是当前 run 的隔离 staging 名")
        api = QdrantApi(args.qdrant_url, api_key=os.getenv("KB_QDRANT_API_KEY"))
        try:
            before = _t1_snapshot(api)
            ownership = _prepare_remote_target(api, collection=collection, run_manifest=run_manifest, creating=False)
            delete_status = api.delete_collection(collection)
            if delete_status not in {200, 202}:
                raise MigrationError(f"精确 DELETE 未成功：HTTP {delete_status}")
            after_info = api.collection_info(collection)
            if after_info is not None:
                raise MigrationError("DELETE 后 collection 仍可见，未确认 404")
            after = _t1_snapshot(api)
            _assert_t1_unchanged(before, after)
            result = {"schema_version": 1, "status": "cleaned", "collection": collection, "delete_status": delete_status, "get_after_delete": 404, "remote_ownership": ownership, "t1_before": before, "t1_after": after, "completed_at": utc_now()}
            _atomic_write_json(run_dir / "cleanup-report.json", result)
            run_manifest["status"] = "cleaned"
            _write_run_manifest(run_dir, run_manifest)
            print(f"CLEANUP PASS collection={collection} delete_status={delete_status} get_after_delete=404")
            return 0
        finally:
            api.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从冻结 185 文件源重建到隔离 Qdrant staging collection。")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", required=True, help="staging 名称基底；脚本会追加日期/随机后缀")
    parser.add_argument("--kb-id")
    parser.add_argument("--kb-map", action="append", help="相对 source 根的目录前缀到 KB：prefix=kb_id，可重复")
    parser.add_argument("--kb-map-file", help="JSON 对象或 {mapping:{prefix:kb_id}}")
    parser.add_argument("--run-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--cleanup", action="store_true", help="仅删除同 run 的最多 3 文件 sample collection")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-authorized", action="store_true", help="仅在 Luna-2 full-run-gate=READY 后允许 185 全量")
    parser.add_argument("--create-snapshot", action="store_true", help="保留接口位；需 Phase 6 OpenAPI 确认后使用")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ocr-mode", choices=("local", "baidu", "auto"), default="local")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 256:
        parser.error("--batch-size 必须在 1–256")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.sample_limit is not None and not 1 <= args.sample_limit <= 3:
        parser.error("--sample-limit 只能是 1–3")
    if args.full_authorized and not args.execute:
        parser.error("--full-authorized 只能与 --execute 一起使用")
    if args.create_snapshot and not args.execute and not args.resume:
        parser.error("--create-snapshot 只能与 execute/resume 一起使用")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        modes = [args.dry_run, args.execute, args.resume, args.verify_only, args.cleanup]
        if not any(modes):
            print("必须显式指定 --dry-run、--execute、--resume、--verify-only 或 --cleanup。", file=sys.stderr)
            return 2
        validate_loopback_url(args.qdrant_url, field="qdrant-url")
        validate_loopback_url(args.ollama_url, field="ollama-url")
        validate_collection_request(args.collection)
        if args.dry_run:
            if args.sample_limit is not None:
                select_files([], args.sample_limit)
            return dry_run(args)
        if args.verify_only:
            if args.sample_limit is not None or args.full_authorized:
                raise MigrationError("verify-only 不接受 sample/full 授权参数")
            return verify_only(args)
        if args.cleanup:
            if args.sample_limit is not None or args.full_authorized:
                raise MigrationError("cleanup 不接受 sample/full 授权参数")
            return cleanup_sample(args)
        if args.resume:
            if args.full_authorized or args.sample_limit is not None:
                raise MigrationError("resume 使用 run manifest 的固定文件集合，不接受 sample/full 参数")
            return execute_run(args, resume=True)
        return execute_run(args, resume=False)
    except KeyboardInterrupt:
        print("INTERRUPTED 尚未开始新的写入。", file=sys.stderr)
        return 130
    except MigrationError as exc:
        print(f"BLOCKED {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - last-resort non-zero boundary
        print(f"BLOCKED {type(exc).__name__}: {_safe_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
