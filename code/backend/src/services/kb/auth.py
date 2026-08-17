"""用户与权限管理（RBAC）——SQLite 存储（P3 §3.4 / v3 §6.1）。

数据模型：
  users(user_id PK, name, role, api_token_hash, token_expires_at, created_at)   role: member | admin
  kb_access(user_id, kb_id)                   -- member 可访问的知识库

权限模型（本次审查修复后）：
  - admin：全通（可访问任意知识库 + 管理用户/角色）
  - member：只能访问 kb_access 里被授权的知识库（读写均可）
  - 越权防护在入口校验（检索层之前拦截），绝不放检索后再补救

鉴权（🟠4 → 生产加固）：
  - 每用户一个 API token（secrets 生成，kb_ 前缀）。**DB 只存 SHA-256 哈希**，
    明文只在创建/重置时下发一次——即使库文件泄露也拿不到可用 token。
  - token 带过期时间（KB_TOKEN_TTL_DAYS，默认 90 天），过期自动失效，
    `get_user_by_token` 内部清理；重置/创建续期。
  - 旧库明文 token 首次打开时自动迁移为哈希（原 token 字符串不变，老客户端不用换）。
  - user_id 直传仍兼容（过渡期），生产可用 KB_REQUIRE_TOKEN=1 强制只认 token。
  - 首个 admin 引导：配置 KB_BOOTSTRAP_ADMIN_TOKEN 环境变量即可（见 bootstrap_admin）。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
import secrets
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _new_token() -> str:
    """生成 API token（32 字节随机数的 hex，64 字符，不可猜测）。"""
    return "kb_" + secrets.token_hex(32)


def _hash_token(token: str) -> str:
    """token 哈希（SHA-256 hexdigest）。只存哈希，明文不落库。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _default_ttl_days() -> int | None:
    """token 有效期（天）。KB_TOKEN_TTL_DAYS 环境变量覆盖；空/非法视为 90 天。"""
    raw = os.getenv("KB_TOKEN_TTL_DAYS", "").strip()
    if not raw:
        return 90
    try:
        days = int(raw)
        return days if days > 0 else 0  # 0 = 立即过期（测试用），负数按 0
    except ValueError:
        return 90


class AuthStore:
    """用户与知识库访问权限（SQLite）。"""

    VALID_ROLES = {"member", "admin"}

    def __init__(self, db_path: str | Path, token_ttl_days: int | None = None):
        self._ttl_days = _default_ttl_days() if token_ttl_days is None else token_ttl_days
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # P3：WAL 模式 + busy_timeout，避免并发写时 "database is locked"
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id         TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'member',
                api_token       TEXT,
                api_token_hash  TEXT,
                token_expires_at TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS kb_access (
                user_id TEXT NOT NULL,
                kb_id   TEXT NOT NULL,
                PRIMARY KEY (user_id, kb_id)
            );
            """
        )
        self._migrate_schema()
        self._migrate_plaintext_tokens()
        self._backfill_missing_tokens()
        self._conn.commit()

    # ---------- 迁移 ----------
    def _migrate_schema(self) -> None:
        """旧库补列：api_token_hash / token_expires_at（幂等）。"""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()}
        if "role" not in cols:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member'"
            )
        if "api_token_hash" not in cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN api_token_hash TEXT")
        if "token_expires_at" not in cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN token_expires_at TEXT")
        # 新库可能没建 api_token 旧列（上面 CREATE 已含）；旧列用不上了
        if "api_token" not in cols and "api_token" not in {
            r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()
        }:
            self._conn.execute("ALTER TABLE users ADD COLUMN api_token TEXT")

    def _migrate_plaintext_tokens(self) -> int:
        """旧库明文 api_token → 哈希迁移（幂等）。明文清空，token 字符串不变。

        迁移后 DB 中不再存在可用明文 token：即使库文件泄露也拿不到凭据。
        返回迁移的用户数。
        """
        rows = self._conn.execute(
            "SELECT user_id, api_token FROM users "
            "WHERE api_token IS NOT NULL AND api_token != ''"
        ).fetchall()
        migrated = 0
        for uid, plain in rows:
            if plain:
                self._conn.execute(
                    "UPDATE users SET api_token_hash=?, token_expires_at=? WHERE user_id=?",
                    (_hash_token(plain), self._expiry_iso(), uid),
                )
                migrated += 1
            self._conn.execute(
                "UPDATE users SET api_token=NULL WHERE user_id=?", (uid,)
            )
        if migrated:
            logger.info("迁移 %d 个用户明文 token 为哈希", migrated)
        return migrated

    def _backfill_missing_tokens(self) -> int:
        """无哈希的用户补发新 token（幂等）。返回补发数。"""
        rows = self._conn.execute(
            "SELECT user_id FROM users WHERE api_token_hash IS NULL OR api_token_hash = ''"
        ).fetchall()
        for (uid,) in rows:
            token = _new_token()
            self._conn.execute(
                "UPDATE users SET api_token_hash=?, token_expires_at=? WHERE user_id=?",
                (_hash_token(token), self._expiry_iso(), uid),
            )
        if rows:
            logger.info("为 %d 个无 token 用户补发 API token", len(rows))
        return len(rows)

    def _expiry_iso(self) -> str | None:
        """按 TTL 计算过期时间；TTL 为 None（永不）返回 None。"""
        if self._ttl_days is None:
            return None
        now = _dt.datetime.now(_dt.timezone.utc)
        exp = now + _dt.timedelta(days=self._ttl_days)
        return exp.isoformat()

    # ---------- 用户 ----------
    def create_user(self, name: str, role: str = "member") -> tuple[str, str]:
        """新建用户，返回 (user_id, api_token)。role: member | admin。

        token 明文只在此时返回一次，DB 只存哈希。返回的 token 需交给人/接口调用方。
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"非法角色: {role}")
        uid = uuid.uuid4().hex
        token = _new_token()
        self._conn.execute(
            "INSERT INTO users(user_id, name, role, api_token_hash, token_expires_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (uid, name, role, _hash_token(token), self._expiry_iso()),
        )
        self._conn.commit()
        logger.info("新建用户 name=%s id=%s role=%s", name, uid, role)
        return uid, token

    def bootstrap_admin(self, name: str, token: str) -> str:
        """从环境变量引导首个 admin（生产级 bootstrap 入口，替代裸 POST /kb/users）。

        - 若该 token 已存在：确保角色为 admin 并续期，返回其 user_id（幂等）。
        - 否则：新建 admin 用户，token 设为该值（存哈希），永不自动过期。
        引导 token 是一次性的初始化凭据，创建后应删除该环境变量。
        """
        h = _hash_token(token)
        row = self._conn.execute(
            "SELECT user_id FROM users WHERE api_token_hash=?", (h,)
        ).fetchone()
        if row:
            uid = row[0]
            self._conn.execute(
                "UPDATE users SET role='admin', name=?, token_expires_at=NULL WHERE user_id=?",
                (name, uid),
            )
            logger.info("引导 admin 已存在，确认 admin 角色: id=%s", uid)
        else:
            uid = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO users(user_id, name, role, api_token_hash, token_expires_at) "
                "VALUES(?, ?, 'admin', ?, NULL)",
                (uid, name, h),
            )
            logger.info("引导首个 admin: id=%s name=%s", uid, name)
        self._conn.commit()
        return uid

    def reset_token(self, user_id: str) -> str | None:
        """重置用户 API token（泄露时用）。用户不存在返回 None。

        旧 token 立即失效（哈希被覆盖）；返回新 token 明文（仅此时可见一次）。
        """
        if not self.get_user(user_id):
            return None
        token = _new_token()
        self._conn.execute(
            "UPDATE users SET api_token_hash=?, token_expires_at=? WHERE user_id=?",
            (_hash_token(token), self._expiry_iso(), user_id),
        )
        self._conn.commit()
        logger.info("重置 API token: user_id=%s", user_id)
        return token

    def get_user_by_token(self, token: str | None) -> dict[str, Any] | None:
        """按 API token 反查用户（鉴权入口）。无效/过期 token 返回 None。

        哈希匹配 + 过期检查：过期 token 视为无效并清理其哈希（被动失效）。
        """
        if not token:
            return None
        row = self._conn.execute(
            "SELECT user_id, token_expires_at FROM users WHERE api_token_hash=?",
            (_hash_token(token),),
        ).fetchone()
        if not row:
            return None
        uid, expires_at = row[0], row[1]
        if expires_at:
            now_iso = _utc_now_iso()
            if now_iso > expires_at:
                # 过期：清掉哈希，被动失效
                self._conn.execute(
                    "UPDATE users SET api_token_hash=NULL WHERE user_id=?", (uid,)
                )
                self._conn.commit()
                logger.info("token 已过期，清除: user_id=%s", uid)
                return None
        return self.get_user(uid)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT user_id, name, role FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "name": row[1],
            "role": row[2],
            "allowed_kbs": self.get_allowed_kbs(user_id),
        }

    def list_users(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT user_id, name, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
        result = []
        for uid, name, role, created in rows:
            result.append(
                {
                    "user_id": uid,
                    "name": name,
                    "role": role,
                    "created_at": created,
                    "allowed_kbs": self.get_allowed_kbs(uid),
                }
            )
        return result

    def is_admin(self, user_id: str) -> bool:
        """是否为管理员（admin 全通，可管理用户/角色）。"""
        row = self._conn.execute(
            "SELECT role FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return row is not None and row[0] == "admin"

    def set_role(self, user_id: str, role: str) -> bool:
        """提升/降级用户角色。用户不存在返回 False。"""
        if role not in self.VALID_ROLES:
            raise ValueError(f"非法角色: {role}")
        cur = self._conn.execute(
            "UPDATE users SET role=? WHERE user_id=?", (role, user_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ---------- 权限 ----------
    def grant_access(self, user_id: str, kb_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO kb_access(user_id, kb_id) VALUES(?, ?)",
            (user_id, kb_id),
        )
        self._conn.commit()

    def revoke_access(self, user_id: str, kb_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM kb_access WHERE user_id=? AND kb_id=?", (user_id, kb_id)
        )
        self._conn.commit()
        return cur.rowcount

    def get_allowed_kbs(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT kb_id FROM kb_access WHERE user_id=?", (user_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def can_access(self, user_id: str, kb_id: str) -> bool:
        """校验用户是否能访问指定知识库。admin 全通。"""
        if self.is_admin(user_id):
            return True
        row = self._conn.execute(
            "SELECT 1 FROM kb_access WHERE user_id=? AND kb_id=? LIMIT 1",
            (user_id, kb_id),
        ).fetchone()
        return row is not None
