"""RBAC 权限测试：用户/授权/越权拒绝 + token 哈希/过期/迁移（auth.py 单元测试）。"""
from __future__ import annotations

from services.kb.auth import AuthStore


def test_shared_store_serializes_concurrent_reads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    auth = AuthStore(str(tmp_path / "u.db"))
    uid, token = auth.create_user("concurrent")
    auth.grant_access(uid, "default")

    def read_user():
        user = auth.get_user_by_token(token)
        assert user and user["user_id"] == uid
        assert auth.can_access(uid, "default") is True

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: read_user(), range(64)))


def test_create_user_and_grant(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, _ = auth.create_user("alice")
    assert auth.get_user(uid)["name"] == "alice"
    assert auth.get_allowed_kbs(uid) == []  # 初始无权限

    auth.grant_access(uid, "default")
    auth.grant_access(uid, "product")
    assert auth.get_allowed_kbs(uid) == ["default", "product"]
    assert auth.can_access(uid, "default") is True
    assert auth.can_access(uid, "finance") is False  # 未授权 → 拒绝


def test_revoke_access(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, _ = auth.create_user("bob")
    auth.grant_access(uid, "default")
    assert auth.can_access(uid, "default") is True

    removed = auth.revoke_access(uid, "default")
    assert removed == 1
    assert auth.can_access(uid, "default") is False  # 撤销后拒绝


def test_list_users(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, _ = auth.create_user("carol")
    auth.grant_access(uid, "default")
    users = auth.list_users()
    assert len(users) == 1
    assert users[0]["user_id"] == uid
    assert users[0]["allowed_kbs"] == ["default"]


def test_get_user_not_exist(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    assert auth.get_user("nonexistent") is None


def test_grant_idempotent(tmp_path):
    """重复授权同一 kb 不报错（INSERT OR IGNORE）。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, _ = auth.create_user("dan")
    auth.grant_access(uid, "default")
    auth.grant_access(uid, "default")  # 重复
    assert auth.get_allowed_kbs(uid) == ["default"]  # 仍只一条


def test_create_admin_and_is_admin(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    admin, _ = auth.create_user("root", role="admin")
    member, _ = auth.create_user("alice")  # 默认 member
    assert auth.is_admin(admin) is True
    assert auth.is_admin(member) is False
    assert auth.get_user(admin)["role"] == "admin"


def test_admin_can_access_all(tmp_path):
    """admin 全通：无需授权即可访问任意知识库。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    admin, _ = auth.create_user("root", role="admin")
    member, _ = auth.create_user("bob")
    auth.grant_access(member, "default")

    assert auth.can_access(admin, "任意库") is True  # admin 不需要授权
    assert auth.can_access(member, "default") is True
    assert auth.can_access(member, "其他库") is False  # member 未授权则拒


def test_set_role(tmp_path):
    """提升 member 为 admin / 降级。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, _ = auth.create_user("carol")  # member
    assert auth.is_admin(uid) is False

    assert auth.set_role(uid, "admin") is True
    assert auth.is_admin(uid) is True

    assert auth.set_role(uid, "member") is True
    assert auth.is_admin(uid) is False


def test_invalid_role_rejected(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    try:
        auth.create_user("eve", role="superuser")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    uid, _ = auth.create_user("frank")
    try:
        auth.set_role(uid, "root")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


# ==================== token 鉴权（哈希 + 过期）====================

def test_create_user_gets_token(tmp_path):
    """新建用户自动获得 API token（kb_ 前缀，64+ 字符不可猜测）；DB 只存哈希。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, token = auth.create_user("alice")
    assert token and token.startswith("kb_") and len(token) > 32
    # DB 里没有明文
    row = auth._conn.execute(
        "SELECT api_token, api_token_hash FROM users WHERE user_id=?", (uid,)
    ).fetchone()
    assert row[0] is None
    assert row[1] and row[1] != token


def test_get_user_by_token(tmp_path):
    """token 反查用户；无效/空 token 返回 None。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, token = auth.create_user("alice", role="admin")

    user = auth.get_user_by_token(token)
    assert user is not None
    assert user["user_id"] == uid
    assert user["role"] == "admin"

    assert auth.get_user_by_token("kb_invalid_token") is None  # 假 token
    assert auth.get_user_by_token("") is None                   # 空
    assert auth.get_user_by_token(None) is None                 # None


def test_token_grants_same_access_as_user_id(tmp_path):
    """token 与 user_id 权限等价（同一权限模型）。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, token = auth.create_user("alice")
    auth.grant_access(uid, "default")

    assert auth.can_access(auth.get_user_by_token(token)["user_id"], "default") is True


def test_reset_token_invalidates_old(tmp_path):
    """重置 token 后旧 token 立即失效，新 token 可用。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid, old_token = auth.create_user("alice")

    new_token = auth.reset_token(uid)
    assert new_token and new_token != old_token
    assert auth.get_user_by_token(old_token) is None  # 旧 token 失效
    assert auth.get_user_by_token(new_token)["user_id"] == uid
    assert auth.reset_token("nonexistent") is None


def test_expired_token_rejected(tmp_path):
    """过期 token 反查返回 None（被动失效）。TTL=0 → 立即过期。"""
    auth = AuthStore(str(tmp_path / "u.db"), token_ttl_days=0)
    uid, token = auth.create_user("alice")
    assert auth.get_user_by_token(token) is None  # 已过期
    assert auth.get_user(uid) is not None  # 用户本身还在


def test_migrates_legacy_plaintext_token(tmp_path):
    """旧库明文 token → 重开时迁移为哈希，原 token 仍可用且明文被清空。"""
    import sqlite3

    db = str(tmp_path / "u.db")
    auth = AuthStore(db)
    uid, token = auth.create_user("legacy")
    # 模拟旧版：只存明文 api_token，无 hash
    conn = sqlite3.connect(db)
    conn.execute("UPDATE users SET api_token=? WHERE user_id=?", (token, uid))
    conn.execute("UPDATE users SET api_token_hash=NULL WHERE user_id=?", (uid,))
    conn.commit()
    conn.close()

    auth2 = AuthStore(db)  # 重新打开触发迁移
    assert auth2.get_user_by_token(token)["user_id"] == uid  # 原 token 仍可用
    row = auth2._conn.execute(
        "SELECT api_token, api_token_hash FROM users WHERE user_id=?", (uid,)
    ).fetchone()
    assert row[0] is None  # 明文已清空
    assert row[1]  # 只留哈希


def test_legacy_users_backfilled_with_token(tmp_path):
    """历史无 token 用户：重开 AuthStore 时自动补发（迁移幂等）。"""
    import sqlite3

    db = str(tmp_path / "u.db")
    auth = AuthStore(db)
    uid, old_token = auth.create_user("legacy")
    # 手动抹掉 hash，模拟既无明文也无哈希的旧库数据
    conn = sqlite3.connect(db)
    conn.execute("UPDATE users SET api_token_hash=NULL WHERE user_id=?", (uid,))
    conn.commit()
    conn.close()

    auth2 = AuthStore(db)  # 重新打开触发补发
    new_token = auth2.reset_token(uid)
    assert new_token
    assert auth2.get_user_by_token(new_token)["user_id"] == uid
    assert auth2.get_user_by_token(old_token) is None  # 原 token 已不可恢复


# ==================== admin 引导（生产 bootstrap）====================

def test_bootstrap_admin_creates(tmp_path):
    """引导 token 创建首个 admin（存哈希、永不过期）。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid = auth.bootstrap_admin(name="admin", token="kb_bootstrap_secret_123")
    assert auth.is_admin(uid) is True
    # 原 token 可用且不设过期
    user = auth.get_user_by_token("kb_bootstrap_secret_123")
    assert user is not None and user["user_id"] == uid
    row = auth._conn.execute(
        "SELECT token_expires_at FROM users WHERE user_id=?", (uid,)
    ).fetchone()
    assert row[0] is None  # 永不过期


def test_bootstrap_admin_idempotent(tmp_path):
    """重复引导同一 token 幂等：不新建用户，确认 admin 角色。"""
    auth = AuthStore(str(tmp_path / "u.db"))
    uid1 = auth.bootstrap_admin(name="admin", token="kb_boot_again")
    uid2 = auth.bootstrap_admin(name="admin", token="kb_boot_again")
    assert uid1 == uid2
    assert len(auth.list_users()) == 1


def test_customer_service_roles(tmp_path):
    auth = AuthStore(str(tmp_path / "u.db"))
    agent_id, _ = auth.create_user("坐席", role="agent")
    supervisor_id, _ = auth.create_user("主管", role="supervisor")
    member_id, _ = auth.create_user("普通用户", role="member")

    assert auth.is_customer_service(agent_id) is True
    assert auth.is_customer_service(supervisor_id) is True
    assert auth.is_customer_service(member_id) is False
