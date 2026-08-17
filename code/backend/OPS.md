# 运维手册（OPS Runbook）

企业知识库管理助手的日常运维操作。数据目录默认 `code/backend/src/`，含 Chroma 向量库与三个 SQLite 库。

---

## 1. 备份

```bash
cd code/backend
./.venv/Scripts/python.exe scripts/backup_kb.py
# 输出到 code/backend/backups/kb_<时间戳>/，含 manifest.json
```

- SQLite（`kb_users.db` / `kb_checkpoints.db` / `kb_audit.db`）用 `VACUUM INTO` 一致性快照，备份时业务写不中断。
- Chroma 目录（`chroma_data/`）整体复制。
- 建议 cron/计划任务每日一次，保留最近 7 份。

## 2. 恢复

```bash
# 停服务
# 把备份目录里的内容覆盖回 src/
cp backups/kb_<时间戳>/kb_*.db src/
rm -rf src/chroma_data && cp -r backups/kb_<时间戳>/chroma_data src/
# 起服务
```

恢复前先比对 `manifest.json` 的 `chroma_version` 与当前环境是否一致（见下）。

## 3. Chroma 升级（版本锁定）

- 当前锁定 `chromadb==1.5.9`（见 `requirements.txt`）。
- 升级前**必须备份**，因为 Chroma 版本间段落格式可能不兼容。
- 升级步骤：备份 → `pip install chromadb==<新版本>` → 起服务 → 跑 `/readyz` 探活 Chroma → 抽查 `/kb/docs` 列表是否正常 → 全量 `pytest`。
- 若升级后数据读不出，回退：降级版本 + 用上一份备份恢复。

## 4. Token 轮换

- **管理员 token 泄露**：用 `POST /kb/users/{id}/token` 重置（旧 token 立即失效），或重启后换 `KB_BOOTSTRAP_ADMIN_TOKEN`。
- **普通用户 token 泄露**：管理员调重置接口，或直接 `POST /kb/users/{id}/token`。
- token 默认 90 天过期（`KB_TOKEN_TTL_DAYS` 可调），过期自动失效。
- 全部 token 作废（极端情况）：删除 `kb_users.db` 后重启 + 重新 bootstrap admin。

## 5. On-call 常见故障

| 症状 | 排查 |
|------|------|
| `/kb/ask` 慢/超时 | 看 `/admin/diag` 的 token 消耗；`/admin/metrics` 看 ask 延迟分位 |
| 上传 413 | 文件超 `KB_MAX_UPLOAD_MB`（默认 50MB），调大或压缩 |
| 401/403 异常 | 看 `/admin/audit` 审计日志定位是谁、什么操作被拒 |
| 服务起不来 | `ADMIN_API_KEY` 未配置会有告警日志；看 `KB_BOOTSTRAP_ADMIN_TOKEN` 是否设置 |
| 磁盘满 | 备份目录 `backups/` 会累积，定期清理；`_upload_*` 临时文件残留可删 |

## 6. 环境变量速查

| 变量 | 默认 | 说明 |
|------|------|------|
| `ADMIN_API_KEY` | 空 | 管理接口 X-API-Key（**生产必配**） |
| `KB_BOOTSTRAP_ADMIN_TOKEN` | 空 | 首次引导 admin 的 token |
| `KB_OPEN_SIGNUP` | 0 | 1=允许自助注册（强制 member） |
| `KB_REQUIRE_TOKEN` | 0 | 1=强制 token 鉴权，禁 user_id 直传 |
| `KB_TOKEN_TTL_DAYS` | 90 | token 有效期 |
| `KB_MAX_UPLOAD_MB` | 50 | 上传大小上限 |
| `KB_RESEARCH_TOKEN_BUDGET` | 100000 | 单次研究 token 预算 |
