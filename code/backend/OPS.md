# 运维手册（OPS Runbook）

企业知识库管理助手的日常运维操作。数据目录默认 `code/backend/`，含 Chroma 向量库与全部 `kb_*.db` SQLite 业务库。

---

## 0. 密钥管理（P0 运维建议）

- `.env` 里的 DeepSeek/百度 key 是明文（已 `.gitignore`，不会入 git），本地/小规模部署可接受。
- **生产环境**建议改用环境变量注入：容器编排（K8s Secret / Docker secret）或 CI/CD 的加密变量注入，不要落盘 `.env`。
- 云上可上云 KMS（如腾讯云 KMS）托管密钥，应用启动时拉取。
- 日志侧已用 `_mask_secret` 掩码，不会打全量 key；但**不要**在日志里打印任何明文凭据。
- 泄露应急：DeepSeek/百度控制台**立即吊销旧 key**，换新 key 后更新注入。
- Docker Compose 会设置 `APP_ENV=production`。此模式下若没有真实的 `ADMIN_API_KEY`，
  或 `KB_REQUIRE_TOKEN` 不是 `1`，后端会直接拒绝启动，避免误把无鉴权服务暴露出去。
- 前端不会再把用户 token 和管理员 key 长期写入浏览器；关闭当前标签页后需要重新输入。

启动容器前，请在 `code/backend/.env` 至少设置：

```dotenv
ADMIN_API_KEY=<强随机密钥>
KB_REQUIRE_TOKEN=1
KB_BOOTSTRAP_ADMIN_TOKEN=<首次管理员令牌>
KB_BACKUP_SIGNING_KEY=<至少32位随机密钥>
KB_REQUIRE_BACKUP_SIGNATURE=1
```

随后运行 `docker compose up --build`。前端由 Nginx 提供静态文件，并通过同源 `/api`
转发到后端；Compose 会等待 Ollama、后端 `/readyz` 和前端依次健康后再启动下游服务。

## 1. 备份与只读校验

```bash
cd code/backend
./.venv/Scripts/python.exe scripts/backup_kb.py
# 输出到 code/backend/backups/kb_<时间戳>/，含 manifest.json
./.venv/Scripts/python.exe scripts/verify_backup.py --backup-dir backups/kb_<时间戳>
```

- SQLite：自动发现数据目录下全部 `kb_*.db`（用户、会话、FAQ、反馈、工单、元数据、审计、检索日志等，未来新增同名数据库也会自动纳入），用 `VACUUM INTO` 做一致性快照，备份时业务写不中断。
- Chroma 目录（`chroma_data/`）和原始文档归档（`document_originals/`）整体复制。
- `manifest.json` 格式版本为 2，记录备份时间、来源目录、Python/Chroma 版本、总字节数，以及每个 SQLite 文件和 Chroma 目录内每个文件的大小与 SHA-256。
- `verify_backup.py` **只读**：校验清单结构、文件大小和 SHA-256、每个 SQLite（包括 Chroma 的 SQLite 文件）的 `integrity_check`，以及目录完整文件清单。校验失败时禁止进入恢复流程。
- 生产建议设置 `KB_BACKUP_SIGNING_KEY` 和 `KB_REQUIRE_BACKUP_SIGNATURE=1`。备份会生成 `manifest.sig`，防止数据文件与清单被同时替换；签名密钥必须放在备份目录之外。
- 建议每天备份后立即校验；保留最近 7 份日备份，并按月保留 12 份。备份应复制到独立磁盘或对象存储，不能只留在应用主机。

## 2. 恢复演练（非生产）

```bash
cd code/backend
# 目标目录必须由操作人预先创建，且必须为空；不要使用现网数据目录。
mkdir D:/kb-recovery-drills/2026-08-21
./.venv/Scripts/python.exe scripts/verify_backup.py --backup-dir backups/kb_<时间戳>
./.venv/Scripts/python.exe scripts/restore_drill.py \
  --backup-dir backups/kb_<时间戳> \
  --target-dir D:/kb-recovery-drills/2026-08-21
```

- `restore_drill.py` 先只读校验备份；仅可写入**显式指定、已存在且为空**的目录。目标为非目录、非空、备份目录本身/其子目录，或与 manifest 中来源数据目录相同，都会拒绝执行。
- 演练完成后脚本会自动再次运行校验，并在临时副本中实际打开 Chroma、枚举集合和读取计数。之后可在该目录启动隔离副本，检查 `/readyz`、知识库文档数、一次问答和一条会话/工单记录；演练目录经审批后再清理。
- 不得把该脚本用于生产恢复，也不得将生产数据目录作为 `--target-dir`。

## 3. 生产恢复、回滚与目标

- **RPO：24 小时。** 每日备份完成并校验成功后，最多损失最近一次成功备份以来的数据；重大批量导入、版本升级前额外执行一次备份和校验。
- **RTO：2 小时。** 适用于备份位于同地域可访问存储、Chroma 版本一致、且已完成最近一次演练的场景；跨地域取回或版本回退需重新评估。
- 生产恢复必须走变更审批：停写入服务 → 对现网数据目录制作并校验应急快照 → 在隔离临时目录恢复并校验目标备份 → 通过可回退的卷挂载或目录切换切入 → `/readyz` 和业务抽查通过后开放流量。
- **回滚：** 保留切换前的数据目录/卷和应急快照，不做就地删除；若健康检查或抽查失败，停止新实例、重新挂回原数据目录/卷，再按事故流程排查。恢复前先核对 `manifest.json` 的 `chroma_version` 与当前环境是否兼容。
- 计划：每日“备份 + 校验”，每月一次隔离恢复演练，每次 Chroma 升级、数据迁移或灾备切换前后各做一次。把备份、校验和演练结果记录到值班台账。

## 4. Chroma 升级（版本锁定）

- 当前锁定 `chromadb==1.5.9`（见 `requirements.txt`）。
- 升级前**必须备份**，因为 Chroma 版本间段落格式可能不兼容。
- 升级步骤：备份 → `pip install chromadb==<新版本>` → 起服务 → 跑 `/readyz` 探活 Chroma → 抽查 `/kb/docs` 列表是否正常 → 全量 `pytest`。
- 若升级后数据读不出，回退：降级版本 + 用上一份备份恢复。

## 5. Token 轮换

- **管理员 token 泄露**：用 `POST /kb/users/{id}/token` 重置（旧 token 立即失效），或重启后换 `KB_BOOTSTRAP_ADMIN_TOKEN`。
- **普通用户 token 泄露**：管理员调重置接口，或直接 `POST /kb/users/{id}/token`。
- token 默认 90 天过期（`KB_TOKEN_TTL_DAYS` 可调），过期自动失效。
- 全部 token 作废（极端情况）：删除 `kb_users.db` 后重启 + 重新 bootstrap admin。

## 6. On-call 常见故障

| 症状 | 排查 |
|------|------|
| `/kb/ask` 慢/超时 | 看 `/admin/metrics` 的 ask 延迟分位与错误计数；再按 request_id 查应用日志 |
| 上传 413 | 文件超 `KB_MAX_UPLOAD_MB`（默认 50MB），调大或压缩 |
| 401/403 异常 | 看 `/admin/audit` 审计日志定位是谁、什么操作被拒 |
| 服务起不来 | 生产模式会拒绝缺少 `ADMIN_API_KEY` 或未启用 `KB_REQUIRE_TOKEN=1` 的配置；首次部署再检查 `KB_BOOTSTRAP_ADMIN_TOKEN` |
| 磁盘满 | 备份目录 `backups/` 会累积，定期清理；`_upload_*` 临时文件残留可删 |

## 6.1 企业系统接入 API v1

已有投研平台、OA 或文档系统应优先调用 `/api/v1`，旧的 `/kb/*` 接口继续保留兼容前端。
所有 v1 成功响应包含 `request_id`，同时通过 `X-Request-Id` 响应头返回；调用方可传入符合
`[A-Za-z0-9._:-]` 且不超过 128 字符的 `X-Request-Id` 以串联跨系统日志，非法值会被替换为新编号。
v1 身份凭证支持 `Authorization: Bearer <api_token>` 和 `X-Api-Token: <api_token>`，v1 始终要求
其中一个有效 token；两者同时传入时必须完全一致。两者都进入同一套用户、角色和知识库权限校验，
不要仅依赖 `knowledge_base_id` 或请求体中的 `user_id` 判断权限。

问答问题最多 8000 字符、历史最多 20 条且单条内容最多 4000 字符；公司名、报告期、指标代码和
知识库编号均有长度上限，同行公司只能传 2 至 3 家且不能重复。`metadata_filters` 仅允许受控字段，
JSON 大小不超过 4096 字节。

当前接口：

- `POST /api/v1/query`（兼容别名 `/api/v1/queries`）
- `POST /api/v1/query/stream`（SSE，兼容别名 `/api/v1/queries/stream`）
- `GET /api/v1/financial-metrics`
- `POST /api/v1/company-analyses`
- `POST /api/v1/peer-comparisons`
- `POST /api/v1/reports/export`（JSON 返回 Markdown 内容、文件名和 `request_id`）

问答 v1 不返回检索上下文、评分和重试等内部调试字段；公司分析、同行对比和指标响应中的额外字段
仅为兼容现有业务而保留，未列入响应模型核心字段的扩展内容不属于稳定契约。SSE 接口固定声明
`text/event-stream`，事件中的 `request_id` 与响应头一致。

文档导入暂不提供 `/api/v1/documents/import` 和持久任务查询接口：现有 `/kb/ingest` 是同步上传，
直接在接入层包装会导致长请求、任务状态不可恢复。待接入对象存储/任务队列后，再复用同一入库服务
实现异步任务和回调，避免复制解析、分块和写库逻辑。

## 7. 环境变量速查

| 变量 | 默认 | 说明 |
|------|------|------|
| `APP_ENV` | development | `production` 时强制检查生产鉴权配置 |
| `ADMIN_API_KEY` | 空 | 管理接口 X-API-Key（**生产必配**） |
| `KB_BOOTSTRAP_ADMIN_TOKEN` | 空 | 首次引导 admin 的 token |
| `KB_OPEN_SIGNUP` | 0 | 1=允许自助注册（强制 member） |
| `KB_REQUIRE_TOKEN` | 0 | 1=强制 token 鉴权，禁 user_id 直传 |
| `KB_ENFORCE_KB_VISIBILITY` | 0 | 1=开发环境也强制公开/内部知识库边界；生产环境自动强制 |
| `KB_TOKEN_TTL_DAYS` | 90 | token 有效期 |
| `KB_MAX_UPLOAD_MB` | 50 | 上传大小上限 |
| `BAIDU_OCR_API_KEY` | 空 | 百度 OCR 后备通道 API Key（配了才启用百度兜底） |
| `BAIDU_OCR_SECRET_KEY` | 空 | 百度 OCR Secret Key |
| `KB_BAIDU_OCR_QPS` | 2 | 百度 OCR 每秒调用上限（防打爆配额） |
| `KB_OCR_FALLBACK_CONFIDENCE` | 0.6 | auto 模式本地 OCR 置信度低于此值触发百度兜底 |
| `KB_MIN_SIMILARITY` | 0 | 向量相关性阈值（0=不过滤；BGE-M3 相关文档通常 0.5+，实测后调） |
| `KB_SYNC_ALLOWED_ROOT` | `./sync_sources` | 文件数据源允许读取的根目录；相对路径按此目录解析 |
| `KB_SYNC_ALLOWED_HOSTS` | 空 | HTTPS 数据源域名白名单，逗号分隔；重定向后会再次校验 |
| `KB_AGENT_SLA_MINUTES` | 10 | 转人工后的首次真人回复时限 |
| `KB_ALERT_WAITING_COUNT` | 20 | 待接入会话积压告警阈值 |
| `KB_ALERT_MIN_FREE_GB` | 2 | 数据盘剩余空间告警阈值 |
| `KB_AUTO_RETENTION` | 0 | 开发环境设 1 后每日自动执行保留清理；生产默认自动执行 |
| `KB_BACKUP_SIGNING_KEY` | 空 | 备份清单签名密钥，生产应从安全环境注入 |
| `KB_REQUIRE_BACKUP_SIGNATURE` | 0 | 设 1 后拒绝验证无签名备份 |

## 8. 内容治理、隐私和客服值班

- 普通上传会立即形成已发布基线；后续可在文档页上传草稿，依次送审、批准、发布或回滚。原始文件保存在受控目录，接口不会暴露真实磁盘路径。
- 定时数据源只生成“待审核版本”，不会直接覆盖线上知识库。后台和手工同步共用 SQLite 租约，避免多进程重复同步。
- 长期记忆默认关闭；用户明确同意后才保存，关闭同意会立即清空。手机号、身份证、银行卡、邮箱和常见密钥会在消息、反馈、检索日志和记忆写入前脱敏。
- 管理页可预览或执行保留清理。清理覆盖过期记忆、检索、反馈、审计、已到期会话消息、LangGraph 检查点和已关闭工单。
- 坐席“领取”不算首响，只有真人首次回复才完成 SLA。运营看板展示会话、工单、满意度、RAG 质量和告警；异常恢复后告警会自动关闭，再次发生会重新打开。
- 工单绑定 `kb_id`，客服只能查看和处理获授权知识库的工单；主管指派前会校验目标客服账号和知识库权限。

> 百度 OCR 是可选后备通道：`BAIDU_OCR_API_KEY` / `SECRET_KEY` 留空则纯本地识别（零联网零费用）。
> 这两个变量**只从环境变量读**（`load_dotenv` 加载 .env），不进 Configuration 对象——排障时认准
> 环境变量即可，启动日志会打印 `Baidu OCR fallback: configured=...`。

> **Ollama 热加载延迟**：Ollama 默认 keep_alive 5 分钟后卸载模型，隔段时间首个 embedding 请求要
> 重新加载 1.2GB 模型（秒级延迟尖峰）。部署时建议 `ollama run bge-m3 --keepalive 30m`（或修改
> Ollama 环境变量 `OLLAMA_KEEP_ALIVE=30m`），保持模型常驻。

## 9. 一键增量入库

首次或需要重建时，在后端服务已经启动并且 `/readyz` 正常的前提下，运行入库脚本（参数原样透传）：

```bash
cd code/backend
PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_corpus.py --dry-run
PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_corpus.py --user-id <有写权限的用户ID>
PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_corpus.py --force
```

未设置 `KB_INGEST_TOKEN` / `KB_API_TOKEN` / `KB_INGEST_USER_ID` / `KB_USER_ID` 时，
脚本使用本地开发身份 `KB_INGEST_USER_ID=admin`。

脚本默认把可续跑清单写到 `code/backend/.rag_state/ingest_corpus_state.json`。每份 PDF
都会流式算 SHA-256；非 `--dry-run` 时，脚本会先在 `/readyz` 后分页读取目标知识库的完整
文档清单（每页500条），再结合同一 `kb_id` 下的真实 `doc_id`、标题和 chunks 判定动作：
远端存在且内容未变才显示 `SKIP`，新文件使用 `POST`，只有已在目标库清单中的变更文件才使用
`PUT`。状态中的 doc_id 不在目标库时绝不会 PUT；远端同名多文档、状态缺失但远端同名、或
chunks<=0 均会失败并要求人工核对。失败项会保留在清单中，下一次自动重试；服务端返回 409
不算成功，需先核对清单和服务端文档。`--directory` 可改为递归扫描目录，`--testset`
可改为其他题集，`--state` 可指定另一份清单，`--ocr-mode local|baidu|auto` 可选择识别方式。

如果健康检查失败，先启动前后端（`docker compose up --build`，或本地运行 uvicorn），
确认后端已就绪后再重试。规则指纹
变化只会给出“规则已变化，是否用 `--force` 重建”的提醒，不会自动全量重建。状态清单必须
定期备份；丢失后脚本不会凭标题自动认领服务端文档，遇到同名文档会失败并提示恢复清单或
人工核对。不要手工覆盖状态清单；状态每完成一份都会用临时文件替换方式保存，并记录
`last_run.run_status`、计划数、完成数和各动作计数，进程中断后可从最近一次有效快照继续。

## 10. 本地模型部署、切换与回滚

首次部署或需要刷新别名时，在后端虚拟环境已创建的前提下运行：

```bash
cd code/backend
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup_local_llm.ps1
```

它会检查 Ollama 和 `code/backend/.venv`，幂等拉取 `qwen3.5:4b`，依据受版本控制的
`code/backend/Modelfile.qwen35-4b` 创建 `enterprise-kb-qwen35:4b`，并固定 `num_ctx=8192`；然后
安装/同步正式依赖 `fastembed`（保留 `onnxruntime`），实际加载一次
`BAAI/bge-reranker-base`，最后通过 Ollama 的 OpenAI 兼容接口执行 `reasoning_effort=none` 的
ASCII 严格格式冒烟。任一步失败都会返回非 0，不会打印或读取 DeepSeek 密钥。

本机 RTX 4060 Laptop 约 8GB 显存只推荐 Qwen3.5 4B Q4_K_M；首次下载基础模型约 3.4GB，实际
空间和耗时以 Ollama 当前版本为准。不要把 9B 作为本机默认模型，也不要尝试与 4B 同时常驻显存。
重排器还会在首次加载时下载 `BAAI/bge-reranker-base`，其设备分配由 fastembed/onnxruntime 决定，
不应把 `ollama ps` 的结果当作重排器占用。用下列命令检查 Ollama 模型及当前 LLM 的处理器分配：

```text
ollama list
ollama ps
```

部署成功后，在启动进程的 shell 中设置以下环境变量（只影响当前进程及其子进程，
不改写 `.env`），再启动前后端：

```bash
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL_ID=enterprise-kb-qwen35:4b
export LLM_REASONING_EFFORT=none
export KB_RERANK_MODE=crossencoder
export KB_RERANK_MODEL=BAAI/bge-reranker-base
export LLM_TIMEOUT=180
docker compose up --build   # 或本地 uvicorn main:app --app-dir src
```

启动前请先确认 `enterprise-kb-qwen35:4b` 已创建。如果 8000 端口已有后端，
应先关闭旧进程再启动，因为无法保证现有进程已经切换到本地配置。

回滚 DeepSeek 时，先按现有运维流程关闭本地启动的项目进程，再使用 `.env` 中的
`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_ID` 配置重新启动；不要把上面的本地模型
环境变量带进启动 shell。回滚后应检查 `/readyz`
和一次问答，确认服务实际读取的是 DeepSeek 地址。

安装/冒烟成功不代表问答质量达标。两边代码合并后，必须运行完整 36 题真实评测并保存报告，
分别比较本地模型与 DeepSeek 的正确性、拒答安全性、延迟和显存/内存占用；不能用单条冒烟结果
宣称本地模型已经通过质量验收。
