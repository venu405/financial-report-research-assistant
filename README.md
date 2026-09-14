# 上市公司财报问答与研究助手

面向投研、财务和知识管理场景的本地化财报研究原型。用户可以围绕年报/半年报提问，查看页码级原文依据；也可以按公司和报告期整理结构化指标、做有限样本的同业对比，并导出研究底稿。

> 当前定位是可复现的工程原型，不是投资顾问或生产级投研系统。回答必须以资料库证据为准，系统不会替代人工复核，也不提供买卖建议、股价预测或自动行业评级。

## 能力一览

- **财报问答**：支持定期报告问答、公司/年份消歧、数值核验、证据不足时拒答和流式输出。
- **引用溯源**：回答携带知识库、文档、页码/页码范围和原文片段；前端可展开来源，便于回到报告核对。
- **单公司分析**：按公司、报告期读取五项核心指标（营业收入、归母净利润、经营活动现金流净额、资产总额、负债总额），展示同比、利润与经营现金流对照、来源披露定位和待核实事项。
- **同业对比**：手动选择 2–3 家公司和统一报告期；只有期间、报表口径、数据状态一致时才标记为可比，缺失/冲突不会被当作 0，可导出 Markdown 研究底稿，也可基于对比结果生成 LLM 简报（数值仅来自结构化对比结果，LLM 不可用时自动回退规则模板简报）。
- **研究导出与资料治理**：单公司和同业结果可导出；报告资料支持上传、版本审核、发布、回滚、来源同步和权限边界。

## 系统结构

```text
Vue 3 + Vite + TypeScript
          │  /api（Nginx 反向代理）
FastAPI（问答、指标、分析、对比、导出、治理接口）
          │
混合检索：向量召回（Chroma / 可选 Qdrant）+ BM25 关键词召回
          │
重排与问答图：改写 → 召回 → 重排 → 可回答性门槛 → 生成 → 引用/数值核验
          │
Ollama bge-m3（向量） + OpenAI 兼容大模型（默认示例为 DeepSeek）
```

后端入口在 [`code/backend/src/main.py`](code/backend/src/main.py)，核心检索和问答实现位于 `code/backend/src/services/kb/`。结构化切片保留页码、章节、表名、期间、单位、报表口径和相邻块关系；`financial_metric_store.py` 保存原值、归一值、状态及来源定位。默认 Docker Compose 使用 Chroma 持久卷；Qdrant 适配器和验收报告用于可切换后端与固定回归，不代表 Compose 已自动切换到 Qdrant。

## 快速启动

要求：Docker Desktop、Docker Compose，以及可访问的 OpenAI 兼容大模型接口。首次启动会拉取 Ollama 的 `bge-m3` 模型。

```powershell
Copy-Item code/backend/.env.example code/backend/.env
# 编辑 code/backend/.env，至少填写 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL_ID
docker compose up --build
```

打开 <http://localhost:5174>。后端健康检查：<http://localhost:8000/readyz>。停止服务使用 `docker compose down`；数据在 `kb_data` 卷中，除非明确需要，不要使用 `down -v`。

本地开发可分别在 `code/backend` 和 `code/frontend` 安装依赖后启动。`code/backend/data_kb_test`、本地模型、向量库和运行数据库不随 GitHub 展示目录提供；如需复现在线评测，请从恢复清单恢复样本与运行环境，或按自己的语料重新入库。

## 推荐演示流程

1. 启动服务并在“报告管理”上传/发布一份年报。
2. 在“财报问答”询问某公司某年度营业收入或利润变化，打开引用检查文档与页码。
3. 在“公司分析”选择公司和报告期，查看指标、同比、现金流对照及待核实事项。
4. 在“同业对比”手动填写 2–3 家公司和同一报告期，观察不可比项提示，再下载研究底稿。

## 评测与验证

固定 Qdrant 语料回归（185 份报告、集合 `kb_full_codex_20260830_24c7407e`、bge-m3/1024 维）已保留关键证据：

- **36/36，错误数 0**：[`qdrant-full36-after-commitment-eval-fix-20260906.json`](code/backend/reports/qdrant-full36-after-commitment-eval-fix-20260906.json)，配套 lineage 保留；详细 trace 含本地运行日志，已移至恢复目录，不随公开目录提供。
- **34 题防退门禁**：[`qdrant-full36-after-raw-narrative-fix-venv-20260906.json`](code/backend/reports/qdrant-full36-after-raw-narrative-fix-venv-20260906.json) 为 34/36 的基线源报告；门禁清单固定 36 个题目 ID、最低通过数 34、集合和检索配置。
- **最小定向验收**：中国核建数值精度与葫芦娃原因题 2/2 通过，见 [`qdrant-targeted2-after-34-fixes-20260906.json`](code/backend/reports/qdrant-targeted2-after-34-fixes-20260906.json)；详细 trace 含本地运行日志，已移至恢复目录，不随公开目录提供。

离线测试覆盖后端检索、问答图、数值/引用核验、指标层、公司分析、同业对比、导出、Qdrant 适配器和安全边界；前端包含指标、分析、对比、导出及运维工具契约测试。运行示例：

```powershell
cd code/backend
python -m pytest tests/unit tests/test_api_kb.py
python -m ruff check src tests scripts
cd ../frontend
npm test
npm run build
```

在线 36 题报告是固定语料、固定模型和固定配置下的一次结果，不等于所有公司、期间和问法的准确率承诺。报告中的页码命中、回答质量和拒答行为仍应结合原文人工复核。

## 目录说明

```text
code/backend/       FastAPI 服务、问答图、指标与评测脚本
code/frontend/      Vue 3 界面和前端契约测试
code/backend/tests/ 后端单元/集成测试
code/backend/testsets/ 财报 RAG 题集与防退门禁清单
code/backend/reports/ 仅保留少量验收证据
.github/workflows/  CI 配置
docker-compose.yml  本地全栈部署
migration-record/   本次本地资产迁移清单
```

## 已知限制

- 默认 Compose 仍以 Chroma 为持久化向量库；Qdrant 评测使用独立集合和独立服务，切换后必须重新核对连接、集合、向量维度和语料。
- 指标抽取依赖报告版式和解析质量；单位、合并/母公司、年度/半年度、调整前后等口径不一致时系统会标记不可比，但无法替人工判断业务可比性。
- 同业对比只描述用户选择的 2–3 家样本，不证明严格同行关系，也不代表行业全貌。
- 证据不足时会拒答或列入待核实事项；模型可能遗漏披露原因，引用和数字仍需查看原文。
- 未提供实时行情、预测、回测、自动评级、权限审计的云端部署和多租户生产保障。

## License / 使用提醒

本项目代码基于 [MIT License](LICENSE) 发布。

仓库中的年报样本、运行数据、模型和截图属于本地复现资产，已移出 GitHub 展示目录。使用外部报告时请遵守来源网站、版权和公司内部数据合规要求；发布前请自行补充数据来源声明。
