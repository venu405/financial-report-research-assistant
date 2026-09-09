# RAG 最终验收方法

## 验收目标

最终验收不只检查“能不能回答”，还要同时确认：检索命中、答案关键事实、引用完整、知识库隔离、资料不足时拒答、主体不明确时追问、对话历史边界、提示词攻击防护和响应时间。

## 测试层次

1. **代码回归**：运行 RAG 的接口、检索、门槛、编排和向量库测试，要求全部通过。
2. **在线正向问答**：法律库和年报库分别回答有明确资料的问题；关键事实、忠实度和引用必须同时通过。
3. **知识库隔离**：把法律问题发给年报库、把年报问题发给法律库，必须拒答且不能返回其他库引用。
4. **防止乱答**：金额等当前资料无法稳定支持的问题必须转人工，不能生成无引用数字。
5. **主体澄清**：只问“上市时间”时必须追问公司名称或股票代码；助手历史中的猜测不能代替用户确认。
6. **对话上下文**：用户历史已经给出公司时不应重复追问，但资料仍不足时必须安全拒答。
7. **攻击与闲聊**：提示词攻击不得泄露规则或密码；闲聊不应进入检索，两类都不应产生引用。
8. **结构与性能**：引用编号必须连续、正文引用不能越界、引用必须来自当前知识库；单条普通问答不超过 15 秒，年报复杂问答不超过 20 秒，规则型追问不超过 2 秒。

## 发布标准

- 核心代码回归：100% 通过。
- 最终在线用例：100% 通过才建议发布；任何正向检索失败、跨库引用、无依据数字或提示词泄露都属于阻断项。
- 正向回答忠实度：法律问题单条不低于 8/10；年报问题单条不低于 7/10。
- 所有正向回答至少有 1 条来源，并且正文引用编号有效。
- 所有拒答和转人工回答必须为 0 条引用，避免“带着错误证据拒答”。

## 执行命令

在 `code/backend` 目录运行：

```powershell
$env:PYTHONPATH='src'
./.venv/Scripts/python.exe -m pytest tests/test_api_kb.py tests/unit/test_kb_gate.py tests/unit/test_kb_retriever.py tests/unit/test_kb_qa_graph.py tests/unit/test_kb_vector_store.py -q
./.venv/Scripts/python.exe scripts/evaluate_quality.py --rag testsets/rag_final_acceptance.yaml --user-id visitor-final-rag --output reports/rag_final_acceptance.json
```

报告中的每个 `checks` 字段都应为 `true`。失败时先看 `category`、`answer`、`top_score`、`faithfulness` 和 `citation_kb_ids`，不要只看总通过数。
