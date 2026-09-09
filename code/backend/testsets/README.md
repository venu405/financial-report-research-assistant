# RAG 与 OCR 评测基线

在 `code/backend` 目录运行（RAG 服务需已启动）：

```powershell
$env:PYTHONPATH='src'
./.venv/Scripts/python.exe scripts/evaluate_quality.py --rag testsets/rag_law.yaml --user-id admin-test --output reports/rag_law.json
./.venv/Scripts/python.exe scripts/evaluate_quality.py --rag testsets/rag_annual.yaml --user-id admin-test --output reports/rag_annual.json
./.venv/Scripts/python.exe scripts/evaluate_quality.py --ocr testsets/ocr_baseline.yaml --output reports/ocr_baseline.json
```

RAG 评测检查可回答/拒答、关键词、引用数量与可选向量分数下限。分数下限只用于同一模型、同一语料的回归比较，不可跨模型使用。

OCR 基线只验证本地 RapidOCR 管线可用和关键字段覆盖。若要量化 OCR 效果，请加入至少 20 份人工标注扫描 PDF/图片，并计算字符错误率（CER）、标题/日期/金额字段准确率及表格字段准确率。
