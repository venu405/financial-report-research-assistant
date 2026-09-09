"""知识库管理（KB）服务包。

架构：
  ingest.py      文档解析 + 分块（入库链路第一步）
  embeddings.py  Embedding 客户端（智谱 OpenAI 兼容）
  vector_store.py Chroma 向量库封装（存/查/删）
  qa_graph.py    LangGraph 问答编排（检索→生成→评估→重试）

设计原则：
  1. 配置化：一切走 Configuration（.env 可控）
  2. 可插拔：vector_store/embeddings 都做薄封装，后续可换 Qdrant/BGE-M3
 3. 容错：解析失败不中断整体入库，记录 notices
"""

from .financial_metric_store import FinancialMetricStore
from .company_analysis import analyze_company
from .peer_comparison import compare_companies
from .research_export import render_company_analysis, render_peer_comparison

__all__ = [
    "FinancialMetricStore",
    "analyze_company",
    "compare_companies",
    "render_company_analysis",
    "render_peer_comparison",
]
