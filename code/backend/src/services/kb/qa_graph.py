"""LangGraph 问答编排：检索增强生成（RAG）工作流。

这是项目首次引入 LangGraph——用它替代深度研究项目里"手写 run_stream 循环"的做法。

为什么这里用 LangGraph？（面试必讲）
  1. 状态图原生表达"分支/条件边"：检索→生成→评估→(不满意→重试)
  2. checkpointer：对话历史持久化 + 断点续跑（LangGraph 内置）
  3. 每个节点可单独测试、可视化（LangGraph Studio）

对比深度研究项目的手写编排：
  - 深度研究：手写 Thread+Queue 生成器（要完全掌控 SSE 事件流）→ 保留手写
  - 知识库问答：状态图流程（检索/生成/评估是天然图结构）→ 用 LangGraph
  - 结论：按场景选工具，不为用框架而用框架

图结构：
  rewrite(查询改写) → retrieve(检索) → generate(生成) → evaluate(评估)
                                                     └─(不达标)→ 回到 generate 重试
                                                     └─(达标)→ END
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from services.kb.embeddings import EmbeddingClient
from services.kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_RETRY = 1  # 生成后评估不达标，最多重试 1 次

# LLM 生成调用超时（秒）：Ollama/网络 hang 时不拖死请求，及时报错让上层处理。
# 可用 LLM_TIMEOUT 环境变量覆盖（与 HelloAgents 一致，默认 60）。
_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60") or 60)


def _llm_invoke(llm, messages: list[dict[str, str]], model: str = "deepseek-chat") -> str:
    """统一 LLM 调用：兼容 OpenAI 客户端与 LangChain 风格 LLM。

    - OpenAI 客户端（本项目实际使用）：llm.chat.completions.create(model=model, ...)
    - LangChain 风格（mock/其他）：llm.invoke(messages)

    model 显式传参（P1 修复：去掉 llm._model 私有属性 hack，openai 升级不失效）。
    timeout（P0）：防 LLM/网络 hang 拖死请求。
    """
    if hasattr(llm, "invoke"):
        return str(llm.invoke(messages)).strip()
    # OpenAI 兼容客户端
    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        timeout=_LLM_TIMEOUT,
    )
    return (resp.choices[0].message.content or "").strip()


class QaState(TypedDict):
    """图状态：节点间传递的共享数据（LangGraph 的 State）。"""

    question: str                    # 原始问题
    kb_id: str                       # 所属知识库（检索范围限定，P3）
    rewritten: str                   # 改写后问题（多轮上下文）
    history: list[dict[str, str]]    # 对话历史
    contexts: list[dict[str, Any]]   # 检索到的分块
    answer: str                      # 生成的答案
    citations: list[dict[str, Any]]  # 引用（chunk 元数据）
    retries: int                     # 已重试次数
    low_quality: bool                # 评估结果：是否不达标（条件边读它）
    score: int                       # P4：忠实度评分 0-10（评估节点写入，前端展示）
    passages: list[str]              # 上下文文本（喂给 LLM）


def build_qa_graph(
    *,
    llm,
    embeddings: EmbeddingClient,
    vector_store: VectorStore,
    top_k: int = 5,
    hybrid: bool = True,
    checkpointer: Any = None,
    model: str = "deepseek-chat",
    min_score: float = 0.0,
) -> Any:
    """构建 LangGraph 问答图。llm 为 OpenAI 兼容客户端（DeepSeek）。

    hybrid=True 时用混合检索（向量 + BM25 + RRF），False 时退回纯向量。
    checkpointer（P4）：传入 LangGraph checkpointer（如 SqliteSaver）后，
      对话状态按 thread_id 持久化，支持跨请求恢复与断点续跑；None 时不持久化。
    min_score（P0）：向量相关性阈值，余弦相似度低于它视为未命中（0=不过滤）。
    """
    from services.kb.retriever import HybridRetriever, VectorOnlyRetriever

    if hybrid:
        retriever: Any = HybridRetriever(
            vector_store, embeddings=embeddings, top_k=top_k, min_score=min_score
        )
    else:
        retriever = VectorOnlyRetriever(
            vector_store, embeddings=embeddings, top_k=top_k, min_score=min_score
        )

    def node_rewrite(state: QaState) -> dict[str, Any]:
        """查询改写：结合对话历史，把当前问题改写成自包含的检索查询。

        场景：用户问"它支持 PDF 吗？"（"它"指代上文的文档系统）
        → 改写为"知识库系统是否支持 PDF 文档导入"
        """
        question = state["question"]
        history = state.get("history", [])
        if not history:
            return {"rewritten": question}  # 无历史不改写

        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:100]}"
            for m in history[-4:]  # 只看最近 4 轮
        )
        prompt = (
            "基于对话历史，把当前问题改写成独立可检索的查询。\n"
            f"对话历史：\n{history_text}\n"
            f"当前问题：{question}\n"
            "只输出改写后的查询，不要解释。"
        )
        rewritten = _llm_invoke(llm, [{"role": "user", "content": prompt}], model=model)
        # 🟡15：改写结果校验——LLM 返回寒暄/解释类垃圾文本时回退原问题，
        # 避免垃圾文本被当检索词用（召回质量崩塌）。
        # 合法改写应为单行短查询：含换行（解释/多段）或超长（跑题）都判为无效。
        rewritten = rewritten.strip()
        if not rewritten or "\n" in rewritten or len(rewritten) > 120:
            logger.warning("查询改写结果异常（%.0f 字符），回退原问题", len(rewritten))
            return {"rewritten": question}
        return {"rewritten": rewritten}

    def node_retrieve(state: QaState) -> dict[str, Any]:
        """检索：混合检索（向量 + BM25 + RRF 融合）。"""
        query = state.get("rewritten") or state["question"]
        hits = retriever.search(query, top_k=top_k, kb_id=state.get("kb_id", "default"))
        passages = [h["text"] for h in hits]
        return {
            "contexts": hits,
            "passages": passages,
        }

    def node_generate(state: QaState) -> dict[str, Any]:
        """生成：把检索到的分块作为上下文，LLM 生成带引用的答案。"""
        passages = state.get("passages", [])
        if not passages:
            return {
                "answer": "知识库中未检索到相关信息，请尝试换个问法或补充文档。",
                "citations": [],
            }

        # 构造上下文：编号分块，让 LLM 用 [1][2] 标注引用
        context_block = "\n\n".join(
            f"[{i+1}] {p}" for i, p in enumerate(passages)
        )
        prompt = (
            "你是企业知识库助手。仅基于以下资料回答用户问题。\n"
            "要求：\n"
            "1. 若资料不足，明确说明缺少相关信息\n"
            "2. 回答末尾用 [编号] 标注引用来源（如 [1][2]）\n"
            f"资料：\n{context_block}\n\n"
            f"问题：{state['question']}\n"
            "回答："
        )
        answer = _llm_invoke(llm, [{"role": "user", "content": prompt}], model=model)

        # 引用元数据：与答案里的 [n] 对应。
        # P2：平铺定位字段（doc_title/chunk_index/page），前端可直接跳转原文，不用钻 metadata。
        all_citations = []
        for i, h in enumerate(state.get("contexts", [])):
            meta = h.get("metadata", {}) or {}
            all_citations.append(
                {
                    "index": i + 1,
                    "chunk_id": h.get("chunk_id", ""),
                    "text": h.get("text", "")[:120],
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),  # PDF 分页时记录（无则为 None）
                    "source_type": meta.get("source_type", ""),
                    "metadata": meta,
                }
            )
        # P2 修复：只保留答案里实际引用的 [n] 且在有效范围内（防 LLM 编 [9] 悬空）
        import re as _re

        # 🟡14：只解析答案末尾行的引用标记——prompt 已要求"末尾标注引用"，
        # 正文里的 [2026年]、"见[1]章节" 等不应被误判为引用；
        # 末行无标记时回退全文扫描（兼容 LLM 内联标注的习惯）
        non_empty_lines = [ln for ln in answer.splitlines() if ln.strip()]
        citation_zone = non_empty_lines[-1] if non_empty_lines else ""
        cited = {int(m) for m in _re.findall(r"\[(\d+)\]", citation_zone)}
        if not cited:
            cited = {int(m) for m in _re.findall(r"\[(\d+)\]", answer)}
        citations = [c for c in all_citations if c["index"] in cited and 1 <= c["index"] <= len(all_citations)]
        return {"answer": answer, "citations": citations}

    def _evaluate_faithfulness(state: QaState) -> tuple[int, bool]:
        """忠实度评估（LLM 打分版，P2 升级；P4 返回分数供展示）。

        让 LLM 判断"答案是否基于给定资料"（0-10 分，<6 视为不达标）。
        返回 (score, passed)：score 给前端展示，passed 给条件边判定。
        LLM 评估比简单规则更准：能识别"答案没引用资料却胡编"的情况。
        评估失败（LLM 抖动/解析失败）→ 退回简单规则检查（保守收敛）。
        """
        answer = (state.get("answer") or "").strip()
        passages = state.get("passages", [])
        if not answer:
            return 0, False  # 空答案必不达标
        if not passages:
            return 10, True  # 无资料时诚实声明即可，不重试

        context_block = "\n\n".join(
            f"[{i+1}] {p[:300]}" for i, p in enumerate(passages[:4])
        )
        prompt = (
            "你是 RAG 质量评估员。从两个维度评估下面的【答案】：\n"
            "1. 忠实度（是否严格基于资料，无编造）：10=完全基于，0=无关/编造\n"
            "2. 相关性（是否回答了用户问题）：10=切题，0=答非所问\n\n"
            f"【用户问题】{state.get('question', '')}\n"
            f"【资料】\n{context_block}\n\n"
            f"【答案】\n{answer}\n\n"
            "只输出两个 0-10 的整数，格式：忠实度 相关性（空格分隔），不要解释。"
        )
        try:
            raw = _llm_invoke(llm, [{"role": "user", "content": prompt}], model=model)
            import re as _re

            nums = _re.findall(r"\d+", raw)
            faith = int(nums[0]) if nums else 0
            relev = int(nums[1]) if len(nums) > 1 else faith
            score = min(faith, relev)  # 取低分（任一维度差都算不达标）
            logger.info("RAG 评估：忠实度 %d/10，相关性 %d/10，取低 %d/10", faith, relev, score)
            return score, score >= 6
        except Exception:
            logger.warning("忠实度评估失败，退回规则检查")
            passed = bool(answer) and ("未检索到" not in answer)
            return (10 if passed else 0), passed

    def node_evaluate(state: QaState) -> dict[str, Any]:
        """评估：LLM 忠实度打分（P2 升级版），失败退回规则检查。

        结果写入 state.low_quality，供条件边（route_after_evaluate）读取——
        保证"评估判定"和"路由决策"用同一份结论，不会出现一边判重试一边放行。
        """
        answer = state.get("answer", "").strip()
        has_passages = bool(state.get("passages"))
        retries = state.get("retries", 0)

        # 简单规则前置检查（零成本快速拦截明显问题）
        low_quality = not answer or ("未检索到" in answer and has_passages)
        score = 0
        if not low_quality:
            # 通过规则检查后，再用 LLM 深度评估忠实度
            score, passed = _evaluate_faithfulness(state)
            low_quality = not passed

        if low_quality and retries < MAX_RETRY:
            return {"retries": retries + 1, "low_quality": low_quality, "score": score}
        return {"retries": retries, "low_quality": low_quality, "score": score}

    def route_after_evaluate(state: QaState) -> str:
        """条件边：不达标且有重试额度 → 回 generate；否则结束。"""
        if state.get("low_quality") and state.get("retries", 0) < MAX_RETRY:
            return "generate"  # 回到生成节点重试
        return END

    # ---- 组装图 ----
    builder = StateGraph(QaState)
    builder.add_node("rewrite", node_rewrite)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("generate", node_generate)
    builder.add_node("evaluate", node_evaluate)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"generate": "generate", END: END},
    )

    return builder.compile(checkpointer=checkpointer)


def run_qa(
    graph: Any,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    kb_id: str = "default",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """执行问答图，返回 {answer, citations, contexts, score}。

    kb_id：限定检索的知识库（多库隔离，P3）。
    thread_id（P4）：对话线程 ID。传同一 ID 时 LangGraph checkpointer 持久化对话
      状态（SQLite），支持跨请求恢复与断点续跑；不传则每次自动开新线程。
    """
    initial: QaState = {
        "question": question,
        "kb_id": kb_id,
        "rewritten": "",
        "history": history or [],
        "contexts": [],
        "passages": [],
        "answer": "",
        "citations": [],
        "retries": 0,
        "low_quality": False,
        "score": 0,
    }
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    result = graph.invoke(initial, config=config)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "contexts": result.get("contexts", []),
        "retries": result.get("retries", 0),
        "score": result.get("score", 0),
    }


def run_qa_stream(
    graph: Any,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    kb_id: str = "default",
    thread_id: str | None = None,
) -> Any:
    """流式执行问答图：yield 节点进度事件 + 最终结果。

    P2：/kb/ask/stream 用——同步 ask 用户干等 10s+ 无反馈，这里按节点
    （rewrite→retrieve→generate→evaluate）推送进度，SSE 前端可实时展示。
    """
    initial: QaState = {
        "question": question,
        "kb_id": kb_id,
        "rewritten": "",
        "history": history or [],
        "contexts": [],
        "passages": [],
        "answer": "",
        "citations": [],
        "retries": 0,
        "low_quality": False,
        "score": 0,
    }
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    final_state: dict[str, Any] = dict(initial)
    for event in graph.stream(initial, config=config, stream_mode="updates"):
        node = list(event.keys())[0]
        update = event[node]
        final_state.update(update)
        yield {"type": "node", "node": node, "answer": final_state.get("answer", "")}
    yield {
        "type": "final",
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "contexts": final_state.get("contexts", []),
        "retries": final_state.get("retries", 0),
        "score": final_state.get("score", 0),
    }
