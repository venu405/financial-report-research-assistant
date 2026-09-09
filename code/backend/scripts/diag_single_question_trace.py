#!/usr/bin/env python3
"""单题全流程诊断：在进程内跑真实 QA 图，回放每个节点的中间态。

用途：报告里只保留最终答案，无法判断「拒答」是检索没召回、生成写错数，
还是核验误杀。本脚本用 LangGraph checkpointer 回放状态历史，逐个节点打印
answer / verification_reasons / contexts（来源+页码），一次定位断链层次。

不启动服务、不改代码；只在需要时调用一次 LLM（DeepSeek），题数由 --case-id
或 --question 决定，默认 1 题。

用法：
    .venv/Scripts/python.exe scripts/diag_single_question_trace.py \
        --case-id real_furun_2024_adjusted_revenue_synonym
    .venv/Scripts/python.exe scripts/diag_single_question_trace.py \
        --question "ST富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？"
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))


def import_dotenv(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--kb-id", default="cninfo_report")
    ap.add_argument("--max-passage-chars", type=int, default=900)
    args = ap.parse_args()

    question = args.question
    expected_keyword = None

    # 密钥只从 .env 加载，且不被打印；命令行已显式设置的环境变量优先。
    import_dotenv(BACKEND_DIR / ".env")

    if args.case_id:
        import yaml

        data = yaml.safe_load(
            (BACKEND_DIR / "testsets" / "rag_real_quality_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        for case in data.get("tests", []):
            if case.get("id") == args.case_id:
                question = case["question"]
                args.kb_id = case.get("kb_id", args.kb_id)
                expected_keyword = case.get("expect_keyword") or case.get(
                    "expect_keywords_all"
                )
                break
        if question is None:
            print(f"未找到 case_id={args.case_id}")
            return 1
    if not question:
        print("需要 --case-id 或 --question")
        return 1

    print(f"问题: {question}")
    print(f"kb_id: {args.kb_id}  期望关键词: {expected_keyword}")

    from config import Configuration

    cfg = Configuration.from_env()
    print(
        f"后端={cfg.kb_vector_backend} collection={cfg.kb_collection} "
        f"top_k={cfg.kb_top_k} max_hits_per_doc={getattr(cfg, 'kb_max_hits_per_doc', 3)} "
        f"route={os.getenv('KB_STRUCTURED_FIN_ROUTE')} rerank={getattr(cfg, 'kb_rerank_mode', None)}"
    )

    from openai import OpenAI

    from services.kb import qa_graph
    from services.kb.embeddings import EmbeddingClient

    # 与 main.py 保持一致：向量库构造走 main 的私有工厂，避免两处逻辑漂移
    from main import _build_kb_vector_store  # noqa: E402

    embeddings = EmbeddingClient(base_url=cfg.kb_ollama_host, model=cfg.kb_embedding_model)
    store = _build_kb_vector_store(cfg)
    llm = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url or None,
        timeout=float(os.getenv("LLM_TIMEOUT", "60") or 60),
    )

    from services.kb.reranker import build_reranker

    reranker = build_reranker(
        getattr(cfg, "kb_rerank_mode", "llm"),
        llm=llm,
        model=cfg.llm_model_id or "deepseek-chat",
        crossencoder_model=getattr(cfg, "kb_rerank_model", "BAAI/bge-reranker-base"),
    )

    ckpt_path = BACKEND_DIR / ".rag_eval" / "diag_trace_checkpoints.db"
    conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    from langgraph.checkpoint.sqlite import SqliteSaver

    graph = qa_graph.build_qa_graph(
        llm=llm,
        embeddings=embeddings,
        vector_store=store,
        top_k=cfg.kb_top_k,
        checkpointer=SqliteSaver(conn),
        model=cfg.llm_model_id or "deepseek-chat",
        reasoning_effort=cfg.llm_reasoning_effort,
        min_score=getattr(cfg, "kb_min_similarity", 0.0),
        reranker=reranker,
        recall_k=getattr(cfg, "kb_recall_k", max(cfg.kb_top_k * 4, cfg.kb_top_k)),
        max_candidates_per_doc=getattr(cfg, "kb_max_hits_per_doc", 3),
        neighbor_expansion=getattr(cfg, "kb_neighbor_expansion", 0),
    )

    import uuid

    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    result = qa_graph.run_qa(
        graph,
        question=question,
        kb_id=args.kb_id,
        thread_id=thread_id,
    )

    print("\n=== 最终答案 ===")
    print(result.get("answer", ""))
    print(f"citations={len(result.get('citations') or [])} retries={result.get('retries')} "
          f"score={result.get('score')}")
    if expected_keyword:
        keys = expected_keyword if isinstance(expected_keyword, list) else [expected_keyword]
        for k in keys:
            print(f"  关键词 {k!r} 出现在答案中: {str(k) in result.get('answer', '')}")

    print("\n=== 检索到的上下文（来源/页码/是否表格）===")
    for i, ctx in enumerate(result.get("contexts") or []):
        meta = ctx.get("metadata") if isinstance(ctx, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        src = str(meta.get("source_path") or meta.get("source") or meta.get("doc_title") or "")
        print(f"  [{i+1}] page={meta.get('page')} is_table={meta.get('is_table')} "
              f"score={ctx.get('score') if isinstance(ctx, dict) else ''} "
              f"src=...{src[-40:]}")

    print("\n=== 状态历史回放（逐节点）===")
    snapshots = []
    try:
        for snapshot in graph.get_state_history(config):
            snapshots.append(snapshot)
            node = (snapshot.metadata or {}).get("writes")
            values = snapshot.values or {}
            ans = (values.get("answer") or "").strip()
            reasons = values.get("verification_reasons") or []
            print("-" * 70)
            print(f"node_writes={node} retries={values.get('retries')} "
                  f"low_quality={values.get('low_quality')} score={values.get('score')}")
            if ans:
                print(f"  answer: {ans[:400]}")
            if reasons:
                print(f"  verification_reasons={reasons}")
    except Exception as exc:  # noqa: BLE001
        print(f"状态历史回放失败：{exc}")

    # 落盘：保留"草稿答案 + 当次检索上下文"，供离线复现核验拒绝原因，
    # 后续排查不再重复调用 LLM。
    dump_path = BACKEND_DIR / ".rag_eval" / f"diag_trace_{args.case_id or 'adhoc'}.json"
    best = None
    for snapshot in snapshots:
        values = snapshot.values or {}
        ans = (values.get("answer") or "").strip()
        if ans and "无法可靠对应" not in ans and "未通过可靠性核验" not in ans:
            best = values
            break
    payload = {
        "case_id": args.case_id,
        "question": question,
        "kb_id": args.kb_id,
        "draft_answer": (best or {}).get("answer", ""),
        "final_answer": result.get("answer", ""),
        "verification_reasons": (best or {}).get("verification_reasons") or [],
        "contexts": (best or {}).get("contexts") or result.get("contexts") or [],
        "passages": (best or {}).get("passages") or [],
        "passage_records": (best or {}).get("passage_records") or [],
    }
    try:
        dump_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        print(f"\n已落盘：{dump_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"落盘失败：{exc}")

    print("\n=== 喂给模型的 passage 片段（截断显示）===")
    for i, p in enumerate(result.get("passages") or []):
        print(f"--- passage[{i}] ---")
        print(str(p)[: args.max_passage_chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
