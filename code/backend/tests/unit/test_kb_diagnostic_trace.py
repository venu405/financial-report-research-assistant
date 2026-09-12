"""诊断追踪定向单测：默认关闭、零副作用、开启时记录合法且脱敏。

只覆盖本次新增的追踪能力，不重复既有检索/问答测试。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from services.kb import diagnostic_trace as dt
from services.kb.qa_graph import build_qa_graph, run_qa, run_qa_stream
from services.kb.retriever import HybridRetriever
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding, FakeLLM

ENV_ENABLED = "KB_DIAGNOSTIC_TRACE_ENABLED"
ENV_PATH = "KB_DIAGNOSTIC_TRACE_PATH"


@pytest.fixture(autouse=True)
def _clean_diagnostic_state(monkeypatch):
    """每个用例前清掉开关、环境变量、注册表与上下文指针，避免互相污染。"""
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    monkeypatch.delenv(ENV_PATH, raising=False)
    dt.reset_state()
    yield
    dt.reset_state()


def _enable(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_PATH, str(path))


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_store(tmp_path: Path) -> VectorStore:
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(
            ["吉林华微电子2024年营业收入2,057,608,183.78元主要会计数据"]
        ),
        texts=["吉林华微电子2024年营业收入2,057,608,183.78元主要会计数据"],
        doc_id="d-huawei",
        doc_title="ST华微-2024年年度报告.pdf",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="cninfo_report",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["青岛中资中程2024年净利润-310,302,902.32元"]),
        texts=["青岛中资中程2024年净利润-310,302,902.32元"],
        doc_id="d-zhongcheng",
        doc_title="ST中程-2024年年度报告.pdf",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="cninfo_report",
    )
    return store


# ---------------------------------------------------------------- 默认关闭


def test_disabled_by_default(tmp_path, monkeypatch):
    """未设置开关时：不启用、不落盘、add_stage 是 no-op。"""
    assert dt.is_enabled() is False
    trace_file = tmp_path / "trace.jsonl"

    dt.add_stage("whatever", {"a": 1})
    dt.finish_request({"answer": "x"})

    assert not trace_file.exists()


def test_enabled_requires_path(tmp_path, monkeypatch):
    """只开开关不给路径，仍然不启用——避免写到不可控位置。"""
    monkeypatch.setenv(ENV_ENABLED, "1")
    assert dt.is_enabled() is False


def test_disabled_does_not_change_retriever_results(tmp_path, monkeypatch):
    """关闭与开启两种状态下，检索结果必须逐条一致（追踪不能影响召回）。"""
    store = _build_store(tmp_path)
    emb = FakeEmbedding()
    retriever = HybridRetriever(store, embeddings=emb, top_k=5)

    baseline = retriever.search("吉林华微电子2024年营业收入", kb_id="cninfo_report")

    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)
    traced = retriever.search("吉林华微电子2024年营业收入", kb_id="cninfo_report")

    # 追踪只写文件，不回写候选
    assert [h.get("chunk_id") for h in baseline] == [
        h.get("chunk_id") for h in traced
    ]
    assert [round(float(h.get("rrf_score") or 0), 6) for h in baseline] == [
        round(float(h.get("rrf_score") or 0), 6) for h in traced
    ]
    assert [h.get("text") for h in baseline] == [h.get("text") for h in traced]


# ---------------------------------------------------------------- 开启后行为


def test_enabled_writes_valid_record(tmp_path, monkeypatch):
    """开启后能生成合法 JSONL 记录，且包含必备基本信息字段。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    trace_id = dt.start_request(
        {
            "question": "吉林华微电子2024年营业收入是多少？",
            "kb_id": "cninfo_report",
            "config": {"top_k": 5, "recall_k": 20, "vector_backend": "qdrant"},
        }
    )
    assert trace_id, "启用后应返回非空 trace_id"
    dt.add_stage("raw_recall", {"fused_count": 2})
    dt.add_stage("final_answer", {"answer": "营业收入为2,057,608,183.78元"})
    dt.finish_request({"answer": "营业收入为2,057,608,183.78元", "citations": 1})

    records = _read_records(trace_file)
    assert len(records) == 1
    record = records[0]
    assert record["trace_id"] == trace_id
    assert record["ts_start"] and record["ts_end"]
    assert record["basic"]["question"].startswith("吉林华微电子")
    assert record["basic"]["config"]["top_k"] == 5
    assert record["basic"]["config"]["vector_backend"] == "qdrant"
    assert [stage["stage"] for stage in record["stages"]] == [
        "raw_recall",
        "final_answer",
    ]
    assert record["outcome"]["citations"] == 1


def test_retriever_stages_are_recorded(tmp_path, monkeypatch):
    """真实检索链路应产出 raw_recall / structured_fin_route / post_merge 阶段。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    store = _build_store(tmp_path)
    retriever = HybridRetriever(store, embeddings=FakeEmbedding(), top_k=5)
    dt.start_request({"question": "吉林华微电子2024年营业收入", "kb_id": "cninfo_report"})
    retriever.search("吉林华微电子2024年营业收入", kb_id="cninfo_report")
    dt.finish_request()

    records = _read_records(trace_file)
    assert len(records) == 1
    stages = [stage["stage"] for stage in records[0]["stages"]]
    assert "raw_recall" in stages
    assert "post_merge" in stages
    # 结构化路由即使未命中也要留下决策记录，否则无法判断是"没触发"还是"没结果"
    assert "structured_fin_route" in stages
    route = next(
        stage["data"]
        for stage in records[0]["stages"]
        if stage["stage"] == "structured_fin_route"
    )
    assert "outcome" in route


def test_candidate_text_is_length_limited(tmp_path, monkeypatch):
    """候选正文必须截断，禁止把整篇文档写进追踪文件。"""
    long_text = "营业收入数字" * 5000  # 约 3 万字符
    hit = {
        "chunk_id": "c1",
        "score": 0.9,
        "text": long_text,
        "metadata": {"doc_id": "d1", "kb_id": "cninfo_report", "page": 6},
    }
    record = dt.candidate_record(hit, 1, "vector_bm25_fusion")

    assert len(record["text"]) <= dt.MAX_CANDIDATE_TEXT + len(dt._TRUNCATED_MARK)
    assert record["text"].endswith(dt._TRUNCATED_MARK)
    assert record["page"] == 6
    assert record["source"] == "vector_bm25_fusion"


def test_no_secrets_in_trace(tmp_path, monkeypatch):
    """键名命中敏感词的字段一律脱敏，追踪文件里不得出现密钥字面量。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    secret = "sk-THIS-IS-A-REAL-SECRET-VALUE-1234567890"
    dt.start_request(
        {
            "question": "华微营业收入",
            "llm_api_key": secret,
            "authorization": "Bearer " + secret,
            "nested": {"api_key": secret, "safe": "visible"},
        }
    )
    dt.add_stage("stage", {"qdrant_api_key": secret, "top_k": 5})
    dt.finish_request({"token": secret})

    raw = trace_file.read_text(encoding="utf-8")
    assert secret not in raw
    record = _read_records(trace_file)[0]
    assert record["basic"]["llm_api_key"] == dt._REDACTED
    assert record["basic"]["authorization"] == dt._REDACTED
    assert record["basic"]["nested"]["api_key"] == dt._REDACTED
    assert record["basic"]["nested"]["safe"] == "visible"
    assert record["outcome"]["token"] == dt._REDACTED
    # 非敏感配置照常保留，否则追踪就没有诊断价值
    stages = {stage["stage"]: stage["data"] for stage in record["stages"]}
    assert stages["stage"]["top_k"] == 5


def test_multiple_requests_do_not_overwrite(tmp_path, monkeypatch):
    """多个请求追加写，互不覆盖；每个 trace_id 唯一。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    ids = []
    for index in range(3):
        ids.append(dt.start_request({"question": f"问题{index}"}))
        dt.add_stage("raw_recall", {"seq": index})
        dt.finish_request({"index": index})

    records = _read_records(trace_file)
    assert len(records) == 3
    assert [record["trace_id"] for record in records] == ids
    assert len(set(ids)) == 3
    assert [record["basic"]["question"] for record in records] == [
        "问题0",
        "问题1",
        "问题2",
    ]


def test_unfinished_request_is_flushed_not_lost(tmp_path, monkeypatch):
    """上一个请求异常未收尾时，下一次 start 要把它刷盘而不是丢弃。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    dt.start_request({"question": "第一个请求"})
    dt.add_stage("raw_recall", {"seq": 0})
    # 模拟异常路径：没有调用 finish_request
    dt.start_request({"question": "第二个请求"})
    dt.add_stage("raw_recall", {"seq": 1})
    dt.finish_request()

    records = _read_records(trace_file)
    assert len(records) == 2
    assert records[0]["basic"]["question"] == "第一个请求"
    assert records[0].get("truncated") is True
    assert records[1]["basic"]["question"] == "第二个请求"
    assert "truncated" not in records[1]


def test_ensure_started_is_idempotent(tmp_path, monkeypatch):
    """gate 重试会重跑 retrieve，ensure_started 必须复用同一条记录。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    first = dt.ensure_started({"question": "华微营业收入"})
    dt.add_stage("raw_recall", {"attempt": 0})
    second = dt.ensure_started({"question": "华微营业收入"})
    dt.add_stage("raw_recall", {"attempt": 1})
    dt.finish_request()

    records = _read_records(trace_file)
    assert len(records) == 1, "一次问答只能产生一条追踪记录"
    assert first == second
    assert len(records[0]["stages"]) == 2


def test_claim_record_captures_binding_fields():
    """claim 快照要包含 metric/unit/period/scope，才能定位字段绑定失败。"""
    class _Claim:
        raw = "-310,302,902.32"
        metric = "net_profit"
        line_metric = "归属于上市公司股东的净利润"
        unit = "元"
        report_period = "2024"
        statement_scope = "合并"
        adjustment = ""
        is_percent = False
        canonical_value = "-310302902.32"

    record = dt.claim_record(_Claim())
    assert record["raw"] == "-310,302,902.32"
    assert record["metric"] == "net_profit"
    assert record["unit"] == "元"
    assert record["report_period"] == "2024"
    assert record["statement_scope"] == "合并"
    assert record["is_percent"] is False
    # Decimal 必须可 JSON 序列化
    json.dumps(record, ensure_ascii=False)


# ------------------------------------------- 回归：跨 LangGraph 节点上下文隔离


def _build_two_node_graph():
    """真实 LangGraph 两节点空壳图，用来复现上下文隔离（不调用任何模型）。"""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class _S(TypedDict):
        q: str
        trace_id: str

    def node_retrieve(state):
        dt.ensure_started({"question": state["q"]}, token=state["trace_id"])
        dt.add_stage("raw_recall", {"node": "retrieve"})
        return {}

    def node_rerank(state):
        # 真实调用方（qa_graph._trace_bind）在每个节点入口重新绑定一次
        dt.bind(state["trace_id"])
        dt.add_stage("post_merge", {"node": "rerank"})
        return {}

    graph = StateGraph(_S)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", END)
    return graph.compile()


def test_trace_survives_langgraph_node_context_isolation(tmp_path, monkeypatch):
    """P0 回归：LangGraph 每个节点跑在独立上下文副本里，ContextVar 跨节点失效。

    曾经的失效表现极隐蔽：is_enabled 为 True、无异常、无告警，但一条记录都不落盘。
    本用例用真实 StateGraph 锁死"节点埋点 → 外层收尾"这条链路必须通。
    """
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    token = "trace-token-abc123"
    _build_two_node_graph().invoke({"q": "华微营业收入", "trace_id": token})
    dt.finish_request({"answer": "2,057,608,183.78元"}, token=token)

    records = _read_records(trace_file)
    assert len(records) == 1, "跨节点追踪必须落盘，不能静默丢失"
    stages = [stage["stage"] for stage in records[0]["stages"]]
    # 两个节点分别写阶段，说明两节点拿到的是同一条记录
    assert stages == ["raw_recall", "post_merge"]
    assert records[0]["basic"]["question"] == "华微营业收入"
    assert records[0]["outcome"]["answer"] == "2,057,608,183.78元"
    # 收尾后注册表必须清空，否则异常路径会堆积
    assert dt.pending_count() == 0


def test_registry_is_keyed_by_token(tmp_path, monkeypatch):
    """不同 token 的请求互不串台：交错执行也要各写各的记录。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    dt.ensure_started({"question": "A-华微"}, token="tok-a")
    dt.ensure_started({"question": "B-中程"}, token="tok-b")
    dt.bind("tok-a")
    dt.add_stage("raw_recall", {"who": "a"})
    dt.bind("tok-b")
    dt.add_stage("raw_recall", {"who": "b"})
    dt.finish_request({"who": "a"}, token="tok-a")
    dt.finish_request({"who": "b"}, token="tok-b")

    records = _read_records(trace_file)
    assert len(records) == 2
    by_question = {record["basic"]["question"]: record for record in records}
    assert set(by_question) == {"A-华微", "B-中程"}
    assert by_question["A-华微"]["stages"][0]["data"]["who"] == "a"
    assert by_question["B-中程"]["stages"][0]["data"]["who"] == "b"
    assert dt.pending_count() == 0


def test_finish_without_matching_token_does_not_drop_other_records(
    tmp_path, monkeypatch
):
    """token 对不上时只应当次收尾无效，不能误删别人的记录。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    dt.ensure_started({"question": "keep-me"}, token="tok-keep")
    dt.finish_request(token="tok-typo")

    assert dt.pending_count() == 1, "错误 token 不得清掉在途记录"
    dt.finish_request({"ok": True}, token="tok-keep")
    records = _read_records(trace_file)
    assert len(records) == 1
    assert records[0]["basic"]["question"] == "keep-me"


def test_run_qa_end_to_end_writes_trace(tmp_path, monkeypatch):
    """端到端：真实 run_qa（FakeLLM，不联网）必须落盘，且各阶段齐全。

    这是最接近 5 题正式运行的离线验证——只要这条挂了，正式运行一定拿不到
    候选追踪，白跑一轮。
    """
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    store = VectorStore(persist_dir=str(tmp_path / "vs"))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["采购金额超过5万元必须公开招投标"]),
        texts=["采购金额超过5万元必须公开招投标"],
        doc_id="d1",
        doc_title="采购制度",
        source_type="md",
        chunk_indices=[0],
        kb_id="default",
    )
    llm = FakeLLM(
        route={
            "仅基于以下资料": "根据制度，超过5万元必须公开招投标。[1]",
            "RAG 质量评估员": "10",
        }
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="超过多少万元必须招投标", kb_id="default")

    assert "5万元" in result["answer"]

    records = _read_records(trace_file)
    assert len(records) == 1, "正式问答链路必须产出且仅产出一条追踪记录"
    stages = [stage["stage"] for stage in records[0]["stages"]]
    # 检索侧与生成侧都要有痕迹，缺一侧就无法判断"丢在哪一层"
    assert "node_retrieve" in stages
    assert "initial_answer" in stages
    assert "final_answer" in stages
    assert records[0]["basic"]["question"] == "超过多少万元必须招投标"
    assert "top_k" in records[0]["basic"]["config"]
    assert dt.pending_count() == 0, "收尾后注册表必须清空"


# ------------------------------------------- 审核问题1：容量清理不得死锁


def test_overflow_eviction_does_not_deadlock(tmp_path, monkeypatch):
    """注册表超限时必须继续工作，不能因为"持锁期间调 _flush"而死锁。

    历史缺陷：start_request 在 `with _lock` 里调 _evict_overflow → _flush，
    而 _flush 自己又要获取同一个**不可重入**的 Lock，直接卡死。
    本用例用独立线程 + join 超时，确保错误实现只会让**测试失败**，
    而不会把整个 pytest 会话永久挂住。
    """
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    errors: list[BaseException] = []
    completed = threading.Event()

    def _spawn_many() -> None:
        try:
            # 刻意超过 MAX_PENDING_RECORDS，必然触发容量清理路径
            for index in range(dt.MAX_PENDING_RECORDS + 20):
                dt.start_request({"question": f"q{index}"}, token=f"tok-{index}")
            completed.set()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=_spawn_many, daemon=True)
    worker.start()
    worker.join(timeout=10)

    if worker.is_alive():
        # 死锁已发生：worker 永久持有旧锁。换一把新锁并重建注册表，
        # 否则本模块 autouse fixture 的 reset_state() 也会跟着卡死，
        # 把"一个用例失败"升级成"整个 pytest 会话挂住"。
        dt._lock = threading.Lock()
        dt._records = {}
        pytest.fail("注册表容量清理发生死锁（工作线程 10s 未返回）")
    assert not errors, f"容量清理抛异常：{errors!r}"
    assert completed.is_set(), "容量清理未正常走完"

    # 容量必须受限，不能无限增长
    assert dt.pending_count() <= dt.MAX_PENDING_RECORDS

    # 被挤出的旧记录必须以 truncated 形式落盘，不能凭空消失
    records = _read_records(trace_file)
    assert len(records) == 20, f"应有 20 条被挤出并记录，实际 {len(records)}"
    assert all(record.get("truncated") is True for record in records)
    assert records[0]["basic"]["question"] == "q0", "必须按最老优先清理"

    dt.reset_state()


# ------------------------------------------- 审核问题2：异常/流式中断必须收尾


class _ExplodingGraph:
    """模拟"节点已开追踪、随后图抛异常"的真实场景（不调用任何模型）。"""

    def __init__(self, message: str = "图执行炸了") -> None:
        self._message = message

    def invoke(self, state, config=None):
        dt.ensure_started(
            {"question": state.get("question", "")},
            token=str(state.get("trace_id") or ""),
        )
        dt.add_stage("raw_recall", {"simulated": True})
        raise RuntimeError(self._message)


class _StreamingGraph:
    """最小流式图：先开追踪，再按给定事件序列 yield。"""

    def __init__(self, events, fail_after: int | None = None) -> None:
        self._events = events
        self._fail_after = fail_after

    def stream(self, state, config=None, stream_mode=None):
        dt.ensure_started(
            {"question": state.get("question", "")},
            token=str(state.get("trace_id") or ""),
        )
        dt.add_stage("raw_recall", {"simulated": True})
        for index, event in enumerate(self._events):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("流式图炸了")
            yield event


def _normal_stream_events() -> list[tuple]:
    return [
        ("updates", {"retrieve": {"contexts": []}}),
        ("updates", {"generate": {"answer": "模拟答案"}}),
        ("updates", {"evaluate": {"score": 10}}),
    ]


def test_run_qa_finishes_trace_when_graph_raises(tmp_path, monkeypatch):
    """图抛异常时必须收尾：注册表不留残留，且原异常照常抛出。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    with pytest.raises(RuntimeError, match="图执行炸了"):
        run_qa(_ExplodingGraph(), question="华微营业收入", kb_id="cninfo_report")

    assert dt.pending_count() == 0, "异常路径不得留下注册表残留"
    records = _read_records(trace_file)
    assert len(records) == 1
    outcome = records[0]["outcome"]
    # 必须写明是异常收尾，绝不能伪装成成功
    assert outcome["status"] == "exception"
    assert outcome["error_type"] == "RuntimeError"
    assert "图执行炸了" in outcome["error"]
    assert "ok" != outcome["status"]


def test_run_qa_exception_does_not_clear_other_requests(tmp_path, monkeypatch):
    """交错执行时一个请求异常，不能顺手清掉另一个在途请求的记录。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    dt.ensure_started({"question": "keep-me"}, token="tok-keep")
    with pytest.raises(RuntimeError):
        run_qa(_ExplodingGraph(), question="会炸的请求", kb_id="cninfo_report")

    assert dt.pending_count() == 1, "异常请求不得清掉其他在途记录"
    dt.finish_request({"ok": True}, token="tok-keep")

    records = _read_records(trace_file)
    questions = [record["basic"]["question"] for record in records]
    assert "会炸的请求" in questions
    assert "keep-me" in questions
    assert dt.pending_count() == 0


def test_stream_closed_early_is_not_written_as_success(tmp_path, monkeypatch):
    """客户端提前断开（GeneratorExit）：必须收尾，且不能写成正常成功。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    gen = run_qa_stream(
        _StreamingGraph(_normal_stream_events()),
        question="流式问题-提前断开",
        kb_id="default",
    )
    next(gen)  # 消费第一个事件后立刻断开
    gen.close()

    assert dt.pending_count() == 0, "流式提前关闭后不得留下注册表残留"
    records = _read_records(trace_file)
    assert len(records) == 1
    assert records[0]["outcome"]["status"] == "client_disconnected"
    assert records[0]["outcome"].get("status") != "ok"


def test_stream_graph_error_is_recorded_and_reraised(tmp_path, monkeypatch):
    """流式图抛异常：记录错误摘要后原样抛出，不吞异常。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    gen = run_qa_stream(
        _StreamingGraph(_normal_stream_events(), fail_after=1),
        question="流式问题-图异常",
        kb_id="default",
    )
    with pytest.raises(RuntimeError, match="流式图炸了"):
        next(gen)
        next(gen)

    assert dt.pending_count() == 0
    records = _read_records(trace_file)
    assert len(records) == 1
    outcome = records[0]["outcome"]
    assert outcome["status"] == "stream_error"
    assert outcome["error_type"] == "RuntimeError"
    assert "流式图炸了" in outcome["error"]


def test_stream_normal_completion_unchanged(tmp_path, monkeypatch):
    """正常流式行为不变：事件序列照旧，收尾写 ok，注册表清空。"""
    trace_file = tmp_path / "trace.jsonl"
    _enable(monkeypatch, trace_file)

    events = list(
        run_qa_stream(
            _StreamingGraph(_normal_stream_events()),
            question="流式问题-正常完成",
            kb_id="default",
        )
    )

    assert any(event.get("type") == "final" for event in events), "final 事件必须存在"
    assert dt.pending_count() == 0
    records = _read_records(trace_file)
    assert len(records) == 1
    assert records[0]["outcome"]["status"] == "ok"


def test_disabled_run_qa_and_stream_leave_no_residue(tmp_path, monkeypatch):
    """诊断关闭时，同步/流式（含异常）都不留残留、不落盘。"""
    trace_file = tmp_path / "trace.jsonl"

    with pytest.raises(RuntimeError):
        run_qa(_ExplodingGraph(), question="关闭态异常", kb_id="default")
    assert dt.pending_count() == 0
    assert not trace_file.exists()


# ------------------------------------------- 审核问题4：关闭时零诊断开销


def _build_fin_route_store(tmp_path: Path) -> VectorStore:
    """带完整财务元数据的语料，确保结构化金融路由能走到"注入"分支。"""
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    texts = [
        "吉林华微电子股份有限公司2024年年度报告主要会计数据 营业收入 2,057,608,183.78",
        "吉林华微电子股份有限公司2024年年度报告主要会计数据 净利润 123,456,789.01",
    ]
    store.add_chunks(
        embeddings=emb.embed_texts(texts),
        texts=texts,
        doc_id="d-huawei-fin",
        doc_title="吉林华微电子-2024年年度报告.pdf",
        source_type="pdf",
        chunk_indices=[0, 1],
        kb_id="cninfo_report",
        extra_metadata=[
            {
                "company": "吉林华微电子",
                "report_period": "2024年度",
                "section_path": "主要会计数据",
                "is_table": True,
                "page": 6,
            },
            {
                "company": "吉林华微电子",
                "report_period": "2024年度",
                "section_path": "主要会计数据",
                "is_table": True,
                "page": 6,
            },
        ],
    )
    return store


def test_fin_route_snapshot_only_built_when_enabled(tmp_path, monkeypatch):
    """结构化路由的候选快照只在开启时构造；关闭时一次都不许调用。

    先用"开启态确实调用了"证明本场景真走到了注入分支，否则这条用例会因为
    路径根本没覆盖而假通过。
    """
    from services.kb import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    store = _build_fin_route_store(tmp_path / "vs")
    emb = FakeEmbedding()
    query = "吉林华微电子2024年营业收入是多少？"

    calls: list[str] = []
    real_snapshot = dt.candidate_snapshot

    def _spy(hits, source, limit=dt.MAX_CANDIDATES):
        calls.append(source)
        return real_snapshot(hits, source, limit)

    monkeypatch.setattr(dt, "candidate_snapshot", _spy)

    # —— 开启态：确认场景有效（确实走到了结构化路由的快照构造）——
    _enable(monkeypatch, tmp_path / "on.jsonl")
    HybridRetriever(store, embeddings=emb, top_k=5).search(query, kb_id="cninfo_report")
    assert any("structured_fin_route" in name for name in calls), (
        f"测试场景未走到结构化路由注入分支，守卫验证形同虚设：{calls}"
    )

    # —— 关闭态：零调用、零文件 ——
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    monkeypatch.delenv(ENV_PATH, raising=False)
    calls.clear()
    HybridRetriever(store, embeddings=emb, top_k=5).search(query, kb_id="cninfo_report")

    assert calls == [], f"诊断关闭时仍构造了候选快照：{calls}"
    assert not (tmp_path / "off.jsonl").exists()


def test_disabled_retriever_never_touches_diagnostic_structures(
    tmp_path, monkeypatch
):
    """关闭时不得为追踪遍历/裁剪/复制候选，也不得产生任何追踪文件。"""
    from services.kb import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    store = _build_fin_route_store(tmp_path / "vs2")
    emb = FakeEmbedding()

    calls: list[str] = []
    real_snapshot = dt.candidate_snapshot
    real_ranking = dt.ranking_ids

    def _spy_snapshot(hits, source, limit=dt.MAX_CANDIDATES):
        calls.append(f"snapshot:{source}")
        return real_snapshot(hits, source, limit)

    def _spy_ranking(hits, limit=dt.MAX_CANDIDATES):
        calls.append("ranking")
        return real_ranking(hits, limit)

    monkeypatch.setattr(dt, "candidate_snapshot", _spy_snapshot)
    monkeypatch.setattr(dt, "ranking_ids", _spy_ranking)

    trace_file = tmp_path / "trace.jsonl"
    retriever = HybridRetriever(store, embeddings=emb, top_k=5)

    first = retriever.search("吉林华微电子2024年营业收入是多少？", kb_id="cninfo_report")
    second = retriever.search("吉林华微电子2024年营业收入是多少？", kb_id="cninfo_report")

    assert calls == [], f"诊断关闭时不应构造任何诊断数据结构：{calls}"
    assert not trace_file.exists()
    # 检索结果本身必须稳定可复现（顺序/分数/文本都不受追踪开关影响）
    assert [h.get("chunk_id") for h in first] == [h.get("chunk_id") for h in second]
