"""evaluate_quality.py 评测脚本的纯函数与离线行为测试（不联网）。"""
from __future__ import annotations

import importlib.util
import json
from importlib.util import spec_from_file_location
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_module(name: str):
    # scripts 不是包：按文件路径加载，避免修改共享 conftest 的 sys.path
    spec = spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eq = _load_module("evaluate_quality")


def test_numeric_keyword_semantics_support_decline_and_exact_units_only():
    assert eq.semantic_keyword_matches("-22.55", "营业收入同比减少22.55%") is True
    assert eq.semantic_keyword_matches("-22.55", "营业收入同比增长22.55%") is False
    assert eq.semantic_keyword_matches("22.55", "数值为122.55") is False
    assert eq.semantic_keyword_matches("1,200,000,000元", "12亿元") is True
    assert eq.semantic_keyword_matches("20,774,218,376.59元", "2,077,421.84万元") is False
    assert eq.semantic_keyword_matches("1,403,456,789.12元", "约14.03亿元") is False
    assert eq.semantic_keyword_matches("３９，３８９，０５３．４８元", "金额为39,389,053.48元") is True


def test_numeric_keyword_semantics_allow_one_way_rounded_display_units():
    assert eq.semantic_keyword_matches(
        "1135.41亿元", "113,541,282,968.02元"
    ) is True
    assert eq.semantic_keyword_matches(
        "113,541,282,968.02元", "1135.41亿元"
    ) is False
    assert eq.semantic_keyword_matches("1135.41亿元", "1135.42亿元") is False
    assert eq.semantic_keyword_matches(
        "1135.41亿元", "约113,541,282,968.02元"
    ) is False

    # ROUND_HALF_UP 边界：小于半个最小展示单位通过，正好半个单位进位后失败。
    assert eq.semantic_keyword_matches("1.00亿元", "100,499,999元") is True
    assert eq.semantic_keyword_matches("1.00亿元", "100,500,000元") is False


def test_performance_commitment_negation_allows_only_same_clear_polarity():
    expected = "不构成对投资者的业绩承诺"
    assert eq.semantic_keyword_matches(
        expected, "该计划不构成业绩承诺，请投资者注意风险"
    ) is True
    assert eq.semantic_keyword_matches(expected, "该计划构成业绩承诺") is False
    assert eq.semantic_keyword_matches(expected, "是否构成业绩承诺？") is False
    assert eq.semantic_keyword_matches(expected, "无法判断是否构成业绩承诺") is False
    assert eq.semantic_keyword_matches(expected, "投资者应关注业绩承诺相关风险") is False
    assert eq.semantic_keyword_matches(
        expected, "公司不构成对投资者的业绩承诺"
    ) is True


def test_percent_tolerance_only_when_answer_is_more_explicit_than_keyword():
    """百分号口径只在一个方向放宽：答案比无单位关键词更明确。

    背景：占比类问题的题集写 expect_keyword "22.32"，而标准答案与模型答案都写
    "22.32%"。百分号的 canonical 是 ÷100，旧逻辑判两者不等，连题集自带的标准
    答案都过不了自己的关键词检查。这里补的是"信息只增不减"方向的容忍。
    """
    # 正向：关键词无单位、答案补了 %，展示数字相同 -> 通过
    assert eq.semantic_keyword_matches("22.32", "研发投入占营业收入的比例为22.32%。") is True
    assert eq.semantic_keyword_matches("22.32", "比例为22.32％") is True
    # 反向：关键词声明了 %，答案省略 % -> 仍是拒绝（单位信息丢失）
    assert eq.semantic_keyword_matches("22.32%", "比例为22.32") is False
    # 数字不同，任何方向都不放宽
    assert eq.semantic_keyword_matches("22.32", "比例为22.35%") is False
    assert eq.semantic_keyword_matches("22.32", "比例为2.232%") is False
    # 下降词守卫不因本次放宽而失效
    assert eq.semantic_keyword_matches("-22.55", "营业收入同比增长22.55%") is False
    assert eq.semantic_keyword_matches("-22.55", "营业收入同比减少22.55%") is True
    # 非百分号单位仍走原有 canonical 规则，不受本次改动影响
    assert eq.semantic_keyword_matches("1,200,000,000元", "12亿元") is True
    assert eq.semantic_keyword_matches("20,774,218,376.59元", "2,077,421.84万元") is False


def test_semantic_keyword_matches_comparison_variants_without_general_text_fuzziness():
    """Comparison wording may vary, but ordinary text and numbers stay exact."""

    subject = "青岚织星"
    assert eq.semantic_keyword_matches(
        f"{subject}的呼吸系统用药销售额减少",
        f"{subject}的呼吸系统用药销售额较上年同期减少",
    ) is True
    assert eq.semantic_keyword_matches(
        f"{subject}营业收入较上年同期下降42.89%",
        f"{subject}营业收入同比下降42.89%",
    ) is True

    assert eq.semantic_keyword_matches(
        f"{subject}营业收入较上年同期下降42.89%",
        f"{subject}营业收入同比增长42.89%",
    ) is False
    assert eq.semantic_keyword_matches(
        f"{subject}的呼吸系统用药销售额减少",
        f"{subject}的呼吸系统用药销售额同比增加",
    ) is False
    assert eq.semantic_keyword_matches(
        f"{subject}营业收入较上年同期下降42.89%",
        f"{subject}营业收入同比下降42.98%",
    ) is False

    # Comparison-term normalization must not become fuzzy matching for
    # unrelated ordinary wording.
    assert eq.semantic_keyword_matches(
        f"{subject}的办公地址",
        f"{subject}的办公地点",
    ) is False


def test_keywords_all_uses_semantic_numeric_match_but_preserves_legacy_fields(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {
                "id": "decline",
                "question": "东时2024年收入同比变化？",
                "expect_keywords_all": ["-22.55"],
            }
        ],
    )
    monkeypatch.setattr(
        eq,
        "ask",
        lambda *args, **kwargs: _ask_response(answer="营业收入同比减少22.55% [1]"),
    )

    report = eq.run_rag(testset, "http://127.0.0.1:8000", None)
    assert report["passed"] == 1
    assert report["results"][0]["checks"]["keywords_all"] is True


def _hit(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    title: str = "文档一",
    page: int | None = None,
    text: str = "正文内容",
) -> dict:
    metadata = {"doc_id": doc_id, "doc_title": title, "kb_id": "default"}
    if page is not None:
        metadata["page"] = page
    return {"chunk_id": chunk_id, "text": text, "metadata": metadata}


# ---------- 候选归一化 ----------


def test_hit_doc_fields_supports_nested_and_flat_shapes():
    nested = {"chunk_id": "a", "metadata": {"doc_id": "d1", "doc_title": "T"}}
    flat = {"doc_id": "d2", "doc_title": "T2", "text": "x"}
    assert eq.hit_doc_fields(nested) == ("d1", "T")
    assert eq.hit_doc_fields(flat) == ("d2", "T2")
    assert eq.hit_doc_fields({"metadata": None}) == ("", "")


def test_hit_page_reads_page_and_page_start():
    assert eq.hit_page({"page": 3}) == 3
    assert eq.hit_page({"metadata": {"page_start": 2}}) == 2
    assert eq.hit_page({"metadata": {"page": 0}}) is None  # 0 不是合法页码
    assert eq.hit_page({}) is None


def test_dedupe_hits_by_chunk_id_then_index_then_text():
    hits = [
        _hit(chunk_id="a"),
        _hit(chunk_id="a", text="重复块"),
        {"text": "无 id 块", "metadata": {"doc_id": "d", "chunk_index": 2}},
        {"text": "无 id 块", "metadata": {"doc_id": "d", "chunk_index": 2}},
        {"text": "纯文本"},
        {"text": "纯文本"},
    ]
    ranked = eq.dedupe_hits(hits)
    assert len(ranked) == 3
    assert ranked[0]["chunk_id"] == "a"
    # 非 dict 元素被安全跳过
    assert eq.dedupe_hits([{"text": "x"}, "garbage", None])  # type: ignore[list-item]


# ---------- 真值归一化 ----------


def test_normalize_expected_sources_accepts_str_and_dict():
    truth = eq.normalize_expected_sources(["doc-1", {"doc_title": "采购制度"}, {}, 123])
    assert truth == [
        {"doc_id": "doc-1", "doc_title": "doc-1"},
        {"doc_id": "", "doc_title": "采购制度"},
    ]


def test_normalize_expected_pages_accepts_int_and_dict():
    pages = eq.normalize_expected_pages([3, {"page": 5, "doc_id": "d1"}, {"page": None}, True])
    assert pages == [
        {"page": 3, "doc_id": "", "doc_title": ""},
        {"page": 5, "doc_id": "d1", "doc_title": ""},
    ]


# ---------- 检索指标 ----------


def test_hit_and_recall_at_k_with_source_truth():
    truth = eq.build_case_truth(
        {"expected_sources": ["d1", {"doc_title": "文档二"}]}
    )
    hits = [
        _hit(chunk_id="c1", doc_id="d1"),
        _hit(chunk_id="c2", doc_id="other", title="其他"),
        _hit(chunk_id="c3", doc_id="d2", title="文档二"),
    ]
    record = eq.evaluate_pool(truth, hits, (1, 3, 5))
    assert record["hit@1"] is True  # top-1（c1）就是 d1
    assert record["hit@3"] is True
    assert record["recall@3"] == 1.0
    assert record["recall@1"] == 0.5  # 两个期望来源中，top-1 只命中 d1
    assert record["evidence_rate"] is None  # 无证据真值 -> 不计分


def test_source_and_evidence_matching_normalize_paths_and_pdf_whitespace():
    truth = eq.build_case_truth(
        {
            "expected_sources": ["data_kb_test/cninfo/示例年报.pdf"],
            "expected_evidence": ["营业收入 23,357,748.41"],
        }
    )
    hits = [
        _hit(
            chunk_id="c1",
            title="示例年报.pdf",
            text="营业收入\n23,357,748.41 元",
        )
    ]

    record = eq.evaluate_pool(truth, hits, (1,))

    assert record["hit@1"] is True
    assert record["evidence_rate"] == 1.0


def test_source_matching_supports_qdrant_path_and_extensionless_stem():
    truth = eq.build_case_truth(
        {"expected_sources": ["data_kb_test/cninfo/示例年报.pdf"]}
    )
    hit = _hit(
        chunk_id="qdrant-c1",
        doc_id="doc-550e8400-e29b-41d4-a716-446655440000",
        title="示例年报",
    )
    hit["metadata"]["source_path"] = (
        "code/backend/data_kb_test/cninfo/示例年报.pdf"
    )

    record = eq.evaluate_pool(truth, [hit], (1, 3, 5))

    assert record["hit@1"] is True
    assert record["hit@3"] is True
    assert record["recall@5"] == 1.0


def test_qdrant_source_path_mismatch_is_not_hidden_by_same_title():
    truth = eq.build_case_truth(
        {"expected_sources": ["data_kb_test/cninfo/示例年报.pdf"]}
    )
    hit = _hit(title="示例年报")
    hit["metadata"]["source_path"] = "code/backend/data_kb_test/flk/示例年报.pdf"

    assert eq.evaluate_pool(truth, [hit], (1,))["hit@1"] is False


def test_confirmed_testset_false_positive_fixes_are_minimal_and_semantic():
    import yaml

    path = Path(__file__).resolve().parents[2] / "testsets" / "rag_real_quality_v2.yaml"
    cases = {
        case["id"]: case
        for case in yaml.safe_load(path.read_text(encoding="utf-8"))["tests"]
    }
    jinzhou = cases["real_jinzhou_port_2025_plan_not_promise"]
    yutai = cases["real_yutai_2025_h1_growth_rate"]

    assert jinzhou["expect_keywords_all"] == [
        "17.21亿元",
        "不构成对投资者的业绩承诺",
    ]
    assert eq.semantic_keyword_matches(
        jinzhou["expect_keywords_all"][1],
        "该经营计划不构成对投资者的业绩承诺。[1]",
    )
    assert yutai["expect_keyword"] == "43.41%"
    assert eq.semantic_keyword_matches(yutai["expect_keyword"], "同比增加43.41%。")


def test_recall_counts_misses_when_expected_doc_absent():
    truth = eq.build_case_truth({"expected_sources": ["d1", "d2"]})
    hits = [_hit(chunk_id="c1", doc_id="d1"), _hit(chunk_id="c2", doc_id="d9")]
    record = eq.evaluate_pool(truth, hits, (5,))
    assert record["recall@5"] == 0.5
    assert record["hit@5"] is True


def test_evidence_rate_over_candidate_texts():
    truth = eq.build_case_truth({"expected_evidence": ["五万元", "公开招标", "不存在"]})
    hits = [_hit(chunk_id="c1", text="单笔超过五万元须公开招标"), _hit(chunk_id="c2", text="其他")]
    record = eq.evaluate_pool(truth, hits, (5,))
    assert record["evidence_rate"] == round(2 / 3, 4)
    assert record["hit@5"] is None  # 无来源真值


def test_page_metric_requires_truth_and_page_metadata():
    truth = eq.build_case_truth({"expected_pages": [2]})
    # 候选无页码元数据 -> 不可计分（不是失败）
    no_meta = eq.evaluate_pool(truth, [_hit(chunk_id="c1")], (5,))
    assert no_meta["page_hit"] is None
    assert no_meta["page_no_metadata"] is True
    # 有页码元数据且命中
    hit_case = eq.evaluate_pool(truth, [_hit(chunk_id="c1", page=2)], (5,))
    assert hit_case["page_hit"] is True
    assert hit_case["page_recall"] == 1.0
    # 页码不匹配 -> 计分且未命中
    miss = eq.evaluate_pool(truth, [_hit(chunk_id="c1", page=9)], (5,))
    assert miss["page_hit"] is False
    assert miss["page_no_metadata"] is False
    # 无页码真值 -> 不计分
    none_truth = eq.build_case_truth({})
    assert eq.evaluate_pool(none_truth, [_hit(page=1)], (5,))["page_hit"] is None


def test_page_truth_with_source_constraint():
    truth = eq.build_case_truth({"expected_pages": [{"page": 3, "doc_id": "d1"}]})
    wrong_doc = eq.evaluate_pool(truth, [_hit(chunk_id="c1", doc_id="d2", page=3)], (5,))
    assert wrong_doc["page_hit"] is False
    right_doc = eq.evaluate_pool(truth, [_hit(chunk_id="c1", doc_id="d1", page=3)], (5,))
    assert right_doc["page_hit"] is True


# ---------- 状态区分：无真值 / 检索失败 / 请求错误 ----------


def test_case_retrieval_status_matrix():
    truth = eq.build_case_truth({"expected_sources": ["d1"]})
    # 无真值 -> not_scoreable
    no_truth = eq.evaluate_case_retrieval(eq.build_case_truth({}), {"contexts": []}, (5,))
    assert no_truth["status"] == "not_scoreable"
    # 有真值但两个池都空 -> retrieval_failure（进分母按未命中计）
    failure = eq.evaluate_case_retrieval(truth, {"contexts": [], "search_meta": {}}, (5,))
    assert failure["status"] == "retrieval_failure"
    assert failure["contexts"]["hit@5"] is False
    # 有真值且召回 -> scored
    scored = eq.evaluate_case_retrieval(
        truth, {"contexts": [_hit()], "search_meta": {"recall_raw": [_hit()]}}, (5,)
    )
    assert scored["status"] == "scored"
    assert scored["contexts"]["hit@5"] is True
    # 请求错误 -> error，不进检索分母
    error = eq.evaluate_case_retrieval(truth, None, (5,))
    assert error["status"] == "error"
    agg = eq.aggregate_pool(
        [error, scored, no_truth, failure], "contexts", (5,)
    )
    # 分母只含 scored + retrieval_failure（error 与 not_scoreable 排除）
    assert agg["hit@5_denominator"] == 2
    assert agg["hit@5"] == 0.5


def test_contexts_falls_back_to_citations_shape():
    truth = eq.build_case_truth({"expected_sources": ["d1"]})
    citation = {"index": 1, "doc_id": "d1", "doc_title": "文档一", "page": 2, "text": "内容"}
    record = eq.evaluate_case_retrieval(truth, {"citations": [citation], "search_meta": {}}, (5,))
    assert record["status"] == "scored"
    assert record["contexts"]["hit@5"] is True


# ---------- 聚合 ----------


def test_aggregate_pool_tracks_per_metric_denominators():
    truth_source = eq.build_case_truth({"expected_sources": ["d1"]})
    truth_evidence = eq.build_case_truth({"expected_evidence": ["关键词"]})
    records = [
        eq.evaluate_case_retrieval(
            truth_source, {"contexts": [_hit(doc_id="d1")]}, (5,)
        ),
        # 只有证据真值：hit 不进分母，evidence 进
        eq.evaluate_case_retrieval(
            truth_evidence, {"contexts": [_hit(text="含关键词")]}, (5,)
        ),
        eq.evaluate_case_retrieval(truth_source, {"contexts": []}, (5,)),  # 检索失败
    ]
    agg = eq.aggregate_pool(records, "contexts", (5,))
    assert agg["hit@5_denominator"] == 2
    assert agg["hit@5"] == 0.5
    assert agg["evidence_rate_denominator"] == 1
    assert agg["evidence_rate"] == 1.0
    assert agg["page"]["denominator"] == 0
    assert agg["page"]["hit_rate"] is None


def test_percentile_handles_empty_and_values():
    assert eq._percentile([], 0.95) is None
    values = [1.0, 2.0, 3.0, 4.0]
    assert eq._percentile(values, 0.5) == 2.0
    assert eq._percentile(values, 0.95) == 4.0


def test_sanitize_base_url_strips_credentials_and_paths():
    assert eq.sanitize_base_url("https://user:pass@host.example:8000/api?x=1") == "https://host.example:8000"
    assert eq.sanitize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert eq.sanitize_base_url("") == ""


# ---------- run_rag 离线集成（mock ask，旧题集兼容 + 新真值） ----------


def _ask_response(
    *,
    answer: str = "根据制度回答 [1]",
    hits: list[dict] | None = None,
    recall: list[dict] | None = None,
    escalate: bool = False,
) -> dict:
    hits = hits if hits is not None else [_hit()]
    recall = recall if recall is not None else hits
    citations = [
        {
            "index": i + 1,
            "chunk_id": hit["chunk_id"],
            "doc_id": hit["metadata"]["doc_id"],
            "doc_title": hit["metadata"]["doc_title"],
            "page": hit["metadata"].get("page"),
            "text": hit["text"],
            "metadata": hit["metadata"],
        }
        for i, hit in enumerate(hits)
    ]
    return {
        "answer": answer,
        "citations": citations,
        "contexts": hits,
        "score": 9,
        "escalate": escalate,
        "search_meta": {
            "intent": "kb_question",
            "top_score": 0.8,
            "escalate": escalate,
            "recall_raw": recall,
            "contexts": hits,
        },
    }


def _write_testset(tmp_path: Path, tests: list[dict]) -> Path:
    path = tmp_path / "testset.yaml"
    import yaml

    path.write_text(yaml.safe_dump({"kb_id": "default", "tests": tests}), encoding="utf-8")
    return path


def test_run_rag_keeps_legacy_fields_and_adds_summary(tmp_path, monkeypatch):
    """旧题集（只有 expect_* 字段）继续工作，且报告新增 summary/by_tags。"""
    testset = _write_testset(
        tmp_path,
        [
            {
                "id": "old-1",
                "category": "positive_retrieval",
                "question": "采购金额超过多少必须公开招标？",
                "expect_answerable": True,
                "expect_keyword": "5万",
                "expect_min_citations": 1,
            }
        ],
    )
    monkeypatch.setattr(
        eq, "ask", lambda *a, **k: _ask_response(answer="超过5万元必须公开招标 [1]")
    )
    report = eq.run_rag(testset, "http://127.0.0.1:8000", "user-secret-123")
    assert report["kind"] == "rag" and report["schema_version"] == 2
    assert report["total"] == 1 and report["passed"] == 1
    item = report["results"][0]
    assert item["checks"]["answerable"] is True
    assert item["checks"]["keyword"] is True
    # 无真值 -> 不可计分，不进分母
    assert item["retrieval"]["status"] == "not_scoreable"
    assert report["summary"]["retrieval"]["contexts"]["hit@5_denominator"] == 0
    assert report["summary"]["retrieval"]["contexts"]["hit@5"] is None
    # 旧 by_category 键保留
    assert report["by_category"]["positive_retrieval"]["passed"] == 1
    # user_id 绝不能写入报告
    assert "user-secret-123" not in json.dumps(report, ensure_ascii=False)


def test_run_rag_scores_new_truth_fields_and_tag_slices(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {
                "id": "new-1",
                "question": "问一",
                "tags": ["law", "core"],
                "expected_sources": ["d1"],
                "expected_pages": [2],
                "expected_evidence": ["五万元"],
            },
            {
                "id": "new-2",
                "question": "问二",
                "tags": ["law"],
                "expected_sources": ["d2"],
            },
        ],
    )

    def fake_ask(base_url, kb_id, question, user_id, history=None):
        if question == "问一":
            return _ask_response(hits=[_hit(chunk_id="c1", doc_id="d1", page=2, text="超过五万元须招标")])
        return _ask_response(hits=[_hit(chunk_id="c9", doc_id="d1")])  # d2 未命中

    monkeypatch.setattr(eq, "ask", fake_ask)
    report = eq.run_rag(testset, "http://127.0.0.1:8000", None)
    contexts = report["summary"]["retrieval"]["contexts"]
    assert contexts["hit@5_denominator"] == 2
    assert contexts["hit@5"] == 0.5
    assert contexts["recall@5"] == 0.5
    assert contexts["evidence_rate_denominator"] == 1
    assert contexts["evidence_rate"] == 1.0
    assert contexts["page"]["denominator"] == 1
    assert contexts["page"]["hit_rate"] == 1.0
    law = report["by_tags"]["law"]
    assert law["total"] == 2
    assert law["retrieval"]["contexts"]["hit@5"] == 0.5
    core = report["by_tags"]["core"]
    assert core["total"] == 1


def test_run_rag_request_error_excluded_from_retrieval_denominator(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {"id": "e1", "question": "会失败", "expected_sources": ["d1"]},
            {"id": "e2", "question": "正常", "expected_sources": ["d1"]},
        ],
    )

    def fake_ask(base_url, kb_id, question, user_id, history=None):
        if question == "会失败":
            raise RuntimeError("connection refused")
        return _ask_response(hits=[_hit(doc_id="d1")])

    monkeypatch.setattr(eq, "ask", fake_ask)
    report = eq.run_rag(testset, "http://127.0.0.1:8000", None)
    assert report["summary"]["error_count"] == 1
    failed = report["results"][0]
    assert failed["passed"] is False and failed["error"]
    assert failed["retrieval"]["status"] == "error"
    contexts = report["summary"]["retrieval"]["contexts"]
    assert contexts["hit@5_denominator"] == 1  # 错误案例不进检索分母
    assert contexts["hit@5"] == 1.0


def test_run_rag_empty_retrieval_with_truth_is_failure(tmp_path, monkeypatch):
    testset = _write_testset(tmp_path, [{"id": "f1", "question": "空召回", "expected_sources": ["d1"]}])
    monkeypatch.setattr(
        eq,
        "ask",
        lambda *a, **k: _ask_response(hits=[], recall=[], answer="未检索到相关信息"),
    )
    report = eq.run_rag(testset, "http://127.0.0.1:8000", None)
    item = report["results"][0]
    assert item["retrieval"]["status"] == "retrieval_failure"
    contexts = report["summary"]["retrieval"]["contexts"]
    assert contexts["hit@5_denominator"] == 1
    assert contexts["hit@5"] == 0.0  # 计分为未命中


def test_run_rag_latency_and_answerable_summary(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {"id": "l1", "question": "快", "expect_answerable": True},
            {"id": "l2", "question": "慢", "expect_answerable": False, "expect_escalate": True},
        ],
    )

    def fake_ask(base_url, kb_id, question, user_id, history=None):
        if question == "慢":
            return _ask_response(answer="已转人工", escalate=True, hits=[])
        return _ask_response()

    monkeypatch.setattr(eq, "ask", fake_ask)
    report = eq.run_rag(testset, "http://127.0.0.1:8000", None)
    summary = report["summary"]
    assert summary["answerable"]["denominator"] == 2
    assert summary["answerable"]["accuracy"] == 1.0
    assert summary["escalate_rate"] == 0.5
    assert summary["latency"]["count"] == 2
    assert summary["latency"]["p50_s"] is not None


def test_main_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        eq.main()
    assert excinfo.value.code  # 无参数 -> argparse.error -> 退出码 2


def test_run_rag_progress_callback_keeps_parseable_partial_reports(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {"id": "p1", "question": "问一"},
            {"id": "p2", "question": "问二"},
        ],
    )
    monkeypatch.setattr(eq, "ask", lambda *a, **k: _ask_response())
    snapshots: list[dict] = []

    report = eq.run_rag(
        testset,
        "http://127.0.0.1:8000",
        None,
        progress_callback=snapshots.append,
    )

    assert [snapshot["total"] for snapshot in snapshots] == [0, 1, 2]
    assert [snapshot["planned_total"] for snapshot in snapshots] == [2, 2, 2]
    assert [snapshot["completed_total"] for snapshot in snapshots] == [0, 1, 2]
    assert [snapshot["run_status"] for snapshot in snapshots] == [
        "in_progress",
        "in_progress",
        "in_progress",
    ]
    assert snapshots[-1]["results"] == report["results"]
    assert report["run_status"] == "completed"
    assert report["planned_total"] == report["completed_total"] == 2
    for snapshot in snapshots:
        json.loads(json.dumps(snapshot, ensure_ascii=False))
        assert snapshot["schema_version"] == 2


def test_atomic_report_write_failure_preserves_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(eq.os, "replace", fail_replace)
    assert eq._write_json_atomic(output, {"new": True}) is False
    assert output.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_main_interrupt_writes_last_completed_cases(tmp_path, monkeypatch):
    testset = _write_testset(
        tmp_path,
        [
            {"id": "i1", "question": "已完成"},
            {"id": "i2", "question": "将中断"},
        ],
    )
    calls = 0

    def fake_ask(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _ask_response()
        raise KeyboardInterrupt

    output = tmp_path / "nested" / "partial.json"
    monkeypatch.setattr(eq, "ask", fake_ask)
    monkeypatch.setattr(
        eq.sys,
        "argv",
        ["evaluate_quality.py", "--rag", str(testset), "--output", str(output)],
    )

    assert eq.main() == 130
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["reports"]) == 1
    assert report["reports"][0]["total"] == 1
    assert report["reports"][0]["planned_total"] == 2
    assert report["reports"][0]["completed_total"] == 1
    assert report["reports"][0]["run_status"] == "interrupted"
    assert [item["id"] for item in report["reports"][0]["results"]] == ["i1"]


def test_progress_write_failure_is_recorded_without_failing_completed_case(
    tmp_path, monkeypatch
):
    testset = _write_testset(tmp_path, [{"id": "w1", "question": "完成"}])
    output = tmp_path / "report.json"
    calls = 0
    real_atomic_write = eq._write_json_atomic

    def write_once_fails(path, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        return real_atomic_write(path, payload)

    monkeypatch.setattr(eq, "ask", lambda *a, **k: _ask_response())
    monkeypatch.setattr(eq, "_write_json_atomic", write_once_fails)
    monkeypatch.setattr(
        eq.sys,
        "argv",
        ["evaluate_quality.py", "--rag", str(testset), "--output", str(output)],
    )

    assert eq.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    rag = report["reports"][0]
    assert rag["run_status"] == "completed"
    assert rag["planned_total"] == rag["completed_total"] == 1
    assert rag["passed"] == 1
    assert rag["report_write_failures"] == 1


def test_final_write_failure_keeps_old_report_and_returns_nonzero(
    tmp_path, monkeypatch, capsys
):
    testset = _write_testset(tmp_path, [{"id": "f1", "question": "完成"}])
    output = tmp_path / "report.json"
    output.write_text('{"old": true}', encoding="utf-8")

    monkeypatch.setattr(eq, "ask", lambda *a, **k: _ask_response())
    monkeypatch.setattr(eq, "_write_json_atomic", lambda *a, **k: False)
    monkeypatch.setattr(
        eq.sys,
        "argv",
        ["evaluate_quality.py", "--rag", str(testset), "--output", str(output)],
    )

    assert eq.main() == 1
    assert output.read_text(encoding="utf-8") == '{"old": true}'
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    rag = printed["reports"][0]
    assert rag["run_status"] == "completed"
    assert rag["passed"] == rag["total"] == 1
    assert rag["report_write_failures"] >= 1
    assert "final report could not be written" in captured.err


def test_main_exception_marks_last_snapshot_failed(tmp_path, monkeypatch):
    testset = _write_testset(tmp_path, [{"id": "x1", "question": "异常"}])
    output = tmp_path / "failed.json"

    def fail_after_snapshot(*args, progress_callback=None, **kwargs):
        progress_callback(
            {
                "kind": "rag",
                "schema_version": 2,
                "run_status": "in_progress",
                "planned_total": 1,
                "completed_total": 0,
                "total": 0,
                "passed": 0,
                "results": [],
            }
        )
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(eq, "run_rag", fail_after_snapshot)
    monkeypatch.setattr(
        eq.sys,
        "argv",
        ["evaluate_quality.py", "--rag", str(testset), "--output", str(output)],
    )

    assert eq.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    rag = report["reports"][0]
    assert rag["run_status"] == "failed"
    assert rag["planned_total"] == 1
    assert rag["completed_total"] == 0
