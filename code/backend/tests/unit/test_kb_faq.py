"""FAQ store 回归测试（收尾第 4 步）。

覆盖：增删查、相似问拆分、删除后索引可查（不全量重建索引）。
"""
from __future__ import annotations

from services.kb.faq_store import FAQStore, _cosine
from tests.mocks import FakeEmbedding


def test_faq_add_list_delete(tmp_path):
    store = FAQStore(str(tmp_path / "f.db"), FakeEmbedding())
    fid = store.add_faq("default", "怎么退款", "请走退款流程", ["如何退款", "退款步骤"])
    faqs = store.list_faqs("default")
    assert len(faqs) == 1
    assert faqs[0]["question"] == "怎么退款"
    assert faqs[0]["similar_questions"] == ["如何退款", "退款步骤"]
    # 删除后可查（不全量重建，只移除该条）
    assert store.delete_faq(fid) == 1
    assert store.list_faqs("default") == []


def test_faq_split_similars():
    assert FAQStore._split_similars("a;b；c,d") == ["a", "b", "c", "d"]
    assert FAQStore._split_similars("  只有一个  ") == ["只有一个"]
    assert FAQStore._split_similars("") == []


def test_faq_search_hits(tmp_path):
    store = FAQStore(str(tmp_path / "f.db"), FakeEmbedding())
    store.add_faq("default", "采购超过多少要招投标", "超过5万", ["招投标金额门槛"])
    hit = store.search("招投标多少钱", "default", threshold=0.0)
    assert hit is not None
    assert hit["answer"] == "超过5万"
    assert "score" in hit


def test_faq_search_miss_on_empty(tmp_path):
    store = FAQStore(str(tmp_path / "f.db"), FakeEmbedding())
    assert store.search("随便问", "default", threshold=0.9) is None


def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    # 维度不一致返回 0
    assert _cosine([1.0], [1.0, 0.0]) == 0.0
    # 空向量返回 0
    assert _cosine([], []) == 0.0
