"""一键增量入库脚本的离线行为测试。"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest_corpus.py"
_SPEC = importlib.util.spec_from_file_location("ingest_corpus", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
corpus = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = corpus
_SPEC.loader.exec_module(corpus)


class FakeResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(
        self,
        *,
        post_status: int = 200,
        post_doc_id: str = "doc-new",
        remote_docs: list[dict[str, Any]] | None = None,
        remote_status: int = 200,
    ):
        self.calls: list[tuple[str, str, dict]] = []
        self.post_status = post_status
        self.post_doc_id = post_doc_id
        self.remote_docs = list(remote_docs or [])
        self.remote_status = remote_status

    def get(self, url: str, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/kb/docs"):
            if self.remote_status != 200:
                return FakeResponse(self.remote_status, {"detail": "模拟对账失败"})
            params = kwargs.get("params") or {}
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 500))
            page = self.remote_docs[offset : offset + limit]
            return FakeResponse(200, {"docs": page, "total": len(self.remote_docs)})
        return FakeResponse(200, {"status": "ok"})

    def post(self, url: str, **kwargs):
        self.calls.append(("post", url, kwargs))
        if self.post_status == 409:
            return FakeResponse(409, {"detail": "文档内容已存在"})
        if self.post_status >= 400:
            return FakeResponse(self.post_status, {"detail": "模拟服务失败"})
        return FakeResponse(200, {"doc_id": self.post_doc_id, "title": kwargs["data"]["title"]})

    def put(self, url: str, **kwargs):
        self.calls.append(("put", url, kwargs))
        return FakeResponse(200, {"doc_id": url.rsplit("/", 1)[-1]})

    def close(self):
        return None


def _args(tmp_path: Path, *extra: str):
    return corpus.parse_args(
        [
            "--directory",
            str(tmp_path),
            "--state",
            str(tmp_path / "state.json"),
            "--base-url",
            "http://kb.test",
            *extra,
        ]
    )


def _write_pdf(tmp_path: Path, name: str = "财务报告-2025年半年度报告.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.7\nfinancial test content\n")
    return path


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(state: dict) -> dict:
    assert len(state["entries"]) == 1
    return next(iter(state["entries"].values()))


def _remote_doc(
    doc_id: str,
    title: str = "财务报告-2025年半年度报告",
    *,
    chunks: int = 1,
    kb_id: str = "cninfo_report",
) -> dict[str, Any]:
    return {"doc_id": doc_id, "title": title, "chunks": chunks, "kb_id": kb_id}


def test_first_run_posts_with_chinese_filename_title_and_records_doc_id(tmp_path):
    _write_pdf(tmp_path)
    session = FakeSession(post_doc_id="doc-1")

    assert corpus.run(_args(tmp_path), session=session) == 0

    calls = [call for call in session.calls if call[0] == "post"]
    assert len(calls) == 1
    assert calls[0][2]["data"]["title"] == "财务报告-2025年半年度报告"
    entry = _entry(_load_state(tmp_path / "state.json"))
    assert entry["kb_id"] == "cninfo_report"
    assert Path(entry["path"]).is_absolute()
    assert entry["title"] == "财务报告-2025年半年度报告"
    assert entry["doc_id"] == "doc-1"
    assert entry["last_status"] == "success"
    assert entry["last_action"] == "NEW"
    assert entry["sha256"]
    assert entry["pipeline_fingerprint"]


def test_same_hash_skips_without_post_or_put(tmp_path):
    _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession()) == 0

    second = FakeSession(remote_docs=[_remote_doc("doc-new")])
    assert corpus.run(_args(tmp_path), session=second) == 0

    assert [call for call in second.calls if call[0] in {"post", "put"}] == []
    assert any(call[0] == "get" for call in second.calls)
    assert _entry(_load_state(tmp_path / "state.json"))["last_action"] == "SKIP"


def test_changed_content_puts_existing_doc_id(tmp_path):
    path = _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession(post_doc_id="doc-keep")) == 0
    path.write_bytes(b"%PDF-1.7\nchanged content\n")

    second = FakeSession(remote_docs=[_remote_doc("doc-keep")])
    assert corpus.run(_args(tmp_path), session=second) == 0

    puts = [call for call in second.calls if call[0] == "put"]
    assert len(puts) == 1
    assert puts[0][1].endswith("/kb/docs/doc-keep")
    assert not [call for call in second.calls if call[0] == "post"]
    assert _entry(_load_state(tmp_path / "state.json"))["last_action"] == "UPDATE"


def test_failed_post_is_recorded_and_retried_next_run(tmp_path):
    _write_pdf(tmp_path)
    first = FakeSession(post_status=500)
    assert corpus.run(_args(tmp_path), session=first) == 1
    failed = _entry(_load_state(tmp_path / "state.json"))
    assert failed["last_status"] == "failed"
    assert failed["last_action"] == "FAIL"

    second = FakeSession(post_doc_id="retry-doc")
    assert corpus.run(_args(tmp_path), session=second) == 0
    assert len([call for call in second.calls if call[0] == "post"]) == 1
    retried = _entry(_load_state(tmp_path / "state.json"))
    assert retried["last_status"] == "success"
    assert retried["doc_id"] == "retry-doc"


def test_mtime_change_with_same_content_still_skips(tmp_path):
    path = _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession()) == 0
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))

    second = FakeSession(remote_docs=[_remote_doc("doc-new")])
    assert corpus.run(_args(tmp_path), session=second) == 0

    assert not [call for call in second.calls if call[0] in {"post", "put"}]
    assert _entry(_load_state(tmp_path / "state.json"))["last_action"] == "SKIP"


def test_dry_run_prints_plan_without_request_or_state_write(tmp_path, capsys):
    _write_pdf(tmp_path)
    session = FakeSession()

    assert corpus.run(_args(tmp_path, "--dry-run"), session=session) == 0

    assert not session.calls
    assert not (tmp_path / "state.json").exists()
    assert "DRY-RUN NEW" in capsys.readouterr().out


def test_testset_deduplicates_the_real_22_pdf_sources():
    testset = Path(__file__).resolve().parents[2] / "testsets" / "rag_real_quality_v2.yaml"

    documents = corpus.collect_documents(testset=testset)

    assert len(documents) == 22
    assert len({str(document.path).casefold() for document in documents}) == 22
    assert all(document.path.suffix.lower() == ".pdf" for document in documents)


def test_directory_scan_is_recursive_and_case_insensitive(tmp_path):
    nested = tmp_path / "中文目录"
    nested.mkdir()
    (nested / "a.PDF").write_bytes(b"a")
    (nested / "b.txt").write_text("not pdf", encoding="utf-8")

    documents = corpus.collect_documents(directory=tmp_path)

    assert [document.title for document in documents] == ["a"]


def test_atomic_state_replace_failure_keeps_old_file(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    old = '{"entries": {"old": {"last_status": "success"}}}\n'
    state_path.write_text(old, encoding="utf-8")

    def fail_replace(*_args, **_kwargs):
        raise OSError("模拟替换失败")

    monkeypatch.setattr(corpus.os, "replace", fail_replace)
    with pytest.raises(OSError, match="模拟替换失败"):
        corpus.save_state_atomic(state_path, {"entries": {"new": {}}})

    assert state_path.read_text(encoding="utf-8") == old


def test_http_failure_is_nonzero_and_409_is_recoverable(tmp_path, capsys):
    _write_pdf(tmp_path)
    session = FakeSession(post_status=409)

    assert corpus.run(_args(tmp_path), session=session) == 1

    output = capsys.readouterr().out
    assert "409" in output
    assert "核对状态清单中的 doc_id" in output
    entry = _entry(_load_state(tmp_path / "state.json"))
    assert entry["last_status"] == "failed"
    assert entry["last_http_status"] == 409


def test_pipeline_change_only_warns_and_does_not_rebuild(tmp_path, capsys):
    _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession()) == 0
    state_path = tmp_path / "state.json"
    state = _load_state(state_path)
    entry = _entry(state)
    entry["pipeline_fingerprint"] = "old-pipeline"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    second = FakeSession(remote_docs=[_remote_doc(_entry(state)["doc_id"])])
    assert corpus.run(_args(tmp_path), session=second) == 0

    assert not [call for call in second.calls if call[0] in {"post", "put"}]
    assert "规则已变化，是否用 --force 重建" in capsys.readouterr().out
    assert _entry(_load_state(state_path))["pipeline_fingerprint"] == "old-pipeline"


def test_health_failure_is_nonzero_and_explains_how_to_start(tmp_path, capsys):
    _write_pdf(tmp_path)

    class UnreadySession(FakeSession):
        def get(self, url: str, **kwargs):
            self.calls.append(("get", url, kwargs))
            return FakeResponse(503, {"detail": "依赖不可用"})

    assert corpus.run(_args(tmp_path), session=UnreadySession()) == 2
    assert "一键启动前后端.cmd" in capsys.readouterr().err
    assert not (tmp_path / "state.json").exists()


def test_remote_document_reconciliation_is_paginated_with_identity(tmp_path, monkeypatch):
    remote = [_remote_doc(f"doc-{index}", f"标题-{index}") for index in range(501)]
    monkeypatch.setenv("KB_INGEST_TOKEN", "test-token")
    session = FakeSession(remote_docs=remote)

    actual = corpus.fetch_remote_documents(
        session,
        "http://kb.test",
        kb_id="cninfo_report",
        user_id="operator",
        timeout=30,
    )

    assert len(actual) == 501
    pages = [call for call in session.calls if call[0] == "get" and call[1].endswith("/kb/docs")]
    assert [call[2]["params"]["offset"] for call in pages] == [0, 500]
    assert all(call[2]["params"]["limit"] == 500 for call in pages)
    assert all(call[2]["params"]["kb_id"] == "cninfo_report" for call in pages)
    assert all(call[2]["params"]["user_id"] == "operator" for call in pages)
    assert all(call[2]["headers"] == {"X-Api-Token": "test-token"} for call in pages)


def test_state_doc_missing_from_remote_is_new_not_put(tmp_path):
    _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession(post_doc_id="old-doc")) == 0

    second = FakeSession(post_doc_id="recreated-doc", remote_docs=[])
    assert corpus.run(_args(tmp_path), session=second) == 0

    assert len([call for call in second.calls if call[0] == "post"]) == 1
    assert not [call for call in second.calls if call[0] == "put"]
    assert _entry(_load_state(tmp_path / "state.json"))["doc_id"] == "recreated-doc"


def test_doc_id_not_in_target_kb_is_never_put(tmp_path):
    path = _write_pdf(tmp_path)
    document = corpus.snapshot_document(corpus.DocumentInput(path=path, title=path.stem))
    state = corpus.empty_state()
    state["entries"][corpus.state_key("cninfo_report", path)] = {
        "kb_id": "cninfo_report",
        "path": str(path.resolve()),
        "sha256": document.sha256,
        "size": document.size,
        "mtime": document.mtime,
        "title": document.title,
        "doc_id": "doc-from-other-kb-or-missing",
        "last_status": "success",
        "last_action": "NEW",
        "pipeline_fingerprint": "test",
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    session = FakeSession(post_doc_id="safe-new", remote_docs=[])
    assert corpus.run(_args(tmp_path), session=session) == 0

    assert len([call for call in session.calls if call[0] == "post"]) == 1
    assert not [call for call in session.calls if call[0] == "put"]


def test_missing_state_with_same_title_is_identity_conflict_without_post(tmp_path):
    _write_pdf(tmp_path)
    session = FakeSession(remote_docs=[_remote_doc("server-doc")])

    assert corpus.run(_args(tmp_path), session=session) == 1

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    assert "恢复状态清单或人工核对" in _entry(_load_state(tmp_path / "state.json"))["last_error"]


def test_duplicate_remote_title_is_conflict_without_post(tmp_path):
    _write_pdf(tmp_path)
    session = FakeSession(
        remote_docs=[_remote_doc("server-a"), _remote_doc("server-b")]
    )

    assert corpus.run(_args(tmp_path), session=session) == 1

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    assert "多个 doc_id" in _entry(_load_state(tmp_path / "state.json"))["last_error"]


def test_remote_zero_chunks_is_incomplete_and_not_overwritten(tmp_path):
    _write_pdf(tmp_path)
    session = FakeSession(remote_docs=[_remote_doc("empty-doc", chunks=0)])

    assert corpus.run(_args(tmp_path), session=session) == 1

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    assert "chunks<=0" in _entry(_load_state(tmp_path / "state.json"))["last_error"]


def test_remote_reconciliation_failure_stops_before_any_write(tmp_path, capsys):
    _write_pdf(tmp_path)
    session = FakeSession(remote_status=503)

    assert corpus.run(_args(tmp_path), session=session) == 2

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    assert "远端文档对账失败" in capsys.readouterr().err
    assert _load_state(tmp_path / "state.json")["last_run"]["run_status"] == "failed"


def test_last_run_completed_counts_cases_and_actions(tmp_path):
    first = _write_pdf(tmp_path)
    assert corpus.run(_args(tmp_path), session=FakeSession(post_doc_id="existing-doc")) == 0
    second = _write_pdf(tmp_path, "另一份报告-2024年年度报告.pdf")
    session = FakeSession(
        post_doc_id="new-doc",
        remote_docs=[_remote_doc("existing-doc", first.stem)],
    )

    assert corpus.run(_args(tmp_path), session=session) == 0

    last_run = _load_state(tmp_path / "state.json")["last_run"]
    assert last_run["run_status"] == "completed"
    assert last_run["planned_total"] == 2
    assert last_run["completed_total"] == 2
    assert last_run["new"] == 1
    assert last_run["updated"] == 0
    assert last_run["skipped"] == 1
    assert last_run["failed"] == 0
    assert last_run["started_at"] and last_run["finished_at"]
    assert second.exists()


def test_checkpoint_write_failure_does_not_count_completed_request_as_business_failure(
    tmp_path, monkeypatch, capsys
):
    _write_pdf(tmp_path, "a-first.pdf")
    second = _write_pdf(tmp_path, "b-second.pdf")
    real_save = corpus.save_state_atomic
    calls = 0

    def fail_one_checkpoint(path, state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("模拟进度快照失败")
        return real_save(path, state)

    monkeypatch.setattr(corpus, "save_state_atomic", fail_one_checkpoint)
    session = FakeSession()
    assert corpus.run(_args(tmp_path), session=session) == 1

    output = capsys.readouterr()
    assert "已完成请求不计为业务失败" in output.out
    assert len([call for call in session.calls if call[0] == "post"]) == 1
    assert not [call for call in session.calls if call[0] == "put"]
    last_run = _load_state(tmp_path / "state.json")["last_run"]
    saved = _load_state(tmp_path / "state.json")
    assert len(saved["entries"]) == 1
    assert all(str(second.resolve()) != entry["path"] for entry in saved["entries"].values())
    assert last_run["failed"] == 0
    assert last_run["completed_total"] == 1
    assert last_run["planned_total"] == 2
    assert last_run["new"] == 1
    assert last_run["state_write_failures"] == 1
    assert last_run["run_status"] == "failed"


def test_skip_checkpoint_failure_stops_later_post(tmp_path, monkeypatch):
    first = _write_pdf(tmp_path, "a-first.pdf")
    assert corpus.run(_args(tmp_path), session=FakeSession(post_doc_id="existing-doc")) == 0
    _write_pdf(tmp_path, "b-second.pdf")
    real_save = corpus.save_state_atomic
    calls = 0

    def fail_after_skip(path, state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("模拟 SKIP 快照失败")
        return real_save(path, state)

    monkeypatch.setattr(corpus, "save_state_atomic", fail_after_skip)
    session = FakeSession(remote_docs=[_remote_doc("existing-doc", first.stem)])
    assert corpus.run(_args(tmp_path), session=session) == 1

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    last_run = _load_state(tmp_path / "state.json")["last_run"]
    assert last_run["completed_total"] == 1
    assert last_run["skipped"] == 1
    assert last_run["new"] == 0


def test_fail_checkpoint_failure_stops_later_post(tmp_path, monkeypatch):
    first = _write_pdf(tmp_path, "a-first.pdf")
    _write_pdf(tmp_path, "b-second.pdf")
    real_save = corpus.save_state_atomic
    calls = 0

    def fail_after_conflict(path, state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("模拟 FAIL 快照失败")
        return real_save(path, state)

    monkeypatch.setattr(corpus, "save_state_atomic", fail_after_conflict)
    session = FakeSession(remote_docs=[_remote_doc("conflict-doc", first.stem)])
    assert corpus.run(_args(tmp_path), session=session) == 1

    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    last_run = _load_state(tmp_path / "state.json")["last_run"]
    assert last_run["completed_total"] == 1
    assert last_run["failed"] == 1
    assert last_run["new"] == 0


def test_cli_final_state_write_failure_is_nonzero_and_keeps_valid_snapshot(
    tmp_path, monkeypatch, capsys
):
    _write_pdf(tmp_path)
    session = FakeSession()
    monkeypatch.setattr(corpus.requests, "Session", lambda: session)
    real_save = corpus.save_state_atomic
    calls = 0

    def fail_final(path, state):
        nonlocal calls
        calls += 1
        if calls == 3:  # initial in_progress, per-document checkpoint, final snapshot
            raise OSError("模拟最终替换失败")
        return real_save(path, state)

    monkeypatch.setattr(corpus, "save_state_atomic", fail_final)
    assert corpus.main(
        [
            "--directory",
            str(tmp_path),
            "--state",
            str(tmp_path / "state.json"),
            "--base-url",
            "http://kb.test",
        ]
    ) == 1

    assert "最终状态清单写入失败" in capsys.readouterr().err
    preserved = _load_state(tmp_path / "state.json")
    assert preserved["last_run"]["run_status"] == "in_progress"
    assert preserved["last_run"]["completed_total"] == 1


def test_cli_initial_state_write_failure_is_nonzero_and_keeps_old_state(
    tmp_path, monkeypatch, capsys
):
    _write_pdf(tmp_path)
    state_path = tmp_path / "state.json"
    old = json.dumps(
        {
            "schema_version": 1,
            "entries": {"old": {"last_status": "success", "doc_id": "old-doc"}},
            "last_run": {"run_status": "completed", "planned_total": 0},
        },
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    state_path.write_text(old, encoding="utf-8")
    session = FakeSession()
    monkeypatch.setattr(corpus.requests, "Session", lambda: session)

    def fail_initial(*_args, **_kwargs):
        raise OSError("模拟初始快照失败")

    monkeypatch.setattr(corpus, "save_state_atomic", fail_initial)
    assert corpus.main(
        [
            "--directory",
            str(tmp_path),
            "--state",
            str(state_path),
            "--base-url",
            "http://kb.test",
        ]
    ) == 1

    assert state_path.read_text(encoding="utf-8") == old
    assert not [call for call in session.calls if call[0] in {"post", "put"}]
    assert "未执行任何入库请求" in capsys.readouterr().err


def test_cmd_has_utf8_identity_defaults_sync_wait_passthrough_and_exit_code():
    cmd_path = Path(__file__).resolve().parents[4] / "一键增量入库.cmd"
    text = cmd_path.read_bytes().decode("utf-8-sig")

    assert "chcp 65001 >nul" in text
    assert "set \"ROOT_DIR=%~dp0\"" in text
    assert '"%PYTHON_EXE%" "%SCRIPT%" %*' in text
    assert 'set "EXIT_CODE=%ERRORLEVEL%"' in text
    assert "exit /b %EXIT_CODE%" in text
    assert "set \"KB_INGEST_USER_ID=admin\"" in text
    assert "No ingest identity found" in text
    assert "if not defined KB_INGEST_TOKEN" in text
    assert "if not defined KB_API_TOKEN" in text
    assert "if not defined KB_INGEST_USER_ID" in text
    assert "if not defined KB_USER_ID" in text
    assert "start " not in text.lower()
