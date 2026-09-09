import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts" / "launch_isolated_backend.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("launch_isolated_backend_under_test", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_backend(tmp_path: Path) -> Path:
    backend = tmp_path / "backend"
    (backend / "src").mkdir(parents=True)
    (backend / "src" / "main.py").write_text("# smoke fixture\n", encoding="utf-8")
    return backend


def test_launch_popen_contract_inherits_env_and_returns_pid(tmp_path, monkeypatch, capsys):
    helper = load_helper()
    backend = make_backend(tmp_path)
    stdout_path = tmp_path / "backend.stdout.log"
    stderr_path = tmp_path / "backend.stderr.log"
    marker = "launch-helper-secret-must-not-print"
    monkeypatch.setenv("LUNAB_HELPER_MARKER", marker)
    captured = {}

    class FakeProcess:
        pid = 43210

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(helper.subprocess, "Popen", fake_popen)
    pid = helper.launch(str(backend), 18081, str(stdout_path), str(stderr_path))

    assert pid == 43210
    assert captured["command"] == [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--app-dir",
        "src",
        "--host",
        "127.0.0.1",
        "--port",
        "18081",
    ]
    assert captured["kwargs"]["cwd"] == str(backend.resolve())
    assert captured["kwargs"]["env"]["LUNAB_HELPER_MARKER"] == marker
    assert captured["kwargs"]["env"] is not os.environ
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert Path(captured["kwargs"]["stdout"].name) == stdout_path
    assert Path(captured["kwargs"]["stderr"].name) == stderr_path
    assert stdout_path.is_file() and stderr_path.is_file()
    assert marker not in capsys.readouterr().out


def test_main_prints_only_pid_on_success(tmp_path, monkeypatch, capsys):
    helper = load_helper()
    backend = make_backend(tmp_path)
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    monkeypatch.setattr(helper, "launch", lambda *args: 9876)

    assert helper.main(
        [
            "--backend-dir",
            str(backend),
            "--port",
            "18082",
            "--stdout",
            str(stdout_path),
            "--stderr",
            str(stderr_path),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "9876\n"
    assert captured.err == ""


def test_launch_refuses_existing_log_without_starting_process(tmp_path, monkeypatch):
    helper = load_helper()
    backend = make_backend(tmp_path)
    existing = tmp_path / "existing.log"
    existing.write_text("keep", encoding="utf-8")
    other = tmp_path / "other.log"
    called = False

    def fail_popen(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Popen must not run when a log already exists")

    monkeypatch.setattr(helper.subprocess, "Popen", fail_popen)
    with pytest.raises(helper.LaunchError, match="already exists"):
        helper.launch(str(backend), 18083, str(existing), str(other))
    assert not called
    assert existing.read_text(encoding="utf-8") == "keep"
