"""本地模型部署脚本、依赖和 Windows 启动入口的静态安全回归测试。"""

import os
import subprocess
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND.parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_fastembed_is_a_formal_dependency_and_onnxruntime_is_retained():
    pyproject = _read(BACKEND / "pyproject.toml")
    requirements = _read(BACKEND / "requirements.txt")

    assert '"fastembed>=0.7.4,<0.9.0"' in pyproject
    assert "fastembed>=0.7.4,<0.9.0" in requirements
    assert '"onnxruntime>=1.19"' in pyproject
    assert "onnxruntime>=1.19" in requirements


def test_modelfile_pins_the_supported_4b_context():
    modelfile = _read(BACKEND / "Modelfile.qwen35-4b")

    assert modelfile.startswith("FROM qwen3.5:4b")
    assert "PARAMETER num_ctx 8192" in modelfile
    assert "9b" not in modelfile.lower()


def test_setup_script_is_bom_encoded_idempotent_and_runs_both_smokes():
    path = BACKEND / "scripts" / "setup_local_llm.ps1"
    raw = path.read_bytes()
    script = _read(path)

    assert raw.startswith(b"\xef\xbb\xbf")
    assert "ollama.exe" in script
    assert 'list 2>&1' in script
    assert 'pull", $baseModel' in script
    assert 'create", $aliasModel' in script
    assert "fastembed>=0.7.4,<0.9.0" in script
    assert "TextCrossEncoder" in script
    assert "BAAI/bge-reranker-base" in script
    assert "/v1/chat/completions" in script
    assert 'reasoning_effort = "none"' in script
    assert "UTF8.GetBytes" in script
    assert 'ArgumentList @("ps")' in script
    assert "LOCAL_LLM_OK" in script
    assert "exit 1" in script
    assert "LLM_API_KEY" not in script
    assert "your-api-key" not in script.lower()


def test_local_deployment_cmd_is_utf8_synchronous_and_forwards_arguments():
    cmd = _read(PROJECT_ROOT / "一键部署本地模型.cmd")

    assert "chcp 65001 >nul" in cmd
    assert 'set "ROOT_DIR=%~dp0"' in cmd
    assert 'call powershell.exe' in cmd.lower()
    assert "setup_local_llm.ps1" in cmd
    assert '"%SCRIPT%" %*' in cmd
    assert "start " not in cmd.lower()
    assert "pause" not in cmd.lower()
    assert "taskkill" not in cmd.lower()
    assert "exit /b" in cmd.lower()


def test_local_deployment_cmd_really_executes_under_cmd_without_parser_errors():
    """用 PowerShell -? 做无副作用冒烟，捕捉 CMD 截断命令的真实回归。"""
    if os.name != "nt":
        return

    path = PROJECT_ROOT / "一键部署本地模型.cmd"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "call", str(path), "-?"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "not recognized" not in combined.lower()
    assert "[INFO]" in combined


def test_local_start_cmd_sets_only_child_process_local_model_and_rejects_port_conflict():
    cmd = _read(PROJECT_ROOT / "一键启动本地模型版.cmd")

    for setting in (
        "LLM_BASE_URL=http://127.0.0.1:11434/v1",
        "LLM_API_KEY=ollama",
        "LLM_MODEL_ID=enterprise-kb-qwen35:4b",
        "LLM_REASONING_EFFORT=none",
        "KB_RERANK_MODE=crossencoder",
        "KB_RERANK_MODEL=BAAI/bge-reranker-base",
        "LLM_TIMEOUT=180",
    ):
        assert f'set "{setting}"' in cmd
    assert "Connect('127.0.0.1',8000)" in cmd
    assert "无法保证" in cmd and "本地模型" in cmd
    assert "一键启动前后端.cmd" in cmd
    assert '"%ROOT_DIR%\一键启动前后端.cmd" %*' in cmd
    assert "chcp 65001 >nul" in cmd
    assert 'set "ROOT_DIR=%~dp0"' in cmd
    assert "start " not in cmd.lower()
    assert "taskkill" not in cmd.lower()
    assert "pause" not in cmd.lower()


def test_env_and_ops_explain_switchback_limits_and_full_evaluation():
    env_example = _read(BACKEND / ".env.example")
    ops = _read(BACKEND / "OPS.md")

    for text in (
        "enterprise-kb-qwen35:4b",
        "LLM_REASONING_EFFORT=none",
        "KB_RERANK_MODE=crossencoder",
        "BAAI/bge-reranker-base",
    ):
        assert text in env_example or text in ops
    for text in (
        "8GB",
        "3.4GB",
        "9B",
        "ollama ps",
        "36",
        "回滚",
    ):
        assert text in ops
