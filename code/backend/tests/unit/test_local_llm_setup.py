"""本地模型部署脚本、依赖和静态安全回归测试。"""

from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]


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
