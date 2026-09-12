import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_qdrant_targeted8_accuracy_20260902.ps1"
REPORT = ROOT / "reports" / "Qdrant定向8题运行说明_20260902.md"

CASE_IDS = {
    "real_yitai_2025_h1_operating_cash",
    "real_zhongcheng_2024_net_profit",
    "real_longyu_2024_revenue",
    "real_huawei_2024_revenue",
    "real_shenlian_2025_h1_rnd_ratio_synonym",
    "real_gree_2023_core_table_fields",
    "real_guangdao_full_summary_consistency",
    "real_furun_2024_adjusted_revenue_synonym",
}


SMOKE_BEGIN = "# --- BEGIN BackendSmokeOnly block ---"
SMOKE_END = "# --- END BackendSmokeOnly block ---"


def read_script() -> str:
    raw = SCRIPT.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "PowerShell 5.1脚本必须带UTF-8 BOM"
    return raw[3:].decode("utf-8")


def smoke_block() -> str:
    text = read_script()
    start = text.index(SMOKE_BEGIN)
    end = text.index(SMOKE_END, start)
    return text[start:end]


def test_script_is_frozen_targeted8_and_has_explicit_config():
    text = read_script()
    for case_id in CASE_IDS:
        assert f'"{case_id}"' in text
        assert text.count(f'"{case_id}"') == 1
    assert text.count('"real_') == len(CASE_IDS)
    required = {
        'KB_VECTOR_BACKEND = "qdrant"',
        'KB_QDRANT_URL = "http://127.0.0.1:16333"',
        'KB_QDRANT_COLLECTION = "kb_full_codex_20260830_24c7407e"',
        'KB_QDRANT_CREATE_IF_MISSING = "false"',
        'KB_EMBEDDING_MODEL = "bge-m3"',
        'KB_QDRANT_VECTOR_SIZE = "1024"',
        'KB_TOP_K = "5"',
        'KB_RECALL_K = "20"',
        'KB_MAX_HITS_PER_DOC = "12"',
        'KB_STRUCTURED_FIN_ROUTE = "1"',
        'KB_FIN_ROUTE_TOP_N = "8"',
        'KB_RERANK_MODE = "llm"',
        'KB_ANSWERABILITY_SCORE = "0.5"',
    }
    for line in required:
        assert line in text
    assert "run_qdrant_full36.py" not in text
    assert "run_qdrant_regression.py" not in text
    assert "--case-id" in text


def test_script_has_gate_first_unique_outputs_and_case_level_report_checks():
    text = read_script()
    assert "run_rag_anti_regression_gate.ps1" in text
    assert text.index("先执行现有离线防退门禁") < text.index("\n    Test-DockerAndQdrant")
    assert "New-AttemptId" in text
    assert "Assert-OutputUnused" in text
    for field in ("passed", "page_hit", "evidence_rate", "elapsed_s", "citation_kb_ids"):
        assert field in text
    assert 'Enter-RunEnvironment "0"' in text
    assert "AnswerabilityZeroDiagnostic" in text
    assert "Assert-DiagnosticCaseFailed" in text
    assert "passed -eq $ExpectedCaseIds.Count" not in text
    assert "passed -eq 8" not in text


def test_docker_is_advisory_but_qdrant_http_is_hard_gate():
    text = read_script()
    start = text.index("function Test-DockerAndQdrant")
    http_gate = text.index("    $uri =", start)
    docker_part = text[start:http_gate]
    qdrant_part = text[http_gate:text.index("function Enter-RunEnvironment", http_gate)]
    assert "Docker传输层不检查，以Qdrant HTTP为硬门槛" in docker_part
    assert "docker.exe" not in docker_part
    assert "docker.Source" not in docker_part
    assert "& $" not in docker_part
    assert "throw" not in docker_part
    assert "Qdrant集合只读检查失败" in qdrant_part
    assert "集合状态不是green" in qdrant_part
    assert "向量维度不符" in qdrant_part


def test_backend_launch_uses_python_helper_and_not_start_process():
    text = read_script()
    assert "Start-Process" not in text
    assert "Normalize-ProcessEnvironmentKeys" not in text
    assert "launch_isolated_backend.py" in text
    start = text.index("function Start-IsolatedBackend")
    stop = text.index("function Stop-IsolatedBackend", start)
    start_body = text[start:stop]
    assert "& $script:PythonPath $helperPath" in start_body
    assert "--backend-dir" in start_body
    assert "--stdout" in start_body and "--stderr" in start_body
    assert "Get-Process -Id $backendPid" in start_body
    assert start_body.index("& $script:PythonPath $helperPath") < start_body.index("Get-Process -Id $backendPid")
    assert start_body.index("Get-Process -Id $backendPid") < start_body.index("Wait-BackendReady")


def test_evaluator_receives_absolute_script_and_testset_paths():
    text = read_script()
    start = text.index("function Invoke-Evaluation")
    end = text.index("function Get-RagReport", start)
    block = text[start:end]
    assert '$script:EvaluatorPath, "--rag", $script:TestsetPath' in block
    assert '"--output", $ReportPath' in block
    assert '"testsets/rag_real_quality_v2.yaml"' not in text


def test_run_note_states_scope_and_no_side_effects():
    if not REPORT.is_file():
        pytest.skip(
            f"运行说明 {REPORT.name} 属于本地复现资产，未随公开仓库提供"
        )
    note = REPORT.read_text(encoding="utf-8")
    for case_id in CASE_IDS:
        assert case_id in note
    for phrase in ("不运行36题", "不写入Qdrant", "error_count=0", "KB_ANSWERABILITY_SCORE=0"):
        assert phrase in note


def main_block() -> str:
    """冒烟块之后、catch 之前的正式评测路径。"""
    text = read_script()
    start = text.index(SMOKE_END)
    end = text.index("} catch {", start)
    return text[start:end]


def cleanup_body() -> str:
    """Stop-IsolatedBackend 函数体。"""
    text = read_script()
    start = text.index("function Stop-IsolatedBackend")
    return text[start:text.index("function Invoke-Evaluation", start)]


def test_backend_smoke_only_never_reaches_evaluation_or_report():
    block = smoke_block()
    assert "Start-IsolatedBackend" in block
    for forbidden in (
        "Invoke-Evaluation",
        "Assert-TargetedReport",
        "Assert-DiagnosticCaseFailed",
        "Test-CaseSet",
        "evaluate_quality.py",
        "--rag",
    ):
        assert forbidden not in block, f"冒烟模式不得触碰：{forbidden}"
    for case_id in CASE_IDS:
        assert case_id not in block
    assert "exit 0" in block


def test_backend_smoke_only_runs_checks_before_launch_and_releases_port():
    block = smoke_block()
    order = [
        block.index("Test-RequiredConfiguration"),
        block.index("Test-PythonDependencies"),
        block.index("Test-PortFree"),
        block.index("Test-DockerAndQdrant"),
        block.index('Enter-RunEnvironment "0.5"'),
        block.index("Start-IsolatedBackend"),
        block.index("Stop-IsolatedBackend"),
        block.index("Exit-RunEnvironment"),
        block.index("Wait-PortReleased"),
    ]
    assert order == sorted(order), "冒烟模式调用顺序被打乱"
    assert block.count("Start-IsolatedBackend") == 1, "后端只能启动一次"


def test_backend_smoke_only_cleans_up_in_finally():
    block = smoke_block()
    try_index = block.index("try {")
    finally_index = block.index("} finally {", try_index)
    finally_body = block[finally_index:block.index("}", block.index("Exit-RunEnvironment", finally_index))]
    assert "Stop-IsolatedBackend" in finally_body
    assert "Exit-RunEnvironment" in finally_body
    # 清理与恢复必须在 try 体内部，且 ready 检查发生在 try 之前由 helper 完成
    assert finally_index > block.index("Start-IsolatedBackend")


def test_cleanup_targets_only_helper_pid_and_never_kills_port_owner():
    body = cleanup_body()
    assert "$script:BackendProcess.Id" in body
    for forbidden in ("Get-PortListeners", "Get-NetTCPConnection", "OwningProcess", "Stop-Process -Name"):
        assert forbidden not in body, f"清理不得按端口/进程名查杀：{forbidden}"


def test_smoke_mode_does_not_reserve_report_path():
    SMOKE_GUARD = "$guardedPaths = @($script:LogPath, $mainStdoutPath, $mainStderrPath)"
    text = read_script()
    # 冒烟分支只守护 runner 日志与本次后端 stdout/stderr，不预留任何评测JSON路径
    start = text.index("$script:SmokeOnly = ")
    block = text[start:text.index("Assert-OutputUnused $guardedPaths", start)]
    assert SMOKE_GUARD in block
    smoke_branch = block[block.index(SMOKE_GUARD):]
    smoke_branch = smoke_branch[:smoke_branch.index("\n    } else {")]
    assert "mainReportPath" not in smoke_branch, "冒烟模式不得预留评测JSON路径"
    assert "diagnosticReportPath" not in smoke_branch
    else_branch = block[block.index("\n    } else {"):]
    assert "mainReportPath" in else_branch, "正式模式仍必须保留报告路径守护"
    # 冒烟与正式使用不同 attempt 前缀，日志与产物天然可区分
    assert '"qdrant-backend-smoke-20260902"' in text
    assert '"qdrant-targeted8-accuracy-20260902"' in text


def test_smoke_mode_rejects_diagnostic_combination():
    text = read_script()
    assert "$BackendSmokeOnly -and $AnswerabilityZeroDiagnostic" in text
    assert "-AnswerabilityZeroDiagnostic" in text


def test_full_eight_case_path_is_untouched_after_smoke_block():
    body = main_block()
    assert "Test-CaseSet" in body
    assert "Invoke-Evaluation" in body
    assert "Assert-TargetedReport" in body
    assert 'Enter-RunEnvironment "0.5"' in body


def test_cleanup_also_stops_uv_shim_child_but_never_port_owner():
    """helper 返回的可能是 uv 外壳 PID，真正的 uvicorn 是其 python 子进程。"""
    body = cleanup_body()
    assert "ParentProcessId = $targetPid" in body
    assert '$_.Name -eq "python.exe"' in body
    assert "Stop-Process -Id $targetPid -Force" in body
    assert "Stop-Process -Id $childPid -Force" in body
    for forbidden in ("Get-PortListeners", "Get-NetTCPConnection", "OwningProcess", "Stop-Process -Name"):
        assert forbidden not in body, f"清理不得按端口/进程名查杀：{forbidden}"


def test_stop_suppresses_waitforexit_boolean_output():
    """WaitForExit(int) 返回 bool，未接收会污染 runner 日志与调用方输出。"""
    body = cleanup_body()
    assert "$null = $script:BackendProcess.WaitForExit(10000)" in body
    assert "$script:BackendProcess.WaitForExit" in body
    for line in body.splitlines():
        if "WaitForExit" in line:
            assert line.strip().startswith("$null ="), f"WaitForExit 返回值必须被接收：{line.strip()}"


def test_no_cjk_variable_interpolation_hazard():
    """$Port已释放 会被 PS 解析成变量名 $Port已释放，导致插值丢失。"""
    text = read_script()
    hazards = re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?[\u4e00-\u9fff]", text)
    assert not hazards, f"存在变量名吞掉中文的插值隐患：{set(hazards)}"


def test_script_stays_powershell_51_compatible_with_bom():
    text = read_script()
    for ps7_only in ("??", "?.", "&&", "||", "-Parallel", "$PSStyle", "Get-Error", "ForEach-Object -Parallel"):
        assert ps7_only not in text, f"PS 5.1 不兼容的构造：{ps7_only}"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert not stripped.startswith("using "), "using 语句需置于脚本最前，易破坏 PS 5.1"
