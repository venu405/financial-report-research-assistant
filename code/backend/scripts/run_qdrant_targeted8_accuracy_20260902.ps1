[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [int]$Port = 18080,
    [int]$StartupTimeoutSeconds = 60,
    [switch]$AnswerabilityZeroDiagnostic,
    [string]$DiagnosticCaseId = "",
    [switch]$BackendSmokeOnly
)

$ErrorActionPreference = "Stop"
$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ScriptsDir = Split-Path -Parent $script:ScriptPath
$script:BackendDir = Split-Path -Parent $script:ScriptsDir
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $script:BackendDir)
$script:ReportsDir = Join-Path $script:BackendDir "reports"
$script:TestsetPath = Join-Path $script:BackendDir "testsets\rag_real_quality_v2.yaml"
$script:AuditPath = Join-Path $script:ReportsDir "Qdrant准确率提升_配置与失败题审计_20260902.md"
$script:GatePath = Join-Path $script:ScriptsDir "run_rag_anti_regression_gate.ps1"
$script:EvaluatorPath = Join-Path $script:ScriptsDir "evaluate_quality.py"

# This list is deliberately frozen to the audit's eight cases.  The evaluator
# receives one --case-id per item; no full36/regression launcher is called.
$script:CaseIds = @(
    "real_yitai_2025_h1_operating_cash",
    "real_zhongcheng_2024_net_profit",
    "real_longyu_2024_revenue",
    "real_huawei_2024_revenue",
    "real_shenlian_2025_h1_rnd_ratio_synonym",
    "real_gree_2023_core_table_fields",
    "real_guangdao_full_summary_consistency",
    "real_furun_2024_adjusted_revenue_synonym"
)

$script:FixedConfig = [ordered]@{
    KB_VECTOR_BACKEND = "qdrant"
    KB_QDRANT_URL = "http://127.0.0.1:16333"
    KB_QDRANT_COLLECTION = "kb_full_codex_20260830_24c7407e"
    KB_QDRANT_VECTOR_SIZE = "1024"
    KB_QDRANT_CREATE_IF_MISSING = "false"
    KB_EMBEDDING_MODE = "bge_m3"
    KB_EMBEDDING_MODEL = "bge-m3"
    KB_OLLAMA_HOST = "http://127.0.0.1:11434"
    KB_TOP_K = "5"
    KB_RECALL_K = "20"
    KB_MAX_HITS_PER_DOC = "12"
    KB_STRUCTURED_FIN_ROUTE = "1"
    KB_FIN_ROUTE_TOP_N = "8"
    KB_RERANK_MODE = "llm"
    KB_ANSWERABILITY_SCORE = "0.5"
}

$script:EnvironmentBackup = @{}
$script:EnvironmentEntered = $false
$script:BackendProcess = $null
$script:LogPath = $null
$script:SmokeOnly = $false

function Write-RunnerLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString("o")), $Message
    Write-Host $line
    if ($script:LogPath) {
        Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    }
}

function Get-DotEnvValue {
    param([string]$Name)
    $envPath = Join-Path $script:BackendDir ".env"
    if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
        return $null
    }
    foreach ($line in (Get-Content -LiteralPath $envPath -Encoding UTF8)) {
        if ($line -match ("^\s*" + [regex]::Escape($Name) + "\s*=\s*(.*?)\s*$")) {
            $value = $Matches[1].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($value) { return $value }
        }
    }
    return $null
}

function Get-ConfiguredValue {
    param([string]$Name)
    $processValue = [Environment]::GetEnvironmentVariable($Name, [EnvironmentVariableTarget]::Process)
    if ($processValue) { return $processValue }
    return Get-DotEnvValue $Name
}

function Test-RequiredConfiguration {
    $required = @("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID")
    foreach ($name in $required) {
        if (-not (Get-ConfiguredValue $name)) {
            throw "必要配置缺失：$name（仅检查变量是否存在，未输出值）"
        }
        Write-RunnerLog "配置已设置：$name（值已隐藏）"
    }
    Write-RunnerLog "Embedding配置固定：KB_EMBEDDING_MODE=bge_m3、KB_EMBEDDING_MODEL=bge-m3、KB_OLLAMA_HOST已固定"
}

function Resolve-Python {
    if ($PythonPath) {
        $candidate = $PythonPath
    } else {
        $candidate = Join-Path $script:BackendDir ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $command = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($command) { $candidate = $command.Source }
        }
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Python executable does not exist: $candidate"
    }
    $version = & $candidate --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python preflight failed: $version" }
    Write-RunnerLog "Python可用：$version"
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Test-PythonDependencies {
    $probe = "import fastapi, httpx, uvicorn, yaml"
    $null = & $script:PythonPath -c $probe 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python必要依赖缺失：fastapi/httpx/uvicorn/yaml"
    }
    Write-RunnerLog "Python必要依赖可导入（未调用模型或向量库）"
}

function Test-CaseSet {
    if (-not (Test-Path -LiteralPath $script:AuditPath -PathType Leaf)) {
        throw "审计报告不存在：$script:AuditPath"
    }
    if (-not (Test-Path -LiteralPath $script:TestsetPath -PathType Leaf)) {
        throw "题集不存在：$script:TestsetPath"
    }
    $testset = Get-Content -LiteralPath $script:TestsetPath -Raw -Encoding UTF8
    foreach ($caseId in $script:CaseIds) {
        if ($testset -notmatch ("(?m)^\s*-\s+id:\s*" + [regex]::Escape($caseId) + "\s*$")) {
            throw "题集缺少审计指定case：$caseId"
        }
    }
    Write-RunnerLog "题集静态检查通过：固定8题，不扩展为36题"
}

function Get-PortListeners {
    $tcpCommand = Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue
    if ($tcpCommand) {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    }
    $netstat = netstat.exe -ano -p tcp 2>$null
    return @($netstat | Where-Object { $_ -match (":" + $Port + "\s+\S+\s+LISTENING\s+") })
}

function Test-PortFree {
    if ((Get-PortListeners).Count -gt 0) {
        throw "隔离端口已被占用：$Port；不会接管或停止外部进程"
    }
    Write-RunnerLog "隔离端口可用：$Port"
}

function Wait-PortReleased {
    param([int]$TimeoutSeconds = 15)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((Get-PortListeners).Count -eq 0) {
            Write-RunnerLog "隔离端口已释放：$Port"
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "隔离端口在停止后端后仍未释放：$Port；未尝试停止任何外部进程"
}

function Test-DockerAndQdrant {
    Write-RunnerLog "Docker传输层不检查，以Qdrant HTTP为硬门槛"

    $uri = $script:FixedConfig.KB_QDRANT_URL.TrimEnd('/') + "/collections/" +
        [Uri]::EscapeDataString($script:FixedConfig.KB_QDRANT_COLLECTION)
    $headers = @{}
    $apiKey = Get-ConfiguredValue "KB_QDRANT_API_KEY"
    if ($apiKey) { $headers["api-key"] = $apiKey }
    try {
        $collection = Invoke-RestMethod -Uri $uri -Method Get -Headers $headers -TimeoutSec 5
    } catch {
        throw "Qdrant集合只读检查失败：$($script:FixedConfig.KB_QDRANT_COLLECTION)；$($_.Exception.Message)"
    }
    $status = $collection.result.status
    if ($status -and $status -ne "green") {
        throw "Qdrant集合状态不是green：实际$status"
    }
    if (-not $status) {
        Write-RunnerLog "WARNING: Qdrant API未返回status字段；按API结构兼容继续维度检查"
    }
    $size = $collection.result.config.params.vectors.size
    if ($null -eq $size -or [int]$size -ne 1024) {
        throw "Qdrant集合向量维度不符：期望1024，实际$size"
    }
    Write-RunnerLog "Qdrant集合存在且向量维度为1024（仅GET，无写入）"
}

function Enter-RunEnvironment {
    param([string]$AnswerabilityScore)
    if ($script:EnvironmentEntered) { return }
    $script:EnvironmentBackup = @{}
    foreach ($name in $script:FixedConfig.Keys) {
        $script:EnvironmentBackup[$name] = [Environment]::GetEnvironmentVariable($name, [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable($name, $script:FixedConfig[$name], [EnvironmentVariableTarget]::Process)
    }
    [Environment]::SetEnvironmentVariable("KB_ANSWERABILITY_SCORE", $AnswerabilityScore, [EnvironmentVariableTarget]::Process)
    $script:EnvironmentBackup["PYTHONPATH"] = [Environment]::GetEnvironmentVariable("PYTHONPATH", [EnvironmentVariableTarget]::Process)
    $src = Join-Path $script:BackendDir "src"
    $oldPythonPath = $script:EnvironmentBackup["PYTHONPATH"]
    $newPythonPath = if ($oldPythonPath) { $src + [IO.Path]::PathSeparator + $oldPythonPath } else { $src }
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $newPythonPath, [EnvironmentVariableTarget]::Process)
    $script:EnvironmentEntered = $true
}

function Exit-RunEnvironment {
    if (-not $script:EnvironmentEntered) { return }
    foreach ($name in $script:EnvironmentBackup.Keys) {
        [Environment]::SetEnvironmentVariable($name, $script:EnvironmentBackup[$name], [EnvironmentVariableTarget]::Process)
    }
    $script:EnvironmentBackup = @{}
    $script:EnvironmentEntered = $false
}

function Wait-BackendReady {
    param([System.Diagnostics.Process]$Process)
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $lastError = "无响应"
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Process.HasExited) { throw "隔离后端在readyz前退出：$($Process.ExitCode)" }
        try {
            $ready = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/readyz" -f $Port) -Method Get -TimeoutSec 5
            if ($ready.status -eq "ok") {
                Write-RunnerLog "隔离后端readyz=ok，PID=$($Process.Id)"
                return
            }
            $lastError = "status=$($ready.status)"
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    throw "隔离后端readyz超时：$lastError"
}

function Start-IsolatedBackend {
    param([string]$StdoutPath, [string]$StderrPath)
    if ((Test-Path -LiteralPath $StdoutPath -PathType Leaf) -or (Test-Path -LiteralPath $StderrPath -PathType Leaf)) {
        throw "后端日志文件已存在，拒绝覆盖"
    }
    $helperPath = Join-Path $script:ScriptsDir "launch_isolated_backend.py"
    if (-not (Test-Path -LiteralPath $helperPath -PathType Leaf)) {
        throw "后端启动helper不存在：$helperPath"
    }
    $helperOutput = & $script:PythonPath $helperPath `
        --backend-dir $script:BackendDir `
        --port ([string]$Port) `
        --stdout $StdoutPath `
        --stderr $StderrPath 2>&1
    $helperExit = $LASTEXITCODE
    if ($helperExit -ne 0) {
        foreach ($line in $helperOutput) { Write-RunnerLog ([string]$line) }
        throw "后端启动helper失败：exit=$helperExit"
    }
    $pidLines = @($helperOutput | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
    $backendPid = 0
    if ($pidLines.Count -ne 1 -or -not [int]::TryParse($pidLines[0], [ref]$backendPid) -or $backendPid -le 0) {
        throw "后端启动helper未返回单一有效PID"
    }
    try {
        $script:BackendProcess = Get-Process -Id $backendPid -ErrorAction Stop
    } catch {
        throw "后端启动helper返回的PID不存在：$backendPid"
    }
    Wait-BackendReady -Process $script:BackendProcess
}

function Stop-IsolatedBackend {
    if ($null -eq $script:BackendProcess) { return }
    $targetPid = $script:BackendProcess.Id
    # uv 的 venv python.exe 是外壳启动器：helper 返回的是外壳 PID，真正的
    # uvicorn 运行在它派生的 python 子进程里。只杀外壳 PID 会把监听进程
    # 留成孤儿，继续占用隔离端口并让后续运行的 Test-PortFree 硬失败。
    # 因此这里连带清理「本次启动进程」的 python 子进程，仍然绝不查杀端口占用者。
    $childPids = @()
    if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
        $childPids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $targetPid" -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessId -ne $targetPid -and $_.Name -eq "python.exe" } |
            ForEach-Object { $_.ProcessId })
    }
    try {
        foreach ($childPid in $childPids) {
            try {
                Stop-Process -Id $childPid -Force -ErrorAction Stop
                Write-RunnerLog "停止本次后端外壳派生的子进程PID=$childPid"
            } catch {
                Write-RunnerLog "后端子进程清理失败（未尝试外部进程）：$($_.Exception.Message)"
            }
        }
        if (-not $script:BackendProcess.HasExited) {
            Stop-Process -Id $targetPid -Force -ErrorAction Stop
            $null = $script:BackendProcess.WaitForExit(10000)
            Write-RunnerLog "仅停止本次启动的后端PID=$targetPid"
        }
    } catch {
        Write-RunnerLog "后端清理失败（未尝试外部进程）：$($_.Exception.Message)"
        throw
    } finally {
        $script:BackendProcess = $null
    }
}

function Invoke-Evaluation {
    param(
        [string]$ReportPath,
        [string]$Label,
        [string[]]$SelectedCaseIds
    )
    if (Test-Path -LiteralPath $ReportPath -PathType Leaf) {
        throw "输出JSON已存在，拒绝覆盖：$ReportPath"
    }
    $arguments = @(
        $script:EvaluatorPath, "--rag", $script:TestsetPath,
        "--base-url", ("http://127.0.0.1:{0}" -f $Port),
        "--label", $Label, "--output", $ReportPath
    )
    foreach ($caseId in $SelectedCaseIds) { $arguments += @("--case-id", $caseId) }
    Write-RunnerLog "调用现有评测器：固定$($SelectedCaseIds.Count)题；报告=$ReportPath"
    $output = & $script:PythonPath @arguments 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) { Write-RunnerLog ([string]$line) }
    Write-RunnerLog "评测器退出码：$exitCode（逐题passed由报告保留，不以总passed作为本门禁阈值）"
    return [int]$exitCode
}

function Get-RagReport {
    param([string]$ReportPath)
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        throw "评测未生成报告：$ReportPath"
    }
    try {
        $payload = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "评测报告JSON损坏：$ReportPath；$($_.Exception.Message)"
    }
    $reports = @($payload.reports)
    $ragReports = @($reports | Where-Object { $_.kind -eq "rag" })
    if ($ragReports.Count -ne 1) { throw "报告必须包含且仅包含一个kind=rag报告" }
    return $ragReports[0]
}

function Assert-TargetedReport {
    param(
        [string]$ReportPath,
        [string[]]$ExpectedCaseIds
    )
    $report = Get-RagReport $ReportPath
    if ($report.run_status -ne "completed") { throw "报告run_status不是completed：$($report.run_status)" }
    if ($null -eq $report.summary -or $report.summary.error_count -ne 0) {
        throw "报告error_count不是0：$($report.summary.error_count)"
    }
    $selected = @($report.selected_case_ids | ForEach-Object { [string]$_ })
    $expectedSorted = @($ExpectedCaseIds | Sort-Object)
    $selectedSorted = @($selected | Sort-Object)
    if ($selected.Count -ne $ExpectedCaseIds.Count -or (@($selectedSorted) -join "|") -ne (@($expectedSorted) -join "|")) {
        throw "报告selected_case_ids不匹配固定题集"
    }
    $results = @($report.results)
    if ($results.Count -ne $ExpectedCaseIds.Count -or $report.total -ne $ExpectedCaseIds.Count -or
        $report.planned_total -ne $ExpectedCaseIds.Count -or $report.completed_total -ne $ExpectedCaseIds.Count) {
        throw "报告未完成固定题数：total=$($report.total), planned=$($report.planned_total), completed=$($report.completed_total)"
    }
    $expectedSet = @{}; foreach ($id in $ExpectedCaseIds) { $expectedSet[$id] = $true }
    $seen = @{}
    $failedCount = 0
    foreach ($result in $results) {
        $id = [string]$result.id
        if (-not $expectedSet.ContainsKey($id) -or $seen.ContainsKey($id)) { throw "报告包含未知或重复case：$id" }
        $seen[$id] = $true
        if ($null -eq $result.passed) { throw "case缺少passed字段：$id" }
        if ($null -eq $result.retrieval.contexts) { throw "case缺少retrieval.contexts：$id" }
        if ($null -eq $result.retrieval.contexts.page_hit -or $null -eq $result.retrieval.contexts.evidence_rate) {
            throw "case缺少page_hit/evidence_rate：$id"
        }
        if ($null -eq $result.elapsed_s -or $null -eq $result.citation_kb_ids) {
            throw "case缺少elapsed_s/citation_kb_ids：$id"
        }
        $citationIds = @($result.citation_kb_ids | ForEach-Object { [string]$_ })
        $crossKb = @($citationIds | Where-Object { $_ -ne "cninfo_report" })
        if ($crossKb.Count -gt 0) { throw "case存在跨KB引用：$id -> $($crossKb -join ',')" }
        if ($result.passed -eq $false) { $failedCount++ }
        Write-RunnerLog ("case={0}; passed={1}; page_hit={2}; evidence_rate={3}; elapsed_s={4}; citation_kb_ids={5}" -f `
            $id, $result.passed, $result.retrieval.contexts.page_hit, $result.retrieval.contexts.evidence_rate, `
            $result.elapsed_s, ($citationIds -join ","))
    }
    if ($seen.Count -ne $ExpectedCaseIds.Count) { throw "报告缺少固定case" }
    Write-RunnerLog "报告完整性通过：$($ExpectedCaseIds.Count)/$($ExpectedCaseIds.Count) completed，error_count=0；failed=$failedCount"
    return $report
}

function Assert-DiagnosticCaseFailed {
    param([object]$Report, [string]$CaseId)
    $result = @($Report.results | Where-Object { $_.id -eq $CaseId })
    if ($result.Count -ne 1 -or $result[0].passed -ne $false) {
        throw "0门限诊断只允许主报告中明确failed的单个case：$CaseId"
    }
}

function New-AttemptId {
    $prefix = if ($script:SmokeOnly) { "qdrant-backend-smoke-20260902" } else { "qdrant-targeted8-accuracy-20260902" }
    return ("{0}-{1}-{2}" -f $prefix, `
        [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ"),
        ([Guid]::NewGuid().ToString("N").Substring(0, 10)))
}

function Assert-OutputUnused {
    param([string[]]$Paths)
    foreach ($path in $Paths) {
        if (Test-Path -LiteralPath $path) { throw "输出或日志已存在，拒绝覆盖：$path" }
    }
}

try {
    if ($Port -lt 1 -or $Port -gt 65535 -or $Port -eq 8000) { throw "Port必须在1..65535且不能使用8000" }
    if ($StartupTimeoutSeconds -le 0) { throw "StartupTimeoutSeconds必须为正数" }
    if ($AnswerabilityZeroDiagnostic -xor [bool]$DiagnosticCaseId) {
        throw "0门限诊断必须同时提供-AnswerabilityZeroDiagnostic和-DiagnosticCaseId"
    }
    if ($DiagnosticCaseId -and $script:CaseIds -notcontains $DiagnosticCaseId) {
        throw "诊断case必须是固定8题之一：$DiagnosticCaseId"
    }
    if ($BackendSmokeOnly -and $AnswerabilityZeroDiagnostic) {
        throw "BackendSmokeOnly只验证隔离后端启动链路，不能与-AnswerabilityZeroDiagnostic同时使用"
    }
    $script:SmokeOnly = [bool]$BackendSmokeOnly

    $attemptId = New-AttemptId
    $mainReportPath = Join-Path $script:ReportsDir ($attemptId + ".json")
    $script:LogPath = Join-Path $script:ReportsDir ($attemptId + ".runner.log")
    $mainStdoutPath = Join-Path $script:ReportsDir ($attemptId + ".backend.stdout.log")
    $mainStderrPath = Join-Path $script:ReportsDir ($attemptId + ".backend.stderr.log")
    $diagnosticReportPath = Join-Path $script:ReportsDir ($attemptId + "-answerability0-" + $DiagnosticCaseId + ".json")
    $diagnosticStdoutPath = Join-Path $script:ReportsDir ($attemptId + ".diagnostic.backend.stdout.log")
    $diagnosticStderrPath = Join-Path $script:ReportsDir ($attemptId + ".diagnostic.backend.stderr.log")
    New-Item -ItemType Directory -Path $script:ReportsDir -Force | Out-Null
    if ($script:SmokeOnly) {
        # Smoke mode only writes the runner log and this launch's backend logs.
        # No report path is reserved, so an evaluation JSON cannot be produced.
        $guardedPaths = @($script:LogPath, $mainStdoutPath, $mainStderrPath)
    } else {
        $guardedPaths = @($script:LogPath, $mainReportPath, $mainStdoutPath, $mainStderrPath, $diagnosticReportPath, $diagnosticStdoutPath, $diagnosticStderrPath)
    }
    Assert-OutputUnused $guardedPaths
    Set-Content -LiteralPath $script:LogPath -Value "" -Encoding UTF8
    Write-RunnerLog "attempt_id=$attemptId"
    $mode = if ($script:SmokeOnly) { "BackendSmokeOnly（只验证隔离后端启动链路）" } else { "固定8题评测" }
    Write-RunnerLog "运行模式：$mode"
    $script:PythonPath = Resolve-Python

    # The existing offline gate is deliberately first; no Qdrant/backend check
    # or model-facing process is started until this gate succeeds.
    if (-not (Test-Path -LiteralPath $script:GatePath -PathType Leaf)) { throw "离线防退门禁不存在：$script:GatePath" }
    Write-RunnerLog "先执行现有离线防退门禁"
    $gateOutput = & (Join-Path $PSHOME "powershell.exe") -NoProfile -ExecutionPolicy Bypass -File $script:GatePath 2>&1
    $gateExit = $LASTEXITCODE
    foreach ($line in $gateOutput) { Write-RunnerLog ([string]$line) }
    if ($gateExit -ne 0) { throw "离线防退门禁失败，停止后续Qdrant评测：exit=$gateExit" }
    Write-RunnerLog "离线防退门禁通过"

    # --- BEGIN BackendSmokeOnly block ---
    if ($script:SmokeOnly) {
        Write-RunnerLog "BackendSmokeOnly：只验证启动→readyz→停止，不调用评测器、不生成题集JSON、不调用LLM或embedding推理"
        Test-RequiredConfiguration
        Test-PythonDependencies
        Test-PortFree
        Test-DockerAndQdrant
        Enter-RunEnvironment "0.5"
        try {
            Start-IsolatedBackend -StdoutPath $mainStdoutPath -StderrPath $mainStderrPath
        } finally {
            Stop-IsolatedBackend
            Exit-RunEnvironment
        }
        Wait-PortReleased
        Write-RunnerLog ("BackendSmokeOnly通过：后端仅启动一次、readyz=ok、本次PID已停止、端口" + $Port + "已释放；未生成任何评测JSON")
        exit 0
    }
    # --- END BackendSmokeOnly block ---

    Test-CaseSet
    Test-RequiredConfiguration
    Test-PythonDependencies
    Test-PortFree
    Test-DockerAndQdrant

    Enter-RunEnvironment "0.5"
    try {
        Start-IsolatedBackend -StdoutPath $mainStdoutPath -StderrPath $mainStderrPath
        $evalExit = Invoke-Evaluation -ReportPath $mainReportPath `
            -Label $attemptId -SelectedCaseIds $script:CaseIds
    } finally {
        Stop-IsolatedBackend
        Exit-RunEnvironment
    }
    $mainReport = Assert-TargetedReport -ReportPath $mainReportPath -ExpectedCaseIds $script:CaseIds
    if ($evalExit -ge 2) { throw "主评测基础设施/评测错误，报告已保留：exit=$evalExit" }

    if ($AnswerabilityZeroDiagnostic) {
        Assert-DiagnosticCaseFailed -Report $mainReport -CaseId $DiagnosticCaseId
        Test-PortFree
        Enter-RunEnvironment "0"
        try {
            Start-IsolatedBackend -StdoutPath $diagnosticStdoutPath -StderrPath $diagnosticStderrPath
            $diagnosticExit = Invoke-Evaluation -ReportPath $diagnosticReportPath `
                -Label ($attemptId + "-answerability0-" + $DiagnosticCaseId) `
                -SelectedCaseIds @($DiagnosticCaseId)
        } finally {
            Stop-IsolatedBackend
            Exit-RunEnvironment
        }
        $null = Assert-TargetedReport -ReportPath $diagnosticReportPath -ExpectedCaseIds @($DiagnosticCaseId)
        if ($diagnosticExit -ge 2) { throw "0门限诊断基础设施/评测错误，报告已保留：exit=$diagnosticExit" }
        Write-RunnerLog "0门限诊断已单独完成，不计入主结果：$diagnosticReportPath"
    }
    Write-RunnerLog "完成：固定8题已完成且error_count=0；逐题passed请结合报告归因"
    exit 0
} catch {
    try { Stop-IsolatedBackend } catch { }
    try { Exit-RunEnvironment } catch { }
    if ($script:LogPath) { Write-RunnerLog ("失败：" + $_.Exception.Message) }
    Write-Error $_.Exception.Message
    exit 2
}
