[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [int]$Port = 18080,
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ScriptsDir = Split-Path -Parent $script:ScriptPath
$script:BackendDir = Split-Path -Parent $script:ScriptsDir
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $script:BackendDir)
$script:ReportsDir = Join-Path $script:BackendDir "reports"
$script:TestsetPath = Join-Path $script:BackendDir "testsets\rag_real_quality_v2.yaml"
$script:GatePath = Join-Path $script:ScriptsDir "run_rag_anti_regression_gate.ps1"
$script:EvaluatorPath = Join-Path $script:ScriptsDir "evaluate_quality.py"

# 本次只追踪正式 8 题中失败的 5 道；不扩展为 8 题或 36 题，也不做阈值 0 诊断。
# 评测器按 --case-id 逐个接收，题集本身不被修改。
$script:CaseIds = @(
    "real_huawei_2024_revenue",
    "real_zhongcheng_2024_net_profit",
    "real_furun_2024_adjusted_revenue_synonym",
    "real_gree_2023_core_table_fields",
    "real_guangdao_full_summary_consistency"
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
$script:TraceEnvBackup = @{}
$script:TraceEnvEntered = $false
$script:BackendProcess = $null
$script:LogPath = $null

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
    if (-not (Test-Path -LiteralPath $script:TestsetPath -PathType Leaf)) {
        throw "题集不存在：$script:TestsetPath"
    }
    $testset = Get-Content -LiteralPath $script:TestsetPath -Raw -Encoding UTF8
    foreach ($caseId in $script:CaseIds) {
        if ($testset -notmatch ("(?m)^\s*-\s+id:\s*" + [regex]::Escape($caseId) + "\s*$")) {
            throw "题集缺少目标case：$caseId"
        }
    }
    Write-RunnerLog "题集静态检查通过：固定5题，不扩展为8题或36题"
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
    Write-RunnerLog "Qdrant集合存在且向量维度为1024（仅GET，无写入、不重建索引）"
}

function Enter-RunEnvironment {
    if ($script:EnvironmentEntered) { return }
    $script:EnvironmentBackup = @{}
    foreach ($name in $script:FixedConfig.Keys) {
        $script:EnvironmentBackup[$name] = [Environment]::GetEnvironmentVariable($name, [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable($name, $script:FixedConfig[$name], [EnvironmentVariableTarget]::Process)
    }
    $script:EnvironmentBackup["PYTHONPATH"] = [Environment]::GetEnvironmentVariable("PYTHONPATH", [EnvironmentVariableTarget]::Process)
    $src = Join-Path $script:BackendDir "src"
    $oldPythonPath = $script:EnvironmentBackup["PYTHONPATH"]
    $newPythonPath = if ($oldPythonPath) { $src + [IO.Path]::PathSeparator + $oldPythonPath } else { $src }
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $newPythonPath, [EnvironmentVariableTarget]::Process)
    $script:EnvironmentEntered = $true
    Write-RunnerLog "正式阈值固定：KB_ANSWERABILITY_SCORE=0.5（不运行阈值0诊断）"
}

function Exit-RunEnvironment {
    if (-not $script:EnvironmentEntered) { return }
    foreach ($name in $script:EnvironmentBackup.Keys) {
        [Environment]::SetEnvironmentVariable($name, $script:EnvironmentBackup[$name], [EnvironmentVariableTarget]::Process)
    }
    $script:EnvironmentBackup = @{}
    $script:EnvironmentEntered = $false
}

function Enter-TraceEnvironment {
    param([string]$TracePath)
    if ($script:TraceEnvEntered) { return }
    $script:TraceEnvBackup = @{}
    $traceVars = [ordered]@{
        KB_DIAGNOSTIC_TRACE_ENABLED = "1"
        KB_DIAGNOSTIC_TRACE_PATH = $TracePath
    }
    foreach ($name in $traceVars.Keys) {
        $script:TraceEnvBackup[$name] = [Environment]::GetEnvironmentVariable($name, [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable($name, $traceVars[$name], [EnvironmentVariableTarget]::Process)
    }
    $script:TraceEnvEntered = $true
    # 只输出变量名与开关状态，绝不输出环境变量值以外的内容（此处无凭据）
    Write-RunnerLog "诊断追踪已启用：KB_DIAGNOSTIC_TRACE_ENABLED=1（追踪文件路径见产物清单）"
}

function Exit-TraceEnvironment {
    if (-not $script:TraceEnvEntered) { return }
    foreach ($name in $script:TraceEnvBackup.Keys) {
        [Environment]::SetEnvironmentVariable($name, $script:TraceEnvBackup[$name], [EnvironmentVariableTarget]::Process)
    }
    $script:TraceEnvBackup = @{}
    $script:TraceEnvEntered = $false
    Write-RunnerLog "诊断追踪环境变量已恢复（进程级）"
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
    # 留成孤儿，继续占用隔离端口。因此连带清理本次启动进程派生的 python
    # 子进程，仍然绝不查杀端口占用者。
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
    Write-RunnerLog "评测器退出码：$exitCode（本轮目标是取证，不以passed为门禁）"
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
    foreach ($result in $results) {
        $id = [string]$result.id
        if (-not $expectedSet.ContainsKey($id) -or $seen.ContainsKey($id)) { throw "报告包含未知或重复case：$id" }
        $seen[$id] = $true
        if ($null -eq $result.passed) { throw "case缺少passed字段：$id" }
        if ($null -eq $result.retrieval.contexts) { throw "case缺少retrieval.contexts：$id" }
        if ($null -eq $result.retrieval.contexts.page_hit -or $null -eq $result.retrieval.contexts.evidence_rate) {
            throw "case缺少page_hit/evidence_rate：$id"
        }
        Write-RunnerLog ("case={0}; passed={1}; page_hit={2}; evidence_rate={3}; elapsed_s={4}; citations={5}" -f `
            $id, $result.passed, $result.retrieval.contexts.page_hit, $result.retrieval.contexts.evidence_rate, `
            $result.elapsed_s, $result.citations)
    }
    if ($seen.Count -ne $ExpectedCaseIds.Count) { throw "报告缺少固定case" }
    Write-RunnerLog "报告完整性通过：$($ExpectedCaseIds.Count)/$($ExpectedCaseIds.Count) completed，error_count=0"
    return $report
}

function Test-TraceSecretLeak {
    <# 递归检查一条已解析的追踪记录：敏感键必须已被脱敏成 ***REDACTED***，
       字符串值不得出现 sk- / Bearer 形态的明文凭据。只看"值"不看"键名"，
       避免 config 里正常的字段名造成误报。 #>
    param($Node, [string]$NodePath = '$')
    $hits = @()
    if ($null -eq $Node) { return $hits }
    if (($Node -is [System.Collections.IEnumerable]) -and ($Node -isnot [string])) {
        $index = 0
        foreach ($item in $Node) {
            $hits += @(Test-TraceSecretLeak -Node $item -NodePath ($NodePath + "[$index]"))
            $index++
        }
        return $hits
    }
    if ($Node -is [System.Management.Automation.PSCustomObject]) {
        foreach ($prop in $Node.PSObject.Properties) {
            $name = [string]$prop.Name
            $value = $prop.Value
            if ($name -match '(?i)(api[_-]?key|secret|password|passwd|credential|authorization|cookie|signature|private[_-]?key|access[_-]?key|bearer)') {
                if ([string]$value -ne '***REDACTED***') {
                    $hits += ($NodePath + "." + $name + " 敏感键未脱敏")
                }
            }
            $hits += @(Test-TraceSecretLeak -Node $value -NodePath ($NodePath + "." + $name))
        }
        return $hits
    }
    if ($Node -is [string]) {
        if ($Node -match 'sk-[A-Za-z0-9]{6,}') { $hits += ($NodePath + " 疑似 sk- 密钥明文") }
        if ($Node -match '(?i)Bearer\s+[A-Za-z0-9._\-]{6,}') { $hits += ($NodePath + " 疑似 Bearer 令牌明文") }
    }
    return $hits
}

function Get-ExpectedQuestions {
    <# 从本次评测 JSON 取固定 5 题的问题文本，用于和追踪记录的 basic.question
       做一一对应。不硬编码题面，题集更新时不会失效。 #>
    param([string]$ReportPath)
    $raw = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8
    $parsed = $raw | ConvertFrom-Json
    $questions = @()
    foreach ($result in $parsed.reports[0].results) {
        $questions += [string]$result.question
    }
    return @($questions)
}

function Assert-TraceArtifact {
    param(
        [string]$TracePath,
        [int]$ExpectedCount,
        [string]$ReportPath
    )
    # —— 1. 文件必须存在 ——
    if (-not (Test-Path -LiteralPath $TracePath -PathType Leaf)) {
        throw "未生成候选追踪文件：$TracePath"
    }

    $expectedQuestions = @(Get-ExpectedQuestions -ReportPath $ReportPath)
    $lines = @(Get-Content -LiteralPath $TracePath -Encoding UTF8 | Where-Object { $_.Trim() })

    # —— 2. 非空行数必须严格等于题数，不允许"差不多" ——
    Write-RunnerLog "候选追踪记录数：实际$($lines.Count)，期望$ExpectedCount"
    if ($lines.Count -ne $ExpectedCount) {
        throw ("候选追踪记录数不符：实际{0}条，期望{1}条（必须严格相等；追踪文件保留待人工核对）" -f $lines.Count, $ExpectedCount)
    }

    # —— 3. 逐条解析 + 结构校验 ——
    $records = @()
    foreach ($line in $lines) {
        try {
            $records += @($line | ConvertFrom-Json)
        } catch {
            throw "候选追踪存在非法JSON记录：$($_.Exception.Message)"
        }
    }

    # 检索链核心阶段：缺任何一个都无法定位"目标页是在哪一层丢的"
    $coreStages = @('raw_recall', 'structured_fin_route', 'post_merge', 'node_retrieve')
    # 精排/最终上下文：二者有一即可（空候选分支只会有 empty_candidates）
    $rankingStages = @('rerank_after_model', 'post_protect')
    # 转人工/降级分支不强制 initial_answer，但必须留下可判读的终局信息
    $terminalStages = @('refusal_escalation', 'safe_fallback', 'final_answer')

    $seenTraceIds = @{}
    $matchedQuestions = @{}
    foreach ($record in $records) {
        $traceId = [string]$record.trace_id
        if ([string]::IsNullOrWhiteSpace($traceId)) { throw "候选追踪存在缺少 trace_id 的记录" }

        # 3a. trace_id 唯一
        if ($seenTraceIds.ContainsKey($traceId)) { throw "候选追踪 trace_id 重复：$traceId" }
        $seenTraceIds[$traceId] = $true

        # 3b. 必填字段
        $question = ''
        if ($record.basic -and $record.basic.question) { $question = [string]$record.basic.question }
        if ([string]::IsNullOrWhiteSpace($question)) { throw "候选追踪 trace_id=$traceId 缺少 basic.question" }
        if (-not $record.ts_start) { throw "候选追踪 trace_id=$traceId 缺少 ts_start" }
        if (-not $record.ts_end) { throw "候选追踪 trace_id=$traceId 缺少 ts_end" }
        if (-not $record.outcome) { throw "候选追踪 trace_id=$traceId 缺少 outcome" }
        if (-not $record.stages -or @($record.stages).Count -eq 0) {
            throw "候选追踪 trace_id=$traceId 的 stages 为空"
        }

        # 3c. 阶段覆盖
        $stageNames = @()
        foreach ($stage in $record.stages) { $stageNames += [string]$stage.stage }
        foreach ($required in $coreStages) {
            if ($stageNames -notcontains $required) {
                throw ("候选追踪 trace_id=$traceId 缺少核心阶段 $required（实际阶段：{0}）" -f ($stageNames -join ','))
            }
        }
        $hasRanking = $false
        foreach ($candidate in $rankingStages) {
            if ($stageNames -contains $candidate) { $hasRanking = $true }
        }
        if (-not $hasRanking) {
            throw ("候选追踪 trace_id=$traceId 缺少精排/最终上下文阶段（{0} 之一；实际阶段：{1}）" -f ($rankingStages -join '/'), ($stageNames -join ','))
        }

        # 3d. 终局可判读：正常路径看 initial_answer，转人工分支至少要有终局阶段或 outcome.escalate
        $isTerminalBranch = $false
        foreach ($candidate in $terminalStages) { if ($stageNames -contains $candidate) { $isTerminalBranch = $true } }
        $escalated = ($record.outcome.escalate -eq $true)
        if (($stageNames -notcontains 'initial_answer') -and (-not $isTerminalBranch) -and (-not $escalated)) {
            throw ("候选追踪 trace_id=$traceId 既无 initial_answer，也无终局阶段/转人工标记，无法判读（实际阶段：{0}）" -f ($stageNames -join ','))
        }

        # 3e. 敏感内容
        $leaks = @(Test-TraceSecretLeak -Node $record -NodePath ('$[' + $traceId + ']'))
        if ($leaks.Count -gt 0) {
            throw ("候选追踪 trace_id=$traceId 存在敏感内容：{0}" -f ($leaks -join '; '))
        }

        $matchedQuestions[$question] = $true
    }

    # —— 4. 必须一一覆盖固定 5 题，不重复不漏题 ——
    foreach ($expected in $expectedQuestions) {
        if (-not $matchedQuestions.ContainsKey($expected)) {
            throw ("候选追踪未覆盖题目：{0}（追踪文件保留待人工核对）" -f $expected)
        }
    }
    if ($matchedQuestions.Count -ne $expectedQuestions.Count) {
        throw ("候选追踪题目数不匹配：追踪{0}题，期望{1}题" -f $matchedQuestions.Count, $expectedQuestions.Count)
    }

    Write-RunnerLog ("候选追踪校验通过：{0}条记录，trace_id唯一，覆盖固定{1}题，核心阶段齐全，无敏感内容" -f $records.Count, $expectedQuestions.Count)
}

function New-AttemptId {
    return ("qdrant-failed5-trace-20260902-{0}-{1}" -f `
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

    $attemptId = New-AttemptId
    $mainReportPath = Join-Path $script:ReportsDir ($attemptId + ".json")
    $tracePath = Join-Path $script:ReportsDir ($attemptId + ".trace.jsonl")
    $script:LogPath = Join-Path $script:ReportsDir ($attemptId + ".runner.log")
    $mainStdoutPath = Join-Path $script:ReportsDir ($attemptId + ".backend.stdout.log")
    $mainStderrPath = Join-Path $script:ReportsDir ($attemptId + ".backend.stderr.log")
    New-Item -ItemType Directory -Path $script:ReportsDir -Force | Out-Null
    Assert-OutputUnused @($script:LogPath, $mainReportPath, $mainStdoutPath, $mainStderrPath, $tracePath)
    Set-Content -LiteralPath $script:LogPath -Value "" -Encoding UTF8
    Write-RunnerLog "attempt_id=$attemptId"
    Write-RunnerLog "运行模式：固定5题候选追踪（只取证，不修复答案、不降低阈值）"
    Write-RunnerLog "产物-评测JSON=$mainReportPath"
    Write-RunnerLog "产物-候选追踪=$tracePath"
    Write-RunnerLog "产物-runner日志=$($script:LogPath)"
    Write-RunnerLog "产物-后端stdout=$mainStdoutPath"
    Write-RunnerLog "产物-后端stderr=$mainStderrPath"
    $script:PythonPath = Resolve-Python

    if (-not (Test-Path -LiteralPath $script:GatePath -PathType Leaf)) { throw "离线防退门禁不存在：$script:GatePath" }
    Write-RunnerLog "先执行现有离线防退门禁"
    $gateOutput = & (Join-Path $PSHOME "powershell.exe") -NoProfile -ExecutionPolicy Bypass -File $script:GatePath 2>&1
    $gateExit = $LASTEXITCODE
    foreach ($line in $gateOutput) { Write-RunnerLog ([string]$line) }
    if ($gateExit -ne 0) { throw "离线防退门禁失败，停止后续Qdrant评测：exit=$gateExit" }
    Write-RunnerLog "离线防退门禁通过"

    Test-CaseSet
    Test-RequiredConfiguration
    Test-PythonDependencies
    Test-PortFree
    Test-DockerAndQdrant

    Enter-RunEnvironment
    Enter-TraceEnvironment -TracePath $tracePath
    try {
        Start-IsolatedBackend -StdoutPath $mainStdoutPath -StderrPath $mainStderrPath
        $evalExit = Invoke-Evaluation -ReportPath $mainReportPath `
            -Label $attemptId -SelectedCaseIds $script:CaseIds
    } finally {
        Stop-IsolatedBackend
        Exit-TraceEnvironment
        Exit-RunEnvironment
    }
    Wait-PortReleased
    if ($evalExit -ge 2) { throw "评测基础设施错误，报告已保留：exit=$evalExit" }

    $null = Assert-TargetedReport -ReportPath $mainReportPath -ExpectedCaseIds $script:CaseIds
    # 追踪校验必须在"报告完整性"之后：题目清单要从本次评测 JSON 里取，
    # 保证追踪记录与真正跑过的 5 题一一对应。任何一项不通过都直接退出非零。
    Assert-TraceArtifact -TracePath $tracePath -ExpectedCount $script:CaseIds.Count -ReportPath $mainReportPath

    Write-RunnerLog "完成：固定5题已完成且error_count=0；候选追踪已落盘，等待归因"
    exit 0
} catch {
    try { Stop-IsolatedBackend } catch { }
    try { Exit-TraceEnvironment } catch { }
    try { Exit-RunEnvironment } catch { }
    if ($script:LogPath) { Write-RunnerLog ("失败：" + $_.Exception.Message) }
    Write-Error $_.Exception.Message
    exit 2
}
