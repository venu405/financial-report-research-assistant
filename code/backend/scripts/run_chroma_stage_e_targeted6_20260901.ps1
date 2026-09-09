$ErrorActionPreference = 'Stop'

$backend = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $backend '.venv\Scripts\python.exe'
$testset = Join-Path $backend 'testsets\rag_real_quality_v2.yaml'
$report = Join-Path $backend 'reports\chroma-stage-e-targeted6-verification-binding-v1-20260901.json'
$envFile = Join-Path $backend '.env'
$attemptId = 'chroma-stage-e-targeted6-verification-binding-v1-20260901'

$expectedQaGraphSha256 = '5C0E50C4B8589C58B14ADD44DA3D8E29B910718AA8C4037E1929E9897F0BA57A'
$expectedRetrieverSha256 = '9D684CBF11CC4EEB58652682EEE2F2A6AB490ABA0F8E42EE293E348311CBDE0D'
$expectedTestsetSha256 = '6AB7AA0CBA3C3A2C29F49A5194EB3DE827D202520CFFFC79FC6CEA97633BD05E'
$qaGraph = Join-Path $backend 'src\services\kb\qa_graph.py'
$retriever = Join-Path $backend 'src\services\kb\retriever.py'

$caseIds = @(
    'real_shenlian_2025_h1_rnd_ratio_synonym',
    'real_furun_2024_adjusted_revenue_synonym',
    'real_yutai_2025_h1_revenue',
    'real_pengbo_2024_revenue',
    'real_yitai_2025_h1_operating_cash',
    'real_guangdao_2025_h1_notice_dissent'
)

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ".env not found: $Path"
    }
    foreach ($rawLine in Get-Content -LiteralPath $Path -ErrorAction Stop) {
        $line = ([string]$rawLine).Trim()
        if (-not $line -or $line.StartsWith('#')) {
            continue
        }
        $line = $line -replace '^export\s+', ''
        if ($line -notmatch '^([^#=\s]+)\s*=\s*(.*?)\s*$') {
            continue
        }
        $name = $matches[1]
        $value = $matches[2]
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        Set-Item -Path ("Env:{0}" -f $name) -Value $value
    }
}

function Assert-Sha256([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path -ErrorAction Stop).Hash.ToUpperInvariant()
    if ($actual -ne $Expected) {
        throw "SHA256 mismatch: $Path"
    }
}

function Get-Listening18081 {
    @(Get-NetTCPConnection -LocalPort 18081 -ErrorAction SilentlyContinue |
        Where-Object { $_.State.ToString() -eq 'Listen' })
}

$server = $null
$startedPid = $null
$evaluationExitCode = 1
$runStartedAt = Get-Date

try {
    if (Test-Path -LiteralPath $report -PathType Leaf) {
        throw "Target report already exists; refusing to run: $report"
    }

    Assert-Sha256 $qaGraph $expectedQaGraphSha256
    Assert-Sha256 $retriever $expectedRetrieverSha256
    Assert-Sha256 $testset $expectedTestsetSha256
    Import-DotEnv $envFile

    $existingListeners = @(Get-Listening18081)
    if ($existingListeners.Count -gt 0) {
        throw 'Port 18081 is already occupied; existing processes were not stopped.'
    }

    $env:PYTHONPATH = Join-Path $backend 'src'
    $env:KB_VECTOR_BACKEND = 'chroma'
    $env:KB_CHROMA_DIR = Join-Path $backend '.rag_eval\chroma_repro_26_current_20260831\chroma_data'
    $env:KB_COLLECTION = 'enterprise_kb'
    $env:KB_TOP_K = '5'
    $env:KB_MAX_HITS_PER_DOC = '3'
    $env:KB_STRUCTURED_FIN_ROUTE = '1'
    $env:KB_FIN_ROUTE_TOP_N = '8'
    $env:KB_EMBEDDING_MODE = 'bge_m3'
    $env:KB_EMBEDDING_MODEL = 'bge-m3'
    $env:KB_OLLAMA_HOST = 'http://127.0.0.1:11434'
    $env:KB_RERANK_MODE = 'llm'
    $env:LLM_BASE_URL = 'https://api.deepseek.com/v1'
    $env:LLM_MODEL_ID = 'deepseek-chat'

    $server = Start-Process `
        -FilePath $python `
        -ArgumentList @(
            '-m', 'uvicorn', 'main:app', '--app-dir', 'src',
            '--host', '127.0.0.1', '--port', '18081'
        ) `
        -WorkingDirectory $backend `
        -WindowStyle Hidden `
        -PassThru
    $startedPid = $server.Id

    $ready = $false
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $server.Refresh()
        if ($server.HasExited) {
            throw "uvicorn exited before readyz (exit $($server.ExitCode))"
        }
        try {
            $readyResponse = Invoke-RestMethod `
                -Uri 'http://127.0.0.1:18081/readyz' `
                -TimeoutSec 2
            if ($readyResponse.status -eq 'ok') {
                $ready = $true
                break
            }
        }
        catch {
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw 'uvicorn readyz timeout after 90 seconds'
    }

    $evaluateArgs = @(
        'scripts/evaluate_quality.py',
        '--rag', $testset,
        '--base-url', 'http://127.0.0.1:18081',
        '--label', $attemptId
    )
    foreach ($caseId in $caseIds) {
        $evaluateArgs += @('--case-id', $caseId)
    }
    $evaluateArgs += @('--output', $report)

    & $python @evaluateArgs
    $evaluationExitCode = $LASTEXITCODE
}
finally {
    if ($startedPid) {
        try {
            $server.Refresh()
            if (-not $server.HasExited) {
                Stop-Process -Id $startedPid -ErrorAction SilentlyContinue
            }
        }
        catch {
        }

        $released = $false
        $releaseDeadline = (Get-Date).AddSeconds(15)
        while ((Get-Date) -lt $releaseDeadline) {
            if (@(Get-Listening18081).Count -eq 0) {
                $released = $true
                break
            }
            Start-Sleep -Milliseconds 250
        }
        if ($released) {
            Write-Output ("stage-e cleanup: stopped PID {0}; port 18081 released" -f $startedPid)
        }
        else {
            Write-Warning ("stage-e cleanup: stopped PID {0}; port 18081 not confirmed released" -f $startedPid)
        }
    }
    $durationSeconds = ((Get-Date) - $runStartedAt).TotalSeconds
    Write-Output ("stage-e duration_seconds: {0:N2}" -f $durationSeconds)
}

exit $evaluationExitCode
