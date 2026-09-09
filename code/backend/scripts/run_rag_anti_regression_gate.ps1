param(
    [string]$ReportPath = ''
)

$ErrorActionPreference = 'Stop'
$backend = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $backend '.venv\Scripts\python.exe'
$pytest = Join-Path $backend '.venv\Scripts\pytest.exe'
$checker = Join-Path $backend 'scripts\check_rag_anti_regression_report.py'
$replay = Join-Path $backend 'scripts\replay_rag_passed_answers_offline.py'
$testFiles = @(
    'tests/unit/test_rag_fixed_context_huawei_jinzhou.py',
    'tests/unit/test_rag_fixed_context_shenlian_longyu.py'
)

if (-not (Test-Path -LiteralPath $pytest -PathType Leaf)) {
    Write-Error "pytest 不存在: $pytest"
    exit 1
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "Python 不存在: $python"
    exit 1
}
if (-not (Test-Path -LiteralPath $replay -PathType Leaf)) {
    Write-Error "离线回放脚本不存在: $replay"
    exit 1
}

Push-Location -LiteralPath $backend
try {
    $pytestOutput = @(& $pytest -q @testFiles 2>&1)
    $pytestExitCode = $LASTEXITCODE
    $pytestOutput | ForEach-Object { Write-Output $_ }
    $pytestText = $pytestOutput -join "`n"
    $hasNonPassSummary = $pytestText -match '(?i)\b(?:xfailed|xpassed|skipped|failed|errors?)\b'
    if ($pytestExitCode -ne 0 -or $hasNonPassSummary) {
        Write-Error '离线固定上下文门禁失败：必须全部 passed，不能有 failure、xfail、xpass、skip 或 collection error。'
        exit 1
    }

    & $python $replay
    if ($LASTEXITCODE -ne 0) {
        Write-Error '历史正确行为离线回放门禁失败：请按 case 和 category 处理，不能用评测 passed 字段替代回放结果。'
        exit 1
    }

    if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
        $resolvedReport = $ReportPath
        if (-not [IO.Path]::IsPathRooted($resolvedReport)) {
            $resolvedReport = Join-Path $backend $resolvedReport
        }
        if (-not (Test-Path -LiteralPath $resolvedReport -PathType Leaf)) {
            Write-Error "评测报告不存在: $resolvedReport"
            exit 1
        }
        & $python $checker $resolvedReport
        if ($LASTEXITCODE -ne 0) {
            Write-Error '既有评测报告未通过四题关键 case 门禁。'
            exit 1
        }
    }
}
finally {
    Pop-Location
}

Write-Output 'RAG 防退化门禁通过：离线固定上下文测试全部 passed。'
exit 0
