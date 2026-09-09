[CmdletBinding()]
param(
    [string]$BackendDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "命令失败（退出码 $exitCode）：$FilePath"
    }
}

try {
    $scriptDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
    if ([string]::IsNullOrWhiteSpace($BackendDir)) {
        $backendPath = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory "..")).Path
    }
    else {
        $backendPath = (Resolve-Path -LiteralPath $BackendDir).Path
    }

    $pythonPath = Join-Path $backendPath ".venv\Scripts\python.exe"
    $modelFilePath = Join-Path $backendPath "Modelfile.qwen35-4b"
    $baseModel = "qwen3.5:4b"
    $aliasModel = "enterprise-kb-qwen35:4b"
    $rerankerModel = "BAAI/bge-reranker-base"
    $ollamaUrl = "http://127.0.0.1:11434"

    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "后端虚拟环境不存在：$pythonPath"
    }
    if (-not (Test-Path -LiteralPath $modelFilePath -PathType Leaf)) {
        throw "Modelfile 不存在：$modelFilePath"
    }

    $ollamaCommand = Get-Command ollama.exe -ErrorAction Stop
    $ollamaPath = $ollamaCommand.Source
    if ([string]::IsNullOrWhiteSpace($ollamaPath)) {
        $ollamaPath = $ollamaCommand.Definition
    }

    Write-Host "[INFO] 检查 Ollama..."
    Invoke-NativeChecked -FilePath $ollamaPath -ArgumentList @("--version")

    $modelLines = & $ollamaPath list 2>&1
    $listExitCode = $LASTEXITCODE
    if ($listExitCode -ne 0) {
        throw "无法读取 Ollama 模型清单（退出码 $listExitCode）"
    }
    $hasBaseModel = $false
    foreach ($line in $modelLines) {
        $columns = ([string]$line).Trim() -split "\s+"
        if ($columns.Count -gt 0 -and $columns[0] -eq $baseModel) {
            $hasBaseModel = $true
            break
        }
    }
    if ($hasBaseModel) {
        Write-Host "[INFO] 已存在 $baseModel，跳过下载。"
    }
    else {
        Write-Host "[INFO] 首次下载 $baseModel，预计需要数 GB 磁盘空间。"
        Invoke-NativeChecked -FilePath $ollamaPath -ArgumentList @("pull", $baseModel)
    }

    Write-Host "[INFO] 创建或刷新本项目模型别名 $aliasModel..."
    Invoke-NativeChecked -FilePath $ollamaPath -ArgumentList @("create", $aliasModel, "-f", $modelFilePath)
    $showLines = & $ollamaPath show --modelfile $aliasModel 2>&1
    $showExitCode = $LASTEXITCODE
    $showText = ($showLines | Out-String)
    if ($showExitCode -ne 0) {
        throw "无法验证模型别名 $aliasModel（退出码 $showExitCode）"
    }
    if ($showText -notmatch "(?im)\bnum_ctx\s+8192\b") {
        throw "模型别名 $aliasModel 未确认 num_ctx=8192"
    }

    Write-Host "[INFO] 安装或同步 fastembed（保留现有 onnxruntime）。"
    Invoke-NativeChecked -FilePath $pythonPath -ArgumentList @(
        "-m", "pip", "install", "--disable-pip-version-check", "fastembed>=0.7.4,<0.9.0"
    )
    Invoke-NativeChecked -FilePath $pythonPath -ArgumentList @("-m", "pip", "check")

    Write-Host "[INFO] 验证 $rerankerModel 可由 fastembed 加载（首次会下载模型）。"
    $crossencoderCode = "from fastembed.rerank.cross_encoder import TextCrossEncoder; model=TextCrossEncoder(model_name='$rerankerModel'); scores=list(model.rerank('What is revenue?', ['Revenue increased.'])); assert len(scores) == 1; print('CROSSENCODER_OK')"
    Invoke-NativeChecked -FilePath $pythonPath -ArgumentList @("-c", $crossencoderCode)

    Write-Host "[INFO] 通过 Ollama OpenAI 兼容接口执行严格 ASCII 冒烟。"
    $smokePayload = @{
        model = $aliasModel
        messages = @(@{
                role = "user"
                content = "Reply with exactly LOCAL_LLM_OK and no other text."
            })
        temperature = 0
        max_tokens = 32
        reasoning_effort = "none"
        think = $false
        stream = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $smokeBytes = [System.Text.Encoding]::UTF8.GetBytes($smokePayload)
    $smokeResponse = Invoke-RestMethod -Method Post `
        -Uri "$ollamaUrl/v1/chat/completions" `
        -Headers @{ Authorization = "Bearer ollama" } `
        -ContentType "application/json; charset=utf-8" `
        -Body $smokeBytes `
        -TimeoutSec 180
    $answer = [string]$smokeResponse.choices[0].message.content
    if ($answer.Trim() -ne "LOCAL_LLM_OK") {
        throw "本地模型严格格式冒烟失败：模型未返回 LOCAL_LLM_OK"
    }

    Write-Host "[INFO] 当前 Ollama 运行状态（可据此查看 GPU/CPU 分配）："
    Invoke-NativeChecked -FilePath $ollamaPath -ArgumentList @("ps")
    Write-Host "[OK] 本地模型部署、重排器加载和 OpenAI 兼容冒烟均成功。"
    exit 0
}
catch {
    [Console]::Error.WriteLine("[ERROR] 本地模型部署失败：{0}" -f $_.Exception.Message)
    exit 1
}
