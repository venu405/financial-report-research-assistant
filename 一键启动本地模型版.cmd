@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "LLM_BASE_URL=http://127.0.0.1:11434/v1"
if not defined LLM_API_KEY (
    echo [ERROR] 未找到 LLM_API_KEY，请先在当前命令行环境中设置后再启动。
    exit /b 2
)
set "LLM_MODEL_ID=enterprise-kb-qwen35:4b"
set "LLM_REASONING_EFFORT=none"
set "KB_RERANK_MODE=crossencoder"
set "KB_RERANK_MODEL=BAAI/bge-reranker-base"
set "LLM_TIMEOUT=180"

where ollama.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 找不到 Ollama，请先运行“一键部署本地模型.cmd”。
    exit /b 2
)
ollama.exe show enterprise-kb-qwen35:4b >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 本地模型别名不存在，请先运行“一键部署本地模型.cmd”。
    exit /b 2
)

powershell.exe -NoProfile -Command "try { $client = New-Object Net.Sockets.TcpClient; $client.Connect('127.0.0.1',8000); $client.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [ERROR] 8000 端口已有后端。无法保证旧进程已切换到本地模型，已拒绝启动；不会自动结束用户进程。
    exit /b 3
)

echo [INFO] 使用本地模型启动前后端；本窗口环境变量不会写入 .env。
call "%ROOT_DIR%\一键启动前后端.cmd" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] 启动失败，退出码 %EXIT_CODE%。
exit /b %EXIT_CODE%
