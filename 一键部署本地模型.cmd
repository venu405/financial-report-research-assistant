@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\code\backend"
set "SCRIPT=%BACKEND_DIR%\scripts\setup_local_llm.ps1"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"

if not exist "%SCRIPT%" (
    echo [ERROR] 找不到部署脚本："%SCRIPT%"
    exit /b 2
)
if not exist "%PYTHON_EXE%" (
    echo [ERROR] 找不到后端虚拟环境："%PYTHON_EXE%"
    exit /b 2
)

pushd "%BACKEND_DIR%"
echo [INFO] 开始部署本地模型；不会启动项目后端。
call powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" echo [ERROR] 本地模型部署失败，退出码 %EXIT_CODE%。
exit /b %EXIT_CODE%
