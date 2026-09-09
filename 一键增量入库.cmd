@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\code\backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%BACKEND_DIR%\scripts\ingest_corpus.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Backend Python not found:
    echo         "%PYTHON_EXE%"
    exit /b 2
)
if not exist "%SCRIPT%" (
    echo [ERROR] Incremental ingest script not found:
    echo         "%SCRIPT%"
    exit /b 2
)
if not defined KB_INGEST_TOKEN if not defined KB_API_TOKEN if not defined KB_INGEST_USER_ID if not defined KB_USER_ID (
    set "KB_INGEST_USER_ID=admin"
    echo [INFO] No ingest identity found; using local development user KB_INGEST_USER_ID=admin.
)
set "PYTHONPATH=%BACKEND_DIR%\src"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

pushd "%BACKEND_DIR%"
echo [INFO] Running incremental PDF ingest against http://127.0.0.1:8000 ...
"%PYTHON_EXE%" "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" echo [ERROR] Incremental ingest failed. Check the message above and rerun after fixing it.
exit /b %EXIT_CODE%
