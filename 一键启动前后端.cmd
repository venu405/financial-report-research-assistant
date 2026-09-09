@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\code\backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "FRONTEND_DIR=%ROOT_DIR%\code\frontend"
set "FRONTEND_PACKAGE=%FRONTEND_DIR%\package.json"

if not exist "%BACKEND_DIR%\" (
    echo [ERROR] Backend directory not found:
    echo         "%BACKEND_DIR%"
    goto :fail
)
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Backend Python not found:
    echo         "%PYTHON_EXE%"
    goto :fail
)
if not exist "%BACKEND_DIR%\src\main.py" (
    echo [ERROR] Backend entry file not found:
    echo         "%BACKEND_DIR%\src\main.py"
    goto :fail
)
if not exist "%FRONTEND_DIR%\" (
    echo [ERROR] Frontend directory not found:
    echo         "%FRONTEND_DIR%"
    goto :fail
)
if not exist "%FRONTEND_PACKAGE%" (
    echo [ERROR] Frontend package file not found:
    echo         "%FRONTEND_PACKAGE%"
    goto :fail
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm.cmd was not found in PATH.
    echo         Install Node.js/npm or add npm to PATH, then try again.
    goto :fail
)

set "BACKEND_ONLINE=0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try { $a=$c.BeginConnect('127.0.0.1',8000,$null,$null); if ($a.AsyncWaitHandle.WaitOne(500)) { $c.EndConnect($a); exit 0 }; exit 1 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
if not errorlevel 1 set "BACKEND_ONLINE=1"

set "FRONTEND_ONLINE=0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try { $a=$c.BeginConnect('127.0.0.1',5174,$null,$null); if ($a.AsyncWaitHandle.WaitOne(500)) { $c.EndConnect($a); exit 0 }; exit 1 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
if not errorlevel 1 set "FRONTEND_ONLINE=1"

if "%BACKEND_ONLINE%"=="1" (
    echo [OK] Backend is already online at http://127.0.0.1:8000
) else (
    echo [INFO] Starting backend in a new window...
    start "Knowledge Backend" /D "%BACKEND_DIR%" "%ComSpec%" /d /k ""%PYTHON_EXE%" -m uvicorn main:app --app-dir src --host 127.0.0.1 --port 8000"
)

if "%FRONTEND_ONLINE%"=="1" (
    echo [OK] Frontend is already online at http://127.0.0.1:5174
) else (
    echo [INFO] Starting frontend in a new window...
    start "Knowledge Frontend" /D "%FRONTEND_DIR%" "%ComSpec%" /d /k "call npm.cmd run dev -- --host 127.0.0.1 --port 5174 --strictPort"
)

echo [INFO] Waiting for backend at http://127.0.0.1:8000/readyz ...
set /a BACKEND_WAIT=0
:wait_backend
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/readyz' -TimeoutSec 3; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :backend_ready
set /a BACKEND_WAIT+=1
if %BACKEND_WAIT% GEQ 30 (
    echo [ERROR] Backend did not become ready. Check the Knowledge Backend window.
    goto :fail
)
timeout /t 1 /nobreak >nul
goto :wait_backend

:backend_ready
echo [OK] Backend is ready.
echo [INFO] Waiting for frontend at http://127.0.0.1:5174 ...
set /a FRONTEND_WAIT=0
:wait_frontend
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5174' -TimeoutSec 3; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :frontend_ready
set /a FRONTEND_WAIT+=1
if %FRONTEND_WAIT% GEQ 30 (
    echo [ERROR] Frontend did not become ready. Check the Knowledge Frontend window.
    goto :fail
)
timeout /t 1 /nobreak >nul
goto :wait_frontend

:frontend_ready
echo [OK] Frontend is ready.
echo [OK] Both services are ready. Opening browser...
start "" "http://127.0.0.1:5174"
exit /b 0

:fail
echo.
pause
exit /b 1
