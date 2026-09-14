@echo off
setlocal EnableExtensions
title Financial Report Research Assistant

cd /d "%~dp0"
set "DOCKER_DESKTOP=C:\Program Files\Docker\Docker\Docker Desktop.exe"
set "APP_URL=http://127.0.0.1:3000"

echo.
echo Starting Financial Report Research Assistant...

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker command was not found. Install Docker Desktop and run this file again.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Engine is not running. Starting Docker Desktop...
  docker desktop start >nul 2>&1
  if errorlevel 1 if exist "%DOCKER_DESKTOP%" start "" /min "%DOCKER_DESKTOP%"
  call :wait_for_docker
  if errorlevel 1 (
    echo Docker Desktop was not ready within 180 seconds. Check Docker Desktop and WSL, then try again.
    pause
    exit /b 1
  )
)

echo Docker is ready. Starting application containers...
docker compose up -d
if errorlevel 1 (
  echo Existing images failed. Rebuilding and starting containers...
  docker compose up --build -d
  if errorlevel 1 (
    echo Startup failed. Run: docker compose logs
    pause
    exit /b 1
  )
)

echo Waiting for the web application...
call :wait_for_frontend
if errorlevel 1 (
  echo Containers started, but the web app was not ready within 180 seconds.
  echo Run: docker compose logs
  pause
  exit /b 1
)

echo.
echo Startup complete. Opening %APP_URL%
start "Financial Report Research Assistant" "%APP_URL%"
echo Backend health check: http://127.0.0.1:8000/readyz
echo Stop command: docker compose down
exit /b 0

:wait_for_docker
for /l %%i in (1,1,60) do (
  docker info >nul 2>&1 && exit /b 0
  timeout /t 3 /nobreak >nul
)
exit /b 1

:wait_for_frontend
for /l %%i in (1,1,60) do (
  curl.exe -fsS --max-time 3 "%APP_URL%" >nul 2>&1 && exit /b 0
  timeout /t 3 /nobreak >nul
)
exit /b 1
