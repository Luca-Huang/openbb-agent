@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8

echo [1/3] Checking aktools (needed for A-share analyst data)...
curl -s -o nul --max-time 2 http://127.0.0.1:8080/api/public/stock_zh_a_st_em
if errorlevel 1 (
    echo   aktools not running, starting in background...
    start "aktools" /MIN cmd /c "python -m aktools"
    echo   waiting 8s for aktools to come up...
    timeout /t 8 /nobreak >nul
) else (
    echo   aktools already running.
)

echo [2/3] Refreshing dashboard...
python scripts\watchlist_dashboard.py
if errorlevel 1 (
    echo.
    echo [FAIL] Script error above. Press any key to close.
    pause >nul
    exit /b 1
)

echo [3/3] Opening HTML...
start "" "outputs\watchlist_dashboard.html"
