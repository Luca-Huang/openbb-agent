@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8

echo [1/2] Refreshing dashboard...
python scripts\watchlist_dashboard.py
if errorlevel 1 (
    echo.
    echo [FAIL] Script error above. Press any key to close.
    pause >nul
    exit /b 1
)

echo [2/2] Opening HTML...
start "" "outputs\watchlist_dashboard.html"
