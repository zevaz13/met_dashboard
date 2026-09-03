@echo off
REM Starts the MET dashboard (Windows). Requires uv: https://docs.astral.sh/uv/
setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo Error: 'uv' is not installed or not on PATH.
    echo Install it from https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

uv sync
if errorlevel 1 exit /b %errorlevel%

uv run streamlit run dashboard/Home.py
