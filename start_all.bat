@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

set "PY=%CD%\.venv\Scripts\python.exe"

echo Starting S5 LangGraph service on 127.0.0.1:8001...
start "Bakery S5 LangGraph Server" "%PY%" -m s5_agent.server

echo Starting Bakery AI main server on 127.0.0.1:8002...
start "Bakery AI Main Server" "%PY%" main.py

echo Open http://127.0.0.1:8002/
