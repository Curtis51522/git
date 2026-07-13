@echo off
setlocal

cd /d "%~dp0"

set "PY=%BAKERY_PYTHON%"
if not defined PY set "PY=%LOCALAPPDATA%\BakeryAI\venv313\Scripts\python.exe"

if not exist "%PY%" (
    echo Bakery AI Python 3.13 environment not found.
    echo Run: py -3.13 -m venv "%LOCALAPPDATA%\BakeryAI\venv313"
    echo Then: "%LOCALAPPDATA%\BakeryAI\venv313\Scripts\python.exe" -m pip install -r requirements.txt
    exit /b 1
)

"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"
if errorlevel 1 (
    echo Python 3.13 is required.
    exit /b 1
)

echo Starting S5 LangGraph service on 127.0.0.1:8001...
start "Bakery S5 LangGraph Server" "%PY%" -m s5_agent.server

echo Starting Bakery AI main server on 127.0.0.1:8002...
start "Bakery AI Main Server" "%PY%" main.py

echo Open http://127.0.0.1:8002/
