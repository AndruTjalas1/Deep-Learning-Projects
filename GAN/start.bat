@echo off
REM Quick start script for Windows
REM This script helps set up and run the DCGAN system

echo.
echo ========================================
echo   DCGAN Training System - Quick Start
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Node.js not found. Frontend will not work.
    echo Install from https://nodejs.org/
)

echo [1/4] Setting up Backend...
cd Backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

if %errorlevel% neq 0 (
    echo Error installing dependencies
    pause
    exit /b 1
)

cd ..

echo.
echo [2/4] Checking dataset...
python utils.py --check-dataset

echo.
echo [3/4] Starting Backend Server...
echo Backend running on http://localhost:8000
echo Press Ctrl+C to stop
echo.

start cmd /k "cd Backend && venv\Scripts\activate.bat && python main.py"

echo.
echo [4/4] Starting Frontend...
echo Frontend will open on http://localhost:3000
cd Frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install -q
)
timeout /t 2 /nobreak
npm run dev

pause
