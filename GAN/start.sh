#!/bin/bash
# Quick start script for macOS/Linux
# This script helps set up and run the DCGAN system

echo ""
echo "========================================"
echo "  DCGAN Training System - Quick Start"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "Warning: Node.js not found. Frontend will not work."
    echo "Install from https://nodejs.org/"
fi

echo "[1/4] Setting up Backend..."
cd Backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing dependencies"
    exit 1
fi

cd ..

echo ""
echo "[2/4] Checking dataset..."
python3 utils.py --check-dataset

echo ""
echo "[3/4] Starting Backend Server..."
echo "Backend running on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start backend in background
(cd Backend && source venv/bin/activate && python3 main.py) &
BACKEND_PID=$!

echo ""
echo "[4/4] Starting Frontend..."
echo "Frontend will open on http://localhost:3000"
cd Frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install -q
fi

sleep 2
npm run dev

# Clean up
kill $BACKEND_PID 2>/dev/null

echo ""
echo "Stopped."
