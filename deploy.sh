#!/bin/bash
set -e

# Deployment script for DNN backend on Azure
echo "Starting deployment for DNN backend..."

# Navigate to DNN backend
cd "Deep Neural Network/backend" || { echo "ERROR: DNN backend folder not found"; exit 1; }

echo "Creating virtual environment..."
python -m venv antenv

echo "Activating virtual environment..."
source antenv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ DNN backend deployment successful"
exit 0
