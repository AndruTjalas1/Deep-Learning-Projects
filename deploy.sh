#!/bin/bash

# Deployment script for monorepo with multiple backends
# Each App Service sets BACKEND_FOLDER environment variable

# Use the environment variable, or default to DNN
BACKEND_FOLDER=${BACKEND_FOLDER:-"Deep Neural Network/backend"}

echo "Deploying from folder: $BACKEND_FOLDER"

# Navigate to the correct backend folder
cd "$BACKEND_FOLDER" || { echo "ERROR: Folder not found: $BACKEND_FOLDER"; exit 1; }

# Create virtual environment
python -m venv antenv

# Activate virtual environment (Linux)
source antenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Deployment successful from $BACKEND_FOLDER"
exit 0
