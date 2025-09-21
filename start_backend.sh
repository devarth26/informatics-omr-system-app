#!/bin/bash

# OMR System Backend Startup Script

echo "Starting OMR System Backend..."

# Check if virtual environment exists
if [ ! -d "../omr_env" ]; then
    echo "Error: Virtual environment not found at ../omr_env"
    echo "Please run the setup script first"
    exit 1
fi

# Activate virtual environment
source ../omr_env/bin/activate

# Change to backend directory
cd backend

# Update the answer key path in main.py to use absolute path
export ANSWER_KEY_PATH="/Users/aashish.rasne/Documents/rsad/workspace-side-projects/omr_system/data/Key (Set A and B).xlsx"

echo "Virtual environment activated"
echo "Starting FastAPI server..."

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000