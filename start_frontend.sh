#!/bin/bash

# OMR System Frontend Startup Script

echo "Starting OMR System Frontend..."

# Check if virtual environment exists
if [ ! -d "../omr_env" ]; then
    echo "Error: Virtual environment not found at ../omr_env"
    echo "Please run the setup script first"
    exit 1
fi

# Activate virtual environment
source ../omr_env/bin/activate

# Change to frontend directory
cd frontend

echo "Virtual environment activated"
echo "Starting Streamlit app..."

# Start the Streamlit app
streamlit run app.py --server.port 8501