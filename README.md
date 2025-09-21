# Automated OMR Evaluation & Scoring System

## Overview
This system automates the evaluation of OMR sheets captured via mobile phone cameras, providing accurate scoring for multiple-choice tests with 100 questions across 5 subjects.

## Features
- Mobile phone camera image processing
- Automatic bubble detection and classification
- Support for multiple test versions (Set A, Set B, etc.)
- Web-based interface for evaluators
- <0.5% error tolerance
- CSV/Excel result export
- Comprehensive audit trail

## Project Structure
```
omr_system/
├── core/           # OMR processing engine
├── backend/        # FastAPI backend
├── frontend/       # Streamlit frontend
├── data/           # Answer keys and configurations
├── uploads/        # Uploaded OMR images
├── results/        # Processing results and exports
└── tests/          # Test files
```

## Installation & Usage

### Quick Start
```bash
# 1. Start the backend API (Terminal 1)
./start_backend.sh

# 2. Start the frontend (Terminal 2)
./start_frontend.sh

# 3. Open your browser
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### Detailed Setup
See [DEPLOYMENT.md](DEPLOYMENT.md) for complete installation and deployment instructions.

## Tech Stack
- **Core Processing**: OpenCV, NumPy, SciPy, Scikit-learn
- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Handling**: Pandas, OpenPyXL