# OMR System Deployment Guide

## Quick Start

### 1. Prerequisites
- Python 3.9 or higher
- 2GB+ RAM recommended
- 1GB+ free disk space

### 2. Installation
```bash
# Navigate to project directory
cd /Users/aashish.rasne/Documents/rsad/workspace-side-projects/omr_system

# Virtual environment is already created at ../omr_env
# If not, create it:
# python3 -m venv ../omr_env

# Install dependencies (already done)
source ../omr_env/bin/activate
pip install -r requirements.txt
```

### 3. Start the System

#### Option 1: Using startup scripts (Recommended)
```bash
# Terminal 1: Start Backend API
./start_backend.sh

# Terminal 2: Start Frontend (in a new terminal)
./start_frontend.sh
```

#### Option 2: Manual startup
```bash
# Terminal 1: Backend
cd backend
source ../../omr_env/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
source ../../omr_env/bin/activate
streamlit run app.py --server.port 8501
```

### 4. Access the System
- **Frontend (User Interface)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## System Components

### Core Processing Engine (`/core`)
- **preprocessor.py**: Image preprocessing and enhancement
- **bubble_detector.py**: Bubble detection and classification
- **answer_key_loader.py**: Excel answer key parsing
- **scoring_engine.py**: Answer matching and scoring
- **omr_processor.py**: Main processing pipeline

### Backend API (`/backend`)
- **main.py**: FastAPI REST API server
- **database.py**: SQLite database management

### Frontend Dashboard (`/frontend`)
- **app.py**: Streamlit web interface

### Configuration
- **config.py**: System configuration settings
- **requirements.txt**: Python dependencies

## Usage Instructions

### Processing Single OMR Sheet
1. Open the web interface at http://localhost:8501
2. Go to "Upload & Process" tab
3. Upload an OMR sheet image (JPG, PNG)
4. Select the answer set (A or B)
5. Optionally enter student information
6. Click "Process OMR Sheet"

### Viewing Results
1. Go to "Results Dashboard" tab
2. View processed sheets and scores
3. Download results as CSV

### System Statistics
1. Go to "Statistics" tab
2. View overall performance metrics
3. Subject-wise analysis

## API Endpoints

### Main Endpoints
- `POST /upload-and-process`: Process single OMR sheet
- `POST /batch-process`: Process multiple sheets
- `GET /results`: Get all results
- `GET /results/{id}`: Get specific result
- `GET /export/csv`: Export results as CSV
- `GET /statistics`: Get system statistics

### System Endpoints
- `GET /`: API information
- `GET /health`: Health check
- `GET /system-info`: System status

## File Structure
```
omr_system/
├── core/                   # Processing engine
├── backend/               # FastAPI server
├── frontend/              # Streamlit UI
├── data/                  # Answer keys
├── uploads/               # Uploaded images
├── results/               # Processing outputs
├── tests/                 # Test files
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── start_*.sh            # Startup scripts
```

## Performance Specifications

### Accuracy
- Target: <0.5% error rate
- Tested with sample data from Set A and Set B

### Processing Speed
- Single sheet: ~10-30 seconds
- Batch processing: Parallel capability

### Image Requirements
- Format: JPEG, PNG
- Size: Up to 10MB
- Resolution: Minimum 1000px width recommended
- Quality: Clear, well-lit images

## Troubleshooting

### Common Issues

#### 1. "Backend API is not running"
- Ensure FastAPI server is started: `./start_backend.sh`
- Check port 8000 is not in use: `lsof -i :8000`

#### 2. "Virtual environment not found"
- Recreate environment: `python3 -m venv ../omr_env`
- Reinstall packages: `pip install -r requirements.txt`

#### 3. "Answer key file not found"
- Verify file exists: `ls data/`
- Should contain: `Key (Set A and B).xlsx`

#### 4. Low accuracy scores
- Check image quality (lighting, focus, alignment)
- Verify answer set selection (A vs B)
- Ensure bubbles are clearly marked

### Logs and Debugging
- Backend logs: Check terminal output
- Processing debug files: `results/` directory
- Database: `omr_results.db` (SQLite)

## Production Deployment

### Security Considerations
1. Change CORS settings in `backend/main.py`
2. Use HTTPS in production
3. Implement authentication if needed
4. Secure file uploads

### Performance Optimization
1. Use PostgreSQL instead of SQLite for large volumes
2. Implement Redis caching
3. Use Nginx reverse proxy
4. Consider Docker containerization

### Monitoring
1. Add application logging
2. Monitor API response times
3. Track processing accuracy
4. Set up health checks

## Sample Data
- Location: `../Theme 1 - Sample Data/`
- Set A images: 13 sample OMR sheets
- Set B images: 10 sample OMR sheets
- Answer keys: Excel file with correct answers

## Support
- Check logs for error messages
- Run test suite: `cd tests && python test_system.py`
- Verify system status via web interface