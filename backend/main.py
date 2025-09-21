from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sys
from pathlib import Path
from typing import Optional, List
import logging
import json

# Add the parent directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.omr_processor import OMRProcessor
from backend.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OMR Evaluation System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ANSWER_KEY_PATH = str(Path(__file__).parent.parent / "data" / "Key (Set A and B).xlsx")
UPLOAD_DIR = Path("../uploads")
RESULTS_DIR = Path("../results")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Initialize OMR processor and database
try:
    omr_processor = OMRProcessor(ANSWER_KEY_PATH)
    db_manager = DatabaseManager()
    logger.info("OMR processor and database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    omr_processor = None
    db_manager = None

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OMR Evaluation System API",
        "version": "1.0.0",
        "status": "running" if omr_processor else "initialization_failed"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if omr_processor else "unhealthy",
        "components": {
            "omr_processor": "ready" if omr_processor else "failed",
            "database": "ready" if db_manager else "failed"
        }
    }

@app.get("/system-info")
async def get_system_info():
    """Get system information and statistics"""
    if not omr_processor:
        raise HTTPException(status_code=500, detail="OMR processor not initialized")

    try:
        stats = omr_processor.get_processing_statistics()
        db_stats = db_manager.get_statistics() if db_manager else {}

        return {
            "processing_stats": stats,
            "database_stats": db_stats,
            "system_ready": bool(omr_processor and db_manager)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}")

@app.post("/upload-and-process")
async def upload_and_process_omr(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    set_letter: str = Form("A"),
    student_name: Optional[str] = Form(None),
    student_id: Optional[str] = Form(None)
):
    """Upload OMR image and process it"""
    if not omr_processor:
        raise HTTPException(status_code=500, detail="OMR processor not initialized")

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded: {file.filename}")

        # Prepare student info
        student_info = {}
        if student_name:
            student_info['name'] = student_name
        if student_id:
            student_info['id'] = student_id

        # Process OMR sheet
        result = omr_processor.process_omr_sheet(
            str(file_path),
            set_letter.upper(),
            student_info,
            save_debug=True,
            output_dir=str(RESULTS_DIR)
        )

        # Save to database
        if db_manager:
            result_id = db_manager.save_result(result)
            result['database_id'] = result_id

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_old_files)

        return {
            "success": True,
            "result": result,
            "message": f"OMR sheet processed successfully. Score: {result['total_score']}/100"
        }

    except Exception as e:
        logger.error(f"Error processing OMR: {str(e)}")
        # Clean up uploaded file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/batch-process")
async def batch_process_omr(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    set_letter: str = Form("A")
):
    """Batch process multiple OMR images"""
    if not omr_processor:
        raise HTTPException(status_code=500, detail="OMR processor not initialized")

    try:
        # Save all uploaded files
        file_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue  # Skip non-image files

            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))

        logger.info(f"Batch processing {len(file_paths)} files")

        # Process batch
        batch_results = omr_processor.batch_process(
            file_paths,
            set_letter.upper(),
            str(RESULTS_DIR)
        )

        # Save results to database
        if db_manager:
            for file_path, result in batch_results['results'].items():
                try:
                    result_id = db_manager.save_result(result)
                    result['database_id'] = result_id
                except Exception as e:
                    logger.error(f"Error saving result to database: {str(e)}")

        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files)

        return {
            "success": True,
            "batch_results": batch_results,
            "message": f"Processed {batch_results['successful_count']}/{batch_results['processed_count']} files"
        }

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/results")
async def get_all_results(limit: int = 50, offset: int = 0):
    """Get all processing results with pagination"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        results = db_manager.get_all_results(limit=limit, offset=offset)
        return {
            "results": results,
            "total": len(results),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")

@app.get("/results/{result_id}")
async def get_result(result_id: int):
    """Get specific result by ID"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        result = db_manager.get_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching result: {str(e)}")

@app.get("/export/csv")
async def export_results_csv():
    """Export all results to CSV"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        import pandas as pd

        results = db_manager.get_all_results(limit=10000)  # Get all results
        if not results:
            raise HTTPException(status_code=404, detail="No results to export")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        csv_path = RESULTS_DIR / "all_results.csv"
        df.to_csv(csv_path, index=False)

        return FileResponse(
            path=str(csv_path),
            filename="omr_results.csv",
            media_type='text/csv'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive system statistics"""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        db_stats = db_manager.get_statistics()
        results = db_manager.get_all_results(limit=1000)

        # Calculate detailed statistics
        if results:
            subject_stats = {}
            subjects = ['python_score', 'eda_score', 'sql_score', 'powerbi_score', 'stats_score']

            for subject in subjects:
                scores = [r.get(subject, 0) for r in results if r.get(subject) is not None]
                if scores:
                    subject_stats[subject.replace('_score', '')] = {
                        'mean': sum(scores) / len(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'count': len(scores)
                    }

            # Score distribution
            percentages = [r['percentage'] for r in results]
            distribution = {
                'excellent': sum(1 for p in percentages if p >= 90),
                'good': sum(1 for p in percentages if 80 <= p < 90),
                'average': sum(1 for p in percentages if 60 <= p < 80),
                'below_average': sum(1 for p in percentages if 40 <= p < 60),
                'poor': sum(1 for p in percentages if p < 40)
            }

            return {
                **db_stats,
                'subject_statistics': subject_stats,
                'score_distribution': distribution
            }

        return db_stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")

@app.delete("/results/{result_id}")
async def delete_result(result_id: int):
    """Delete a specific result"""
    # This would require implementing delete functionality in DatabaseManager
    raise HTTPException(status_code=501, detail="Delete functionality not implemented")

async def cleanup_old_files():
    """Background task to clean up old uploaded files"""
    try:
        import time
        current_time = time.time()

        # Clean up files older than 1 hour
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and current_time - file_path.stat().st_mtime > 3600:
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)