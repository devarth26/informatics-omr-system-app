"""
Configuration settings for OMR System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Answer key configuration
ANSWER_KEY_PATH = DATA_DIR / "Key (Set A and B).xlsx"

# Image processing settings
PREPROCESSING_CONFIG = {
    "target_width": 2000,
    "gaussian_blur_kernel": (3, 3),
    "adaptive_threshold_block_size": 11,
    "adaptive_threshold_c": 2,
    "canny_low_threshold": 50,
    "canny_high_threshold": 150,
    "min_bubble_area": 50,
    "max_bubble_area": 800,
    "min_circularity": 0.3
}

# Bubble detection settings
BUBBLE_DETECTION_CONFIG = {
    "subjects": ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS'],
    "questions_per_subject": 20,
    "options_per_question": 4,
    "total_questions": 100,
    "darkness_threshold": 0.3,  # Threshold for marked bubble detection
    "confidence_threshold": 0.7
}

# Scoring settings
SCORING_CONFIG = {
    "points_per_correct": 1,
    "points_per_incorrect": 0,
    "points_per_unanswered": 0,
    "quality_threshold": 80.0,  # Minimum quality score for reliable results
    "max_ambiguous_answers": 10,
    "max_multiple_marks": 5
}

# Database settings
DATABASE_CONFIG = {
    "url": "sqlite:///./omr_results.db",
    "echo": False  # Set to True for SQL query logging
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "cors_origins": ["*"],  # Configure appropriately for production
    "upload_max_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": {".jpg", ".jpeg", ".png"},
    "cleanup_interval_hours": 1
}

# Frontend settings
FRONTEND_CONFIG = {
    "api_base_url": "http://localhost:8000",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "results_per_page": 50,
    "chart_height": 400
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "omr_system.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 95,
    "good": 85,
    "fair": 70,
    "poor": 50,
    "unreliable": 30
}

# Export settings
EXPORT_CONFIG = {
    "csv_encoding": "utf-8",
    "include_debug_info": False,
    "date_format": "%Y-%m-%d %H:%M:%S"
}