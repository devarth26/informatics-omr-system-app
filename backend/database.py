from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class OMRResult(Base):
    __tablename__ = "omr_results"

    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String, nullable=True)
    student_id = Column(String, nullable=True)
    set_letter = Column(String, nullable=False)
    total_score = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    python_score = Column(Integer, nullable=True)
    eda_score = Column(Integer, nullable=True)
    sql_score = Column(Integer, nullable=True)
    powerbi_score = Column(Integer, nullable=True)
    stats_score = Column(Integer, nullable=True)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    image_filename = Column(String, nullable=True)
    raw_results = Column(JSON, nullable=True)  # Store complete results as JSON
    quality_score = Column(Float, nullable=True)
    flags = Column(Text, nullable=True)  # Store flags as text

    def to_dict(self):
        return {
            'id': self.id,
            'student_name': self.student_name,
            'student_id': self.student_id,
            'set_letter': self.set_letter,
            'total_score': self.total_score,
            'percentage': self.percentage,
            'python_score': self.python_score,
            'eda_score': self.eda_score,
            'sql_score': self.sql_score,
            'powerbi_score': self.powerbi_score,
            'stats_score': self.stats_score,
            'processing_timestamp': self.processing_timestamp.isoformat() if self.processing_timestamp else None,
            'image_filename': self.image_filename,
            'quality_score': self.quality_score,
            'flags': json.loads(self.flags) if self.flags else []
        }

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///./omr_results.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def save_result(self, result_data: dict) -> int:
        """Save OMR processing result to database"""
        session = self.get_session()
        try:
            # Extract subject scores
            subject_scores = result_data.get('subject_scores', {})

            db_result = OMRResult(
                student_name=result_data.get('student_info', {}).get('name'),
                student_id=result_data.get('student_info', {}).get('id'),
                set_letter=result_data.get('set'),
                total_score=result_data.get('total_score'),
                percentage=result_data.get('percentage'),
                python_score=subject_scores.get('PYTHON', {}).get('score'),
                eda_score=subject_scores.get('EDA', {}).get('score'),
                sql_score=subject_scores.get('SQL', {}).get('score'),
                powerbi_score=subject_scores.get('POWER BI', {}).get('score'),
                stats_score=subject_scores.get('ADV STATS', {}).get('score'),
                image_filename=result_data.get('processing_metadata', {}).get('image_path'),
                raw_results=result_data,
                quality_score=result_data.get('quality_score'),
                flags=json.dumps(result_data.get('flags', []))
            )

            session.add(db_result)
            session.commit()
            result_id = db_result.id
            return result_id
        finally:
            session.close()

    def get_result(self, result_id: int) -> dict:
        """Get specific result by ID"""
        session = self.get_session()
        try:
            result = session.query(OMRResult).filter(OMRResult.id == result_id).first()
            return result.to_dict() if result else None
        finally:
            session.close()

    def get_all_results(self, limit: int = 100, offset: int = 0) -> list:
        """Get all results with pagination"""
        session = self.get_session()
        try:
            results = session.query(OMRResult).offset(offset).limit(limit).all()
            return [result.to_dict() for result in results]
        finally:
            session.close()

    def get_statistics(self) -> dict:
        """Get database statistics"""
        session = self.get_session()
        try:
            total_count = session.query(OMRResult).count()
            if total_count == 0:
                return {'total_results': 0}

            # Get average scores
            results = session.query(OMRResult).all()
            scores = [r.total_score for r in results]
            percentages = [r.percentage for r in results]

            return {
                'total_results': total_count,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'average_percentage': sum(percentages) / len(percentages) if percentages else 0,
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0
            }
        finally:
            session.close()