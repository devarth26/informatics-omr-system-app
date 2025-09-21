import cv2
import numpy as np
import json
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from .preprocessor import OMRPreprocessor
from .bubble_detector import BubbleDetector
from .answer_key_loader import AnswerKeyLoader
from .scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)

class OMRProcessor:
    """
    Main OMR Processing Engine

    Coordinates all components to process OMR sheets end-to-end
    """

    def __init__(self, answer_key_path: str):
        """
        Initialize OMR processor with answer keys

        Args:
            answer_key_path: Path to Excel file containing answer keys
        """
        self.preprocessor = OMRPreprocessor()
        self.bubble_detector = BubbleDetector()

        # Load answer keys
        try:
            self.answer_key_loader = AnswerKeyLoader(answer_key_path)
            self.scoring_engine = ScoringEngine(self.answer_key_loader)
            logger.info("OMR Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OMR processor: {str(e)}")
            raise

    def process_omr_sheet(self, image_path: str,
                         set_letter: str,
                         student_info: Dict = None,
                         save_debug: bool = False,
                         output_dir: str = None) -> Dict:
        """
        Complete OMR processing pipeline

        Args:
            image_path: Path to OMR sheet image
            set_letter: Answer set identifier (A, B, etc.)
            student_info: Optional student information
            save_debug: Whether to save debug images
            output_dir: Directory to save outputs

        Returns:
            Complete processing results including scores and metadata
        """
        try:
            logger.info(f"Processing OMR sheet: {image_path}")

            # Step 1: Preprocess image
            logger.info("Step 1: Preprocessing image...")
            preprocessed_image = self.preprocessor.preprocess_image(image_path)

            # Step 2: Detect and classify bubbles
            logger.info("Step 2: Detecting bubbles...")
            extracted_answers = self.bubble_detector.detect_bubbles(preprocessed_image)
            logger.info(f"Extracted {len(extracted_answers)} answers")

            # Step 3: Score against answer key
            logger.info("Step 3: Scoring answers...")
            results = self.scoring_engine.score_omr_sheet(
                extracted_answers, set_letter, student_info
            )

            # Step 4: Add processing metadata
            results['processing_metadata'] = {
                'image_path': str(image_path),
                'image_size': preprocessed_image.shape,
                'bubbles_detected': len(extracted_answers),
                'set_used': set_letter
            }

            # Step 5: Save debug outputs if requested
            if save_debug and output_dir:
                self._save_debug_outputs(
                    preprocessed_image, extracted_answers, results, output_dir, image_path
                )

            logger.info(f"Processing completed. Score: {results['total_score']}/100")
            return results

        except Exception as e:
            logger.error(f"Error processing OMR sheet {image_path}: {str(e)}")
            raise

    def detect_set_from_image(self, image_path: str) -> str:
        """
        Detect set letter from OMR sheet image using OCR or pattern matching

        For now, returns 'A' as default. In production, this would use OCR
        to detect the set marking on the sheet.
        """
        # TODO: Implement OCR-based set detection
        # For now, return A as default
        logger.info("Set detection not implemented, defaulting to Set A")
        return 'A'

    def batch_process(self, image_paths: list,
                     set_letter: str = None,
                     output_dir: str = None) -> Dict:
        """
        Process multiple OMR sheets in batch

        Args:
            image_paths: List of paths to OMR sheet images
            set_letter: Set identifier (if None, will attempt auto-detection)
            output_dir: Directory to save results

        Returns:
            Batch processing results with individual and summary statistics
        """
        batch_results = {
            'processed_count': 0,
            'successful_count': 0,
            'failed_count': 0,
            'results': {},
            'summary_statistics': {},
            'processing_errors': []
        }

        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")

                # Auto-detect set if not provided
                current_set = set_letter or self.detect_set_from_image(image_path)

                # Process individual sheet
                result = self.process_omr_sheet(
                    image_path,
                    current_set,
                    student_info={'id': f'student_{i+1}'},
                    save_debug=bool(output_dir),
                    output_dir=output_dir
                )

                batch_results['results'][str(image_path)] = result
                batch_results['successful_count'] += 1

            except Exception as e:
                error_info = {
                    'image_path': str(image_path),
                    'error': str(e)
                }
                batch_results['processing_errors'].append(error_info)
                batch_results['failed_count'] += 1
                logger.error(f"Failed to process {image_path}: {str(e)}")

            batch_results['processed_count'] += 1

        # Generate summary statistics
        if batch_results['successful_count'] > 0:
            batch_results['summary_statistics'] = self._generate_batch_statistics(
                batch_results['results']
            )

        # Save batch results
        if output_dir:
            self._save_batch_results(batch_results, output_dir)

        return batch_results

    def _save_debug_outputs(self, preprocessed_image: np.ndarray,
                           extracted_answers: Dict,
                           results: Dict,
                           output_dir: str,
                           original_image_path: str):
        """
        Save debug outputs including processed images and JSON results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate base filename from original image
        base_name = Path(original_image_path).stem

        try:
            # Save preprocessed image
            preprocessed_path = output_path / f"{base_name}_preprocessed.jpg"
            cv2.imwrite(str(preprocessed_path), preprocessed_image)

            # Create and save bubble detection visualization
            if hasattr(self.bubble_detector, 'bubble_positions') and self.bubble_detector.bubble_positions:
                vis_image = self.bubble_detector.visualize_detection(
                    preprocessed_image,
                    self.bubble_detector.bubble_positions,
                    extracted_answers
                )
                vis_path = output_path / f"{base_name}_bubble_detection.jpg"
                cv2.imwrite(str(vis_path), vis_image)

            # Save JSON results
            results_path = output_path / f"{base_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save detailed report
            report_path = output_path / f"{base_name}_report.txt"
            with open(report_path, 'w') as f:
                f.write(self.scoring_engine.generate_detailed_report(results))

            logger.info(f"Debug outputs saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving debug outputs: {str(e)}")

    def _generate_batch_statistics(self, results: Dict) -> Dict:
        """
        Generate summary statistics for batch processing
        """
        if not results:
            return {}

        scores = [result['total_score'] for result in results.values()]
        percentages = [result['percentage'] for result in results.values()]

        # Subject-wise statistics
        subject_stats = {}
        for subject in ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS']:
            subject_scores = [
                result['subject_scores'].get(subject, {}).get('score', 0)
                for result in results.values()
            ]
            subject_stats[subject] = {
                'mean_score': np.mean(subject_scores),
                'std_score': np.std(subject_scores),
                'min_score': np.min(subject_scores),
                'max_score': np.max(subject_scores)
            }

        return {
            'total_sheets': len(results),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'mean_percentage': np.mean(percentages),
            'subject_statistics': subject_stats,
            'score_distribution': {
                'above_80': sum(1 for p in percentages if p >= 80),
                'above_60': sum(1 for p in percentages if p >= 60),
                'above_40': sum(1 for p in percentages if p >= 40),
                'below_40': sum(1 for p in percentages if p < 40)
            }
        }

    def _save_batch_results(self, batch_results: Dict, output_dir: str):
        """
        Save batch processing results to files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save complete batch results
            batch_path = output_path / "batch_results.json"
            with open(batch_path, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)

            # Save CSV summary for easy analysis
            if batch_results['successful_count'] > 0:
                self._export_batch_to_csv(batch_results, output_path)

            logger.info(f"Batch results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}")

    def _export_batch_to_csv(self, batch_results: Dict, output_path: Path):
        """
        Export batch results to CSV format
        """
        import pandas as pd

        csv_data = []
        for image_path, result in batch_results['results'].items():
            row_data = self.scoring_engine.export_results_to_dict(result)
            row_data['image_path'] = image_path
            csv_data.append(row_data)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_path / "batch_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV results saved to {csv_path}")

    def get_processing_statistics(self) -> Dict:
        """
        Get processing statistics and system status
        """
        return {
            'answer_keys_loaded': list(self.answer_key_loader.get_all_answer_keys().keys()),
            'subjects': self.bubble_detector.subjects,
            'questions_per_subject': self.bubble_detector.questions_per_subject,
            'total_questions': self.bubble_detector.total_questions,
            'system_ready': True
        }