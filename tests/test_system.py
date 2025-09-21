#!/usr/bin/env python3
"""
Test script for OMR System
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.omr_processor import OMRProcessor
from core.answer_key_loader import AnswerKeyLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_answer_key_loading():
    """Test loading answer keys from Excel"""
    try:
        answer_key_path = "../data/Key (Set A and B).xlsx"
        loader = AnswerKeyLoader(answer_key_path)

        # Test Set A
        set_a_answers = loader.get_answer_key('A')
        logger.info(f"Set A: Loaded {len(set_a_answers)} answers")

        # Test Set B
        set_b_answers = loader.get_answer_key('B')
        logger.info(f"Set B: Loaded {len(set_b_answers)} answers")

        # Print first few answers from each set
        logger.info("First 5 answers from Set A:")
        for i in range(1, 6):
            if i in set_a_answers:
                answer = set_a_answers[i]
                logger.info(f"  Q{i}: {answer['subject']} -> {answer['answer']}")

        logger.info("First 5 answers from Set B:")
        for i in range(1, 6):
            if i in set_b_answers:
                answer = set_b_answers[i]
                logger.info(f"  Q{i}: {answer['subject']} -> {answer['answer']}")

        return True

    except Exception as e:
        logger.error(f"Answer key loading failed: {str(e)}")
        return False

def test_omr_processor_initialization():
    """Test OMR processor initialization"""
    try:
        answer_key_path = "../data/Key (Set A and B).xlsx"
        processor = OMRProcessor(answer_key_path)

        stats = processor.get_processing_statistics()
        logger.info("OMR Processor initialized successfully")
        logger.info(f"System statistics: {stats}")

        return True

    except Exception as e:
        logger.error(f"OMR processor initialization failed: {str(e)}")
        return False

def test_sample_image_processing():
    """Test processing a sample image"""
    try:
        # Check if sample images exist
        sample_images = [
            "../../Theme 1 - Sample Data/Set A/Img1.jpeg",
            "../../Theme 1 - Sample Data/Set B/Img9.jpeg"
        ]

        answer_key_path = "../data/Key (Set A and B).xlsx"
        processor = OMRProcessor(answer_key_path)

        for i, image_path in enumerate(sample_images):
            sample_path = Path(__file__).parent / image_path
            if sample_path.exists():
                logger.info(f"Processing sample image: {sample_path}")

                set_letter = 'A' if 'Set A' in str(image_path) else 'B'

                result = processor.process_omr_sheet(
                    str(sample_path),
                    set_letter,
                    student_info={'name': f'Test Student {i+1}', 'id': f'TEST{i+1:03d}'},
                    save_debug=True,
                    output_dir="../results/test_output"
                )

                logger.info(f"Processing completed successfully!")
                logger.info(f"Score: {result['total_score']}/100 ({result['percentage']:.1f}%)")

                # Log subject-wise scores
                for subject, scores in result['subject_scores'].items():
                    logger.info(f"  {subject}: {scores['score']}/20")

                # Log any flags
                if result['flags']:
                    logger.warning("Quality flags:")
                    for flag in result['flags']:
                        logger.warning(f"  - {flag}")

                return True

            else:
                logger.warning(f"Sample image not found: {sample_path}")

        return False

    except Exception as e:
        logger.error(f"Sample image processing failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("OMR SYSTEM TEST SUITE")
    logger.info("=" * 50)

    tests = [
        ("Answer Key Loading", test_answer_key_loading),
        ("OMR Processor Initialization", test_omr_processor_initialization),
        ("Sample Image Processing", test_sample_image_processing)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            failed += 1

    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "No tests run")

    if failed == 0:
        logger.info("🎉 All tests passed! System is ready.")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs.")

if __name__ == "__main__":
    main()