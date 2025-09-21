from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ScoringEngine:
    """
    OMR Scoring Engine

    Compares extracted answers with answer keys and calculates scores
    """

    def __init__(self, answer_key_loader):
        self.answer_key_loader = answer_key_loader
        self.subjects = ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS']

    def score_omr_sheet(self, extracted_answers: Dict[int, Dict],
                       set_letter: str,
                       student_info: Dict = None) -> Dict:
        """
        Score an OMR sheet against the answer key

        Args:
            extracted_answers: Answers extracted from OMR sheet
            set_letter: Set identifier (A, B, etc.)
            student_info: Optional student information

        Returns:
            Complete scoring results with subject-wise and total scores
        """
        try:
            # Get the correct answer key
            answer_key = self.answer_key_loader.get_answer_key(set_letter)

            # Initialize results structure
            results = {
                'student_info': student_info or {},
                'set': set_letter,
                'timestamp': datetime.now().isoformat(),
                'subject_scores': {},
                'total_score': 0,
                'max_possible_score': 100,
                'percentage': 0.0,
                'question_details': {},
                'summary': {
                    'correct': 0,
                    'incorrect': 0,
                    'unanswered': 0,
                    'ambiguous': 0
                },
                'flags': []
            }

            # Initialize subject scores
            for subject in self.subjects:
                results['subject_scores'][subject] = {
                    'correct': 0,
                    'total': 20,
                    'percentage': 0.0,
                    'score': 0
                }

            # Score each question
            for question_num in range(1, 101):  # Questions 1-100
                question_result = self._score_question(
                    question_num,
                    extracted_answers.get(question_num, {}),
                    answer_key.get(question_num, {})
                )

                results['question_details'][question_num] = question_result

                # Update subject scores
                subject = question_result['subject']
                if subject in results['subject_scores']:
                    if question_result['status'] == 'correct':
                        results['subject_scores'][subject]['correct'] += 1
                        results['subject_scores'][subject]['score'] += 1
                        results['summary']['correct'] += 1
                    elif question_result['status'] == 'incorrect':
                        results['summary']['incorrect'] += 1
                    elif question_result['status'] == 'unanswered':
                        results['summary']['unanswered'] += 1
                    elif question_result['status'] == 'ambiguous':
                        results['summary']['ambiguous'] += 1
                        results['flags'].append(f"Question {question_num}: Ambiguous answer")

            # Calculate final scores and percentages
            self._calculate_final_scores(results)

            # Add quality flags
            self._add_quality_flags(results, extracted_answers)

            return results

        except Exception as e:
            logger.error(f"Error scoring OMR sheet: {str(e)}")
            raise

    def _score_question(self, question_num: int,
                       extracted: Dict,
                       answer_key: Dict) -> Dict:
        """
        Score a single question
        """
        result = {
            'question_number': question_num,
            'subject': extracted.get('subject', answer_key.get('subject', 'UNKNOWN')),
            'extracted_answer': extracted.get('answer'),
            'correct_answer': answer_key.get('answer'),
            'status': 'unknown',
            'confidence': extracted.get('confidence', 'unknown'),
            'notes': []
        }

        # Add any extraction notes
        if 'note' in extracted:
            result['notes'].append(extracted['note'])

        # Determine scoring status
        if not answer_key.get('answer'):
            result['status'] = 'no_answer_key'
            result['notes'].append('No answer key available')
        elif not extracted.get('answer'):
            result['status'] = 'unanswered'
        elif extracted.get('confidence') == 'low':
            result['status'] = 'ambiguous'
            result['notes'].append('Low confidence detection')
        elif extracted['answer'] == answer_key['answer']:
            result['status'] = 'correct'
        else:
            result['status'] = 'incorrect'

        return result

    def _calculate_final_scores(self, results: Dict):
        """
        Calculate final scores and percentages
        """
        total_correct = 0

        # Calculate subject percentages
        for subject, scores in results['subject_scores'].items():
            scores['percentage'] = (scores['correct'] / scores['total']) * 100
            total_correct += scores['correct']

        # Calculate total score
        results['total_score'] = total_correct
        results['percentage'] = (total_correct / results['max_possible_score']) * 100

    def _add_quality_flags(self, results: Dict, extracted_answers: Dict):
        """
        Add quality flags based on extraction confidence and patterns
        """
        low_confidence_count = 0
        multiple_marks_count = 0
        no_marks_count = 0

        for question_data in extracted_answers.values():
            confidence = question_data.get('confidence', 'unknown')
            note = question_data.get('note', '')

            if confidence == 'low':
                low_confidence_count += 1
            if 'Multiple marks' in note:
                multiple_marks_count += 1
            if 'No mark' in note:
                no_marks_count += 1

        # Add flags based on thresholds
        if low_confidence_count > 10:
            results['flags'].append(f"High number of low-confidence answers: {low_confidence_count}")

        if multiple_marks_count > 5:
            results['flags'].append(f"Multiple marks detected in {multiple_marks_count} questions")

        if no_marks_count > 10:
            results['flags'].append(f"No marks detected in {no_marks_count} questions")

        # Overall quality score
        total_questions = len(extracted_answers)
        if total_questions > 0:
            quality_score = ((total_questions - low_confidence_count - multiple_marks_count)
                           / total_questions) * 100
            results['quality_score'] = quality_score

            if quality_score < 80:
                results['flags'].append(f"Low quality score: {quality_score:.1f}%")

    def generate_detailed_report(self, results: Dict) -> str:
        """
        Generate a detailed text report
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("OMR EVALUATION REPORT")
        report_lines.append("=" * 50)

        # Student info
        student_info = results.get('student_info', {})
        if student_info:
            report_lines.append(f"Student: {student_info.get('name', 'N/A')}")
            report_lines.append(f"ID: {student_info.get('id', 'N/A')}")

        report_lines.append(f"Set: {results['set']}")
        report_lines.append(f"Timestamp: {results['timestamp']}")
        report_lines.append("")

        # Overall scores
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Score: {results['total_score']}/100")
        report_lines.append(f"Percentage: {results['percentage']:.1f}%")
        report_lines.append("")

        # Subject-wise scores
        report_lines.append("SUBJECT-WISE SCORES")
        report_lines.append("-" * 30)
        for subject, scores in results['subject_scores'].items():
            report_lines.append(f"{subject:<12}: {scores['score']}/20 ({scores['percentage']:.1f}%)")
        report_lines.append("")

        # Summary statistics
        summary = results['summary']
        report_lines.append("ANSWER SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Correct:     {summary['correct']}")
        report_lines.append(f"Incorrect:   {summary['incorrect']}")
        report_lines.append(f"Unanswered:  {summary['unanswered']}")
        report_lines.append(f"Ambiguous:   {summary['ambiguous']}")
        report_lines.append("")

        # Quality flags
        if results['flags']:
            report_lines.append("QUALITY FLAGS")
            report_lines.append("-" * 30)
            for flag in results['flags']:
                report_lines.append(f"• {flag}")
            report_lines.append("")

        # Quality score
        if 'quality_score' in results:
            report_lines.append(f"Quality Score: {results['quality_score']:.1f}%")

        return "\n".join(report_lines)

    def export_results_to_dict(self, results: Dict) -> Dict:
        """
        Export results in a format suitable for JSON/CSV export
        """
        export_data = {
            'timestamp': results['timestamp'],
            'set': results['set'],
            'total_score': results['total_score'],
            'percentage': results['percentage'],
        }

        # Add student info if available
        if results.get('student_info'):
            export_data.update(results['student_info'])

        # Add subject scores
        for subject, scores in results['subject_scores'].items():
            export_data[f'{subject}_score'] = scores['score']
            export_data[f'{subject}_percentage'] = scores['percentage']

        # Add summary
        export_data.update({
            'correct_answers': results['summary']['correct'],
            'incorrect_answers': results['summary']['incorrect'],
            'unanswered': results['summary']['unanswered'],
            'ambiguous_answers': results['summary']['ambiguous'],
            'quality_score': results.get('quality_score', 0)
        })

        return export_data