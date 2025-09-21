import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class AnswerKeyLoader:
    """
    Load and parse answer keys from Excel files
    """

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.answer_keys = {}
        self._load_answer_keys()

    def _load_answer_keys(self):
        """
        Load answer keys from Excel file for both Set A and Set B
        """
        try:
            # Load Set A
            df_a = pd.read_excel(self.excel_path, sheet_name='Set - A')
            self.answer_keys['A'] = self._parse_answer_sheet(df_a, 'Set A')
            logger.info(f"Loaded {len(self.answer_keys['A'])} answers for Set A")

            # Load Set B
            df_b = pd.read_excel(self.excel_path, sheet_name='Set - B')
            # Remove the first row if it's empty (as seen in the data)
            df_b = df_b.dropna(how='all')
            self.answer_keys['B'] = self._parse_answer_sheet(df_b, 'Set B')
            logger.info(f"Loaded {len(self.answer_keys['B'])} answers for Set B")

        except Exception as e:
            logger.error(f"Error loading answer keys: {str(e)}")
            raise

    def _parse_answer_sheet(self, df: pd.DataFrame, set_name: str) -> Dict[int, Dict]:
        """
        Parse answer sheet DataFrame and extract answers
        """
        answers = {}

        # Column mapping (remove leading/trailing spaces)
        columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

        # Expected subjects in order
        subjects = ['Python', 'EDA', 'SQL', 'POWER BI']

        # Find statistics column (handle different naming)
        stats_col = None
        for col in columns:
            if col and ('stat' in col.lower() or 'satistic' in col):
                stats_col = col
                break

        if stats_col:
            subjects.append(stats_col)

        # Parse each subject column
        question_num = 1

        for row_idx in range(min(20, len(df))):  # 20 questions per subject
            for subject_idx, subject in enumerate(subjects):
                if subject in df.columns:
                    cell_value = df[subject].iloc[row_idx]

                    if pd.notna(cell_value):
                        # Extract answer from format like "1 - a" or "21 - b"
                        answer = self._extract_answer(str(cell_value))
                        if answer:
                            answers[question_num] = {
                                'subject': self._normalize_subject_name(subject),
                                'answer': answer.lower(),
                                'original': str(cell_value)
                            }

                    question_num += 1

                    if question_num > 100:  # Cap at 100 questions
                        break

            if question_num > 100:
                break

        return answers

    def _extract_answer(self, cell_value: str) -> Optional[str]:
        """
        Extract answer letter from cell value like "1 - a" or "21. b"
        """
        # Remove extra spaces and normalize
        cell_value = cell_value.strip()

        # Pattern to match question number followed by answer
        patterns = [
            r'\d+\s*[-\.]\s*([a-dA-D])',  # "1 - a" or "1. a"
            r'([a-dA-D])$',               # Just "a" at the end
        ]

        for pattern in patterns:
            match = re.search(pattern, cell_value)
            if match:
                return match.group(1).lower()

        logger.warning(f"Could not extract answer from: {cell_value}")
        return None

    def _normalize_subject_name(self, subject: str) -> str:
        """
        Normalize subject names to match detection output
        """
        subject = subject.strip().upper()
        mapping = {
            'PYTHON': 'PYTHON',
            'EDA': 'EDA',
            'SQL': 'SQL',
            'POWER BI': 'POWER BI',
            'SATISTICS': 'ADV STATS',
            'STATISTICS': 'ADV STATS'
        }
        return mapping.get(subject, subject)

    def get_answer_key(self, set_letter: str) -> Dict[int, Dict]:
        """
        Get answer key for specific set (A or B)
        """
        set_letter = set_letter.upper()
        if set_letter not in self.answer_keys:
            raise ValueError(f"Answer key for set {set_letter} not found")
        return self.answer_keys[set_letter]

    def get_all_answer_keys(self) -> Dict[str, Dict]:
        """
        Get all loaded answer keys
        """
        return self.answer_keys