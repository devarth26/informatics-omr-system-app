import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class BubbleDetector:
    """
    OMR Bubble Detection and Classification System

    Detects bubble positions and classifies them as marked/unmarked
    """

    def __init__(self):
        self.bubble_contours = []
        self.bubble_positions = {}
        self.debug = False

        # OMR sheet configuration based on analyzed samples
        self.subjects = ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS']
        self.questions_per_subject = 20
        self.options_per_question = 4
        self.total_questions = 100

    def detect_bubbles(self, image: np.ndarray) -> Dict[int, Dict[str, int]]:
        """
        Main bubble detection pipeline

        Args:
            image: Preprocessed OMR sheet image

        Returns:
            Dictionary mapping question numbers to detected answers
            Format: {question_num: {'subject': subject_name, 'answer': option_letter}}
        """
        try:
            # Step 1: Detect all bubble contours
            bubble_contours = self._find_bubble_contours(image)
            logger.info(f"Found {len(bubble_contours)} potential bubbles")

            # Step 2: Filter and organize bubbles into grid
            organized_bubbles = self._organize_bubbles_into_grid(bubble_contours, image.shape)

            # Step 3: Classify each bubble as marked/unmarked
            answers = self._classify_bubbles(image, organized_bubbles)

            return answers

        except Exception as e:
            logger.error(f"Error detecting bubbles: {str(e)}")
            raise

    def _find_bubble_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find all potential bubble contours using improved detection for OMR sheets
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use multiple thresholding approaches to catch all bubbles
        bubble_contours = []

        # Method 1: Adaptive threshold (good for varying lighting)
        binary1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_contours.extend(self._filter_bubble_contours(contours1, method="adaptive"))

        # Method 2: Otsu threshold (good for clear separation)
        _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_contours.extend(self._filter_bubble_contours(contours2, method="otsu"))

        # Method 3: Fixed threshold (backup method)
        _, binary3 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        contours3, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_contours.extend(self._filter_bubble_contours(contours3, method="fixed"))

        # Remove duplicate contours and merge results
        filtered_contours = self._remove_duplicate_contours(bubble_contours)

        logger.info(f"Found {len(filtered_contours)} unique bubble contours after filtering")
        return filtered_contours

    def _filter_bubble_contours(self, contours: List[np.ndarray], method: str = "adaptive") -> List[np.ndarray]:
        """
        Filter contours to identify actual bubbles based on size, shape, and position
        """
        bubble_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Very inclusive area filtering to catch ALL bubbles (filled and unfilled)
            # OMR bubbles can vary significantly in detected size
            if 30 < area < 1000:
                # Check aspect ratio - very lenient to catch all bubble shapes
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Very lenient aspect ratio - accept almost any reasonable shape
                if 0.3 < aspect_ratio < 3.0:
                    # Check circularity - very lenient to catch all bubble-like shapes
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)

                        # Very lenient circularity - accept most shapes
                        if circularity > 0.2:
                            # Check solidity - very lenient
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            if hull_area > 0:
                                solidity = area / hull_area

                                # Very lenient solidity - just filter out very hollow shapes
                                if solidity > 0.3:
                                    bubble_contours.append(contour)

        return bubble_contours

    def _remove_duplicate_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove duplicate or overlapping contours
        """
        if not contours:
            return []

        # Calculate centers for all contours
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy, contour))

        # Remove duplicates based on center proximity
        unique_contours = []
        used_indices = set()

        for i, (cx1, cy1, contour1) in enumerate(centers):
            if i in used_indices:
                continue

            # Check if this contour is too close to any already selected
            is_duplicate = False
            for j, (cx2, cy2, contour2) in enumerate(centers):
                if j <= i or j in used_indices:
                    continue

                # Calculate distance between centers
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

                # If centers are very close (within 20 pixels), consider it a duplicate
                if distance < 20:
                    # Keep the contour with better circularity
                    area1 = cv2.contourArea(contour1)
                    perimeter1 = cv2.arcLength(contour1, True)
                    circularity1 = 4 * np.pi * area1 / (perimeter1 * perimeter1) if perimeter1 > 0 else 0

                    area2 = cv2.contourArea(contour2)
                    perimeter2 = cv2.arcLength(contour2, True)
                    circularity2 = 4 * np.pi * area2 / (perimeter2 * perimeter2) if perimeter2 > 0 else 0

                    if circularity1 >= circularity2:
                        used_indices.add(j)
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_contours.append(contour1)
                used_indices.add(i)

        return unique_contours

    def _organize_bubbles_into_grid(self, contours: List[np.ndarray], image_shape: Tuple) -> Dict:
        """
        Organize detected bubbles into a structured grid based on column-based layout

        Layout: 5 subjects in columns, 20 questions per subject (rows), 4 options per question
        Structure: PYTHON | DATA ANALYSIS | MySQL | POWER BI | Adv STATS
        """
        if not contours:
            return {}

        # Calculate centers of all bubbles
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy, contour))

        if not centers:
            return {}

        logger.info(f"Found {len(centers)} bubble centers to organize")

        # Sort by x-coordinate first (columns), then y-coordinate (rows)
        centers.sort(key=lambda x: (x[0], x[1]))

        # Group bubbles by columns (subjects) using clustering
        x_positions = [center[0] for center in centers]
        columns = self._cluster_positions(x_positions, expected_clusters=5)  # 5 subject columns

        organized = {}

        # Process each column (subject)
        for col_idx, col_x in enumerate(sorted(set(columns))):
            # Get bubbles in this column
            col_bubbles = [center for center, col_cluster in zip(centers, columns) if col_cluster == col_x]

            if len(col_bubbles) < 4:  # Need at least 4 bubbles for one question
                continue

            logger.info(f"Column {col_idx}: {len(col_bubbles)} bubbles")

            # Sort column bubbles by y-coordinate (top to bottom)
            col_bubbles.sort(key=lambda x: x[1])

            # Group bubbles by rows (questions) within this column using adaptive clustering
            y_positions_col = [bubble[1] for bubble in col_bubbles]

            # Use adaptive number of clusters based on available bubbles
            # Each question should have 4 bubbles, so estimate number of questions
            estimated_questions = max(1, len(col_bubbles) // 4)
            max_clusters = min(estimated_questions, len(set(y_positions_col)))

            rows_in_col = self._cluster_positions(y_positions_col, expected_clusters=max_clusters)

            # Process each row (question) in this column
            for row_idx, row_y in enumerate(sorted(set(rows_in_col))):
                # Get bubbles in this specific row of this column
                question_bubbles = [bubble for bubble, row_cluster in zip(col_bubbles, rows_in_col) if row_cluster == row_y]

                # Should have 4 bubbles (A, B, C, D) per question
                # Allow 2 or more bubbles to form a valid question (very lenient)
                if len(question_bubbles) >= 2:
                    # Sort bubbles left to right (A, B, C, D)
                    question_bubbles.sort(key=lambda x: x[0])

                    # Calculate question number based on column and row
                    question_num = col_idx * 20 + row_idx + 1

                    if question_num <= 100:
                        organized[question_num] = {
                            'subject': self._get_subject_for_question_column_based(col_idx),
                            'bubbles': [(bubble[0], bubble[1], bubble[2]) for bubble in question_bubbles],
                            'options': ['a', 'b', 'c', 'd'][:len(question_bubbles)]
                        }

        logger.info(f"Successfully organized {len(organized)} questions")
        return organized

    def _cluster_positions(self, positions: List[int], expected_clusters: int) -> List[int]:
        """
        Cluster positions using KMeans to group bubbles by rows/columns
        """
        if len(positions) < expected_clusters:
            return list(range(len(positions)))

        positions_array = np.array(positions).reshape(-1, 1)
        kmeans = KMeans(n_clusters=min(expected_clusters, len(positions)), random_state=42)
        clusters = kmeans.fit_predict(positions_array)

        return clusters.tolist()

    def _get_subject_for_question(self, question_num: int) -> str:
        """
        Determine subject based on question number (1-100) - Legacy row-based mapping
        """
        if 1 <= question_num <= 20:
            return 'PYTHON'
        elif 21 <= question_num <= 40:
            return 'EDA'
        elif 41 <= question_num <= 60:
            return 'SQL'
        elif 61 <= question_num <= 80:
            return 'POWER BI'
        elif 81 <= question_num <= 100:
            return 'ADV STATS'
        else:
            return 'UNKNOWN'

    def _get_subject_for_question_column_based(self, column_idx: int) -> str:
        """
        Determine subject based on column index (0-4) for column-based layout
        Column order: PYTHON | DATA ANALYSIS | MySQL | POWER BI | Adv STATS
        """
        subjects = ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS']
        if 0 <= column_idx < len(subjects):
            return subjects[column_idx]
        else:
            return 'UNKNOWN'

    def _classify_bubbles(self, image: np.ndarray, organized_bubbles: Dict) -> Dict[int, Dict[str, str]]:
        """
        Classify each bubble as marked or unmarked and extract answers
        """
        answers = {}

        for question_num, bubble_data in organized_bubbles.items():
            bubbles = bubble_data['bubbles']
            options = bubble_data['options']
            subject = bubble_data['subject']

            marked_options = []

            for idx, (x, y, contour) in enumerate(bubbles):
                if idx < len(options):
                    # Extract bubble region
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)

                    # Calculate darkness ratio (filled vs unfilled) with improved method
                    bubble_pixels = image[mask == 255]
                    if len(bubble_pixels) > 0:
                        # Use multiple methods to detect filled bubbles

                        # Method 1: Darkness ratio - pixels darker than threshold
                        darkness_ratio = np.sum(bubble_pixels < 100) / len(bubble_pixels)

                        # Method 2: Mean intensity - lower means more filled
                        mean_intensity = np.mean(bubble_pixels)

                        # Method 3: Standard deviation - filled bubbles have lower variance
                        intensity_std = np.std(bubble_pixels)

                        # Combined scoring for more accurate detection
                        is_filled = False

                        # Primary check: significant darkness
                        if darkness_ratio > 0.4:  # More than 40% dark pixels
                            is_filled = True
                        # Secondary check: low mean intensity (very dark overall)
                        elif mean_intensity < 80:
                            is_filled = True
                        # Tertiary check: moderate darkness with low variance (consistently dark)
                        elif darkness_ratio > 0.25 and mean_intensity < 120 and intensity_std < 40:
                            is_filled = True

                        if is_filled:
                            marked_options.append(options[idx])

            # Determine final answer
            if len(marked_options) == 1:
                answers[question_num] = {
                    'subject': subject,
                    'answer': marked_options[0],
                    'confidence': 'high'
                }
            elif len(marked_options) > 1:
                # Multiple marks - take the darkest one or flag as ambiguous
                answers[question_num] = {
                    'subject': subject,
                    'answer': marked_options[0],  # Take first for now
                    'confidence': 'low',
                    'note': 'Multiple marks detected'
                }
            else:
                # No marks detected
                answers[question_num] = {
                    'subject': subject,
                    'answer': None,
                    'confidence': 'none',
                    'note': 'No mark detected'
                }

        return answers

    def visualize_detection(self, image: np.ndarray, organized_bubbles: Dict,
                          answers: Dict) -> np.ndarray:
        """
        Create visualization of detected bubbles and answers for debugging
        """
        # Convert to color image for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        colors = {
            'detected': (0, 255, 0),    # Green for detected bubbles
            'marked': (0, 0, 255),      # Red for marked bubbles
            'unmarked': (255, 0, 0)     # Blue for unmarked bubbles
        }

        for question_num, bubble_data in organized_bubbles.items():
            answer_data = answers.get(question_num, {})
            marked_answer = answer_data.get('answer')

            for idx, (x, y, contour) in enumerate(bubble_data['bubbles']):
                option = bubble_data['options'][idx] if idx < len(bubble_data['options']) else '?'

                # Determine color based on marking status
                if option == marked_answer:
                    color = colors['marked']
                else:
                    color = colors['unmarked']

                # Draw contour and center point
                cv2.drawContours(vis_image, [contour], -1, color, 2)
                cv2.circle(vis_image, (x, y), 3, color, -1)

                # Add question number and option label
                cv2.putText(vis_image, f"Q{question_num}{option}",
                           (x-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return vis_image