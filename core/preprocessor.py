import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OMRPreprocessor:
    """
    OMR Sheet Image Preprocessing Module

    Handles image preprocessing tasks including:
    - Orientation detection and correction
    - Perspective correction
    - Illumination normalization
    - Noise reduction
    """

    def __init__(self):
        self.debug = False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for OMR sheets

        Args:
            image_path: Path to the OMR sheet image

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

            logger.info(f"Loaded image with shape: {image.shape}")

            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Step 1: Detect and correct orientation
            corrected_image = self.correct_orientation(gray)

            # Step 2: Detect corners and apply perspective correction
            perspective_corrected = self.correct_perspective(corrected_image)

            # Step 3: Normalize illumination
            illumination_corrected = self.normalize_illumination(perspective_corrected)

            # Step 4: Remove noise
            denoised = self.remove_noise(illumination_corrected)

            # Step 5: Enhance contrast
            enhanced = self.enhance_contrast(denoised)

            return enhanced

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation using text/line detection
        """
        # Detect lines to determine orientation
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            # Calculate dominant angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # Find the most common angle (rounded to nearest 5 degrees)
            angles = [round(angle/5)*5 for angle in angles]
            dominant_angle = max(set(angles), key=angles.count) if angles else 0

            # Correct small rotations
            if abs(dominant_angle) > 2:
                rows, cols = image.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -dominant_angle, 1)
                image = cv2.warpAffine(image, rotation_matrix, (cols, rows),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                logger.info(f"Corrected rotation by {dominant_angle} degrees")

        return image

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Detect corners and apply perspective correction
        """
        # Find contours to detect the OMR sheet boundary
        edges = cv2.Canny(image, 30, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour (likely the OMR sheet)
        largest_contour = max(contours, key=cv2.contourArea) if contours else None

        if largest_contour is not None and cv2.contourArea(largest_contour) > 50000:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # If we found a 4-sided polygon, apply perspective correction
            if len(approx) == 4:
                corners = np.array([point[0] for point in approx], dtype=np.float32)

                # Order corners: top-left, top-right, bottom-right, bottom-left
                corners = self._order_corners(corners)

                # Define destination points for perspective correction
                h, w = image.shape
                dst_corners = np.array([
                    [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
                ], dtype=np.float32)

                # Apply perspective transformation
                perspective_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
                corrected = cv2.warpPerspective(image, perspective_matrix, (w, h))

                logger.info("Applied perspective correction")
                return corrected

        return image

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left"""
        # Calculate center point
        center = np.mean(corners, axis=0)

        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        corners = sorted(corners, key=angle_from_center)

        # Reorder to start from top-left
        corners = np.array(corners)
        top_left_idx = np.argmin(np.sum(corners, axis=1))
        corners = np.roll(corners, -top_left_idx, axis=0)

        return corners

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize illumination using morphological operations
        """
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

        # Morphological opening to estimate background
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Subtract background and normalize
        normalized = cv2.subtract(image, background)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise using bilateral filtering
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def resize_to_standard(self, image: np.ndarray, target_width: int = 2000) -> np.ndarray:
        """
        Resize image to standard width while maintaining aspect ratio
        """
        h, w = image.shape
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)

        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized