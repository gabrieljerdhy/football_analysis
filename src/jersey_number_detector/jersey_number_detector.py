#!/usr/bin/env python3
"""
Jersey Number Detection using OCR for Football Analysis

This module provides OCR-based jersey number recognition to improve player identification
in football video analysis. It uses EasyOCR for robust text detection and implements
various preprocessing techniques to optimize accuracy.

Author: AI Assistant
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import easyocr
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JerseyNumberDetector:
    """
    OCR-based jersey number detection system for football players.

    Features:
    - Robust OCR using EasyOCR
    - Image preprocessing for optimal recognition
    - Multi-frame consensus for improved accuracy
    - Confidence scoring and validation
    - Caching mechanism for consistent player identification
    """

    def __init__(
        self,
        languages=["en"],
        confidence_threshold=0.3,
        consensus_frames=5,
        valid_number_range=(1, 99),
    ):
        """
        Initialize the Jersey Number Detector.

        Args:
            languages (list): Languages for OCR recognition
            confidence_threshold (float): Minimum confidence for OCR results
            consensus_frames (int): Number of frames to use for consensus
            valid_number_range (tuple): Valid range for jersey numbers
        """
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.consensus_frames = consensus_frames
        self.valid_number_range = valid_number_range

        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(languages, gpu=True)
            logger.info("EasyOCR initialized with GPU support")
        except Exception as e:
            logger.warning(f"GPU initialization failed, using CPU: {e}")
            self.reader = easyocr.Reader(languages, gpu=False)

        # Player tracking data
        self.player_jersey_history = defaultdict(
            list
        )  # track_id -> list of detected numbers
        self.player_jersey_cache = {}  # track_id -> confirmed jersey number
        self.frame_count = 0

        # Performance tracking
        self.ocr_calls = 0
        self.cache_hits = 0

        # Validation tracking
        self.invalid_detections = defaultdict(
            list
        )  # track invalid numbers for debugging
        self.validation_stats = {
            "total_ocr_results": 0,
            "valid_numbers": 0,
            "invalid_range": 0,
            "invalid_format": 0,
            "filtered_multi_digit": 0,
            "exceeds_100": 0,  # Track numbers >100 specifically
            "high_number_rejected": 0,  # Track 90-99 numbers rejected
            "ocr_artifacts_detected": 0,  # Track OCR artifact rejections
            "suspicious_sequences": 0,  # Track suspicious sequence rejections
            "context_validation_failed": 0,  # Track context validation failures
            "uncommon_numbers_rejected": 0,  # Track uncommon numbers rejected
            "mixed_alphanumeric_rejected": 0,  # Track mixed alphanumeric rejections
        }

    def preprocess_jersey_region(
        self, frame: np.ndarray, bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Extract and preprocess the jersey region from a player bounding box.
        Enhanced with better region extraction and morphological operations.

        Args:
            frame: Input video frame
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            Preprocessed jersey region image or None if processing fails
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Validate bounding box
        if x1 >= x2 or y1 >= y2:
            return None

        # Extract player region with bounds checking
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))

        player_region = frame[y1:y2, x1:x2]

        if player_region.size == 0:
            return None

        # Enhanced jersey region extraction - focus more precisely on chest area
        height, width = player_region.shape[:2]

        # More conservative region to avoid capturing adjacent text/numbers
        chest_y1 = int(height * 0.20)  # Skip head and shoulders
        chest_y2 = int(height * 0.50)  # Focus on upper chest only
        chest_x1 = int(width * 0.25)  # More centered region
        chest_x2 = int(width * 0.75)

        # Ensure minimum region size
        if chest_y2 - chest_y1 < 20 or chest_x2 - chest_x1 < 20:
            # Fallback to larger region if too small
            chest_y1 = int(height * 0.15)
            chest_y2 = int(height * 0.60)
            chest_x1 = int(width * 0.20)
            chest_x2 = int(width * 0.80)

        jersey_region = player_region[chest_y1:chest_y2, chest_x1:chest_x2]

        if jersey_region.size == 0:
            return None

        # Resize for better OCR (minimum 120px height for better accuracy)
        target_height = 120
        if jersey_region.shape[0] < target_height:
            scale_factor = target_height / jersey_region.shape[0]
            new_width = int(jersey_region.shape[1] * scale_factor)
            jersey_region = cv2.resize(jersey_region, (new_width, target_height))

        # Convert to grayscale
        if len(jersey_region.shape) == 3:
            gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = jersey_region

        # Enhanced preprocessing pipeline optimized for jersey number OCR
        # 1. Initial noise reduction with edge-preserving filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. Adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Gaussian blur to smooth out minor artifacts
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 4. Morphological operations to clean up text regions
        # Use different kernels for different operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Close small gaps in text
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close)
        # Remove small noise
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_open)

        # 5. Enhanced unsharp masking for better text clarity
        gaussian = cv2.GaussianBlur(morphed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(morphed, 1.8, gaussian, -0.8, 0)

        # 6. Adaptive thresholding to improve text/background separation
        # This helps OCR distinguish text more clearly
        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 7. Final contrast and brightness adjustment
        final = cv2.convertScaleAbs(adaptive_thresh, alpha=1.1, beta=5)

        # 8. Optional: Apply additional morphological cleaning for very noisy images
        if self._is_noisy_image(final):
            final = self._apply_noise_reduction(final)

        return final

    def _is_noisy_image(self, image: np.ndarray) -> bool:
        """
        Detect if an image is particularly noisy and needs additional processing.

        Args:
            image: Preprocessed grayscale image

        Returns:
            True if the image appears noisy
        """
        # Calculate image statistics to detect noise
        # High standard deviation often indicates noise
        std_dev = np.std(image)

        # Count the number of small connected components (potential noise)
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Count very small contours (likely noise)
        small_contours = sum(1 for contour in contours if cv2.contourArea(contour) < 10)

        # Image is considered noisy if it has high std dev and many small contours
        is_noisy = std_dev > 60 and small_contours > 20

        if is_noisy:
            logger.debug(
                f"Noisy image detected: std_dev={std_dev:.1f}, small_contours={small_contours}"
            )

        return is_noisy

    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply additional noise reduction for very noisy images.

        Args:
            image: Noisy preprocessed image

        Returns:
            Cleaned image
        """
        # Apply median filter to remove salt-and-pepper noise
        median_filtered = cv2.medianBlur(image, 3)

        # Apply morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)

        # Apply morphological closing to fill small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        return closed

    def extract_jersey_number(
        self, preprocessed_image: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Extract jersey number from preprocessed image using OCR with enhanced validation.

        Args:
            preprocessed_image: Preprocessed jersey region

        Returns:
            Tuple of (jersey_number, confidence) or (None, 0.0)
        """
        if preprocessed_image is None:
            return None, 0.0

        self.ocr_calls += 1

        try:
            # Run OCR
            results = self.reader.readtext(preprocessed_image)
            self.validation_stats["total_ocr_results"] += len(results)

            # Process OCR results with enhanced filtering
            valid_candidates = []

            for bbox, text, confidence in results:
                if confidence < self.confidence_threshold:
                    continue

                # Enhanced text processing and validation
                candidate_numbers = self._extract_and_validate_numbers(text, confidence)
                valid_candidates.extend(candidate_numbers)

            # Select best candidate
            if valid_candidates:
                # Sort by confidence and select the best valid number
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                best_number, best_confidence = valid_candidates[0]
                self.validation_stats["valid_numbers"] += 1

                logger.debug(
                    f"Valid jersey number detected: {best_number} (confidence: {best_confidence:.3f})"
                )
                return best_number, best_confidence

            return None, 0.0

        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return None, 0.0

    def _extract_and_validate_numbers(
        self, text: str, confidence: float
    ) -> List[Tuple[int, float]]:
        """
        Extract and validate numbers from OCR text with enhanced filtering.

        Args:
            text: OCR detected text
            confidence: OCR confidence score

        Returns:
            List of (number, confidence) tuples for valid jersey numbers
        """
        valid_numbers = []

        # Enhanced text cleaning - remove common OCR artifacts more intelligently
        cleaned_text = self._clean_ocr_text(text)

        # First, check if the entire cleaned text is a 3+ digit number - if so, reject it entirely
        if re.match(r"^\d{3,}$", cleaned_text.strip()):
            # This is a long number sequence - don't extract anything from it
            logger.debug(f"Rejecting long number sequence: '{cleaned_text}'")
            return valid_numbers

        # Multiple regex patterns to catch different number formats
        patterns = [
            r"\b(\d{1,2})\b",  # Standalone 1-2 digit numbers
            r"^(\d{1,2})$",  # Entire string is 1-2 digits
            r"(\d{1,2})(?=\s|$)",  # 1-2 digits followed by space or end
            r"(?<=\s)(\d{1,2})(?=\s)",  # 1-2 digits surrounded by spaces
        ]

        found_numbers = set()  # Use set to avoid duplicates

        # Only extract numbers if the text doesn't contain long digit sequences
        if not re.search(r"\d{3,}", cleaned_text):
            # Safe to extract - no long sequences present
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text)
                for match in matches:
                    found_numbers.add(match)
        else:
            # Text contains long sequences - be very selective
            # Only extract if the number is clearly separated from long sequences
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text)
                for match in matches:
                    # Check if this number is part of a longer sequence
                    if not self._is_part_of_long_sequence(match, cleaned_text):
                        found_numbers.add(match)
                    else:
                        logger.debug(
                            f"Rejecting {match} as part of long sequence in '{cleaned_text}'"
                        )
                        self.validation_stats["filtered_multi_digit"] += 1

        # Validate each found number
        for num_str in found_numbers:
            try:
                number = int(num_str)

                # Multi-stage validation
                # Stage 1: Basic range and pattern validation
                if not self._is_valid_jersey_number(number, text):
                    self._log_invalid_detection(number, text, "range_validation")
                    continue

                # Stage 2: Context-based validation
                if not self._validate_number_context(number, text, confidence):
                    self._log_invalid_detection(number, text, "context_validation")
                    continue

                # Stage 3: Mixed alphanumeric validation
                if self._is_mixed_alphanumeric_context(text):
                    self.validation_stats["mixed_alphanumeric_rejected"] += 1
                    self._log_invalid_detection(number, text, "mixed_alphanumeric")
                    continue

                # Number passed all validation stages
                valid_numbers.append((number, confidence))

            except ValueError:
                self.validation_stats["invalid_format"] += 1
                self._log_invalid_detection(num_str, text, "format_error")

        return valid_numbers

    def _extract_from_long_sequence(self, sequence: str) -> List[str]:
        """
        Extract valid jersey numbers from long digit sequences.

        Args:
            sequence: Long digit sequence (e.g., "123", "1234")

        Returns:
            List of valid number strings
        """
        valid_extractions = []

        # For sequences like "123", try "12", "23", "1", "2", "3"
        for i in range(len(sequence)):
            for j in range(i + 1, min(i + 3, len(sequence) + 1)):  # Max 2 digits
                candidate = sequence[i:j]
                if len(candidate) <= 2:  # Only 1-2 digit numbers
                    try:
                        num = int(candidate)
                        if (
                            self.valid_number_range[0]
                            <= num
                            <= self.valid_number_range[1]
                        ):
                            valid_extractions.append(candidate)
                    except ValueError:
                        continue

        return valid_extractions

    def _is_part_of_long_sequence(self, number_str: str, text: str) -> bool:
        """
        Check if a detected number is part of a longer digit sequence.

        Args:
            number_str: The detected number as string
            text: The full text context

        Returns:
            True if the number is part of a longer sequence
        """
        # Look for the number within longer digit sequences
        # Find all positions where this number appears
        for match in re.finditer(re.escape(number_str), text):
            start, end = match.span()

            # Check if there are digits immediately before or after
            has_digit_before = start > 0 and text[start - 1].isdigit()
            has_digit_after = end < len(text) and text[end].isdigit()

            if has_digit_before or has_digit_after:
                return True

        return False

    def _is_valid_jersey_number(self, number: int, original_text: str) -> bool:
        """
        Comprehensive validation for jersey numbers with enhanced >100 protection.

        Args:
            number: Detected number
            original_text: Original OCR text for context

        Returns:
            True if number is valid, False otherwise
        """
        # Primary range validation - STRICT enforcement against >100
        if not (self.valid_number_range[0] <= number <= self.valid_number_range[1]):
            self.validation_stats["invalid_range"] += 1
            # Log specific cases of numbers >100 for analysis
            if number > 100:
                logger.warning(
                    f"Rejected number >100: {number} from text '{original_text}'"
                )
                self.validation_stats["exceeds_100"] += 1
                self._log_invalid_detection(number, original_text, "exceeds_100")
            return False

        # Enhanced validation rules for edge cases
        # Rule 1: Strict validation for numbers close to 100
        if number >= 90:
            # Numbers 90-99 need extra scrutiny as they're close to invalid range
            if self._validate_high_number(number, original_text):
                logger.debug(
                    f"High number {number} validated in context '{original_text}'"
                )
            else:
                logger.debug(
                    f"High number {number} rejected in context '{original_text}'"
                )
                self.validation_stats["high_number_rejected"] += 1
                self._log_invalid_detection(
                    number, original_text, "high_number_suspicious"
                )
                return False

        # Rule 2: Reject numbers that are clearly part of longer sequences in suspicious contexts
        if len(original_text.strip()) > 3 and str(number) in original_text:
            # Check if this number appears as part of a longer sequence
            pattern = rf"\d*{number}\d+"
            if re.search(pattern, original_text.replace(" ", "")):
                # This number might be part of a longer sequence, be more cautious
                if number > 50:  # Higher numbers are more suspicious in long sequences
                    logger.debug(
                        f"Suspicious number {number} in context '{original_text}'"
                    )
                    self.validation_stats["suspicious_sequences"] += 1
                    self._log_invalid_detection(
                        number, original_text, "suspicious_sequence"
                    )
                    return False

        # Rule 3: OCR artifact detection - look for patterns that suggest misreads
        if self._detect_ocr_artifacts(number, original_text):
            logger.debug(
                f"OCR artifact detected for number {number} in '{original_text}'"
            )
            self.validation_stats["ocr_artifacts_detected"] += 1
            self._log_invalid_detection(number, original_text, "ocr_artifact")
            return False

        # Rule 4: Detect potential misreads of 3-digit numbers (like "100" read as "10")
        if self._is_likely_misread_of_invalid_number(number, original_text):
            logger.debug(
                f"Potential misread of invalid number: {number} from '{original_text}'"
            )
            self.validation_stats["ocr_artifacts_detected"] += 1
            self._log_invalid_detection(number, original_text, "misread_invalid")
            return False

        # Rule 5: Common jersey number validation (optional - can be enabled if team rosters available)
        # For now, just ensure it's in the valid range

        return True

    def _validate_high_number(self, number: int, original_text: str) -> bool:
        """
        Enhanced validation for numbers 90-99 to prevent false positives near the 100 boundary.

        Args:
            number: The detected number (90-99)
            original_text: Original OCR text for context analysis

        Returns:
            True if the high number is likely valid, False otherwise
        """
        # Check for patterns that suggest this might be a misread of a 3-digit number
        cleaned_text = original_text.replace(" ", "").replace("-", "")

        # Pattern 1: Look for 3+ digit sequences that contain this number
        three_digit_pattern = rf"\d*{number}\d+"
        if re.search(three_digit_pattern, cleaned_text):
            # This number appears in a longer sequence - be very cautious
            logger.debug(
                f"High number {number} found in longer sequence: '{cleaned_text}'"
            )
            return False

        # Pattern 2: Check for common OCR misreads that could create high numbers
        # e.g., "100" misread as "10O" then extracted as "10"
        if len(cleaned_text) >= 3:
            # Look for patterns like "10O", "1OO", etc.
            suspicious_patterns = [
                r"10[O0]",  # 100 misread with O
                r"1[O0]0",  # 100 with middle O
                r"[O0]0\d",  # Leading O followed by digits
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, cleaned_text, re.IGNORECASE):
                    logger.debug(
                        f"Suspicious OCR pattern detected: '{cleaned_text}' for number {number}"
                    )
                    return False

        # Pattern 3: If the original text is much longer than the number, be cautious
        if len(cleaned_text) > len(str(number)) + 2:
            # Long text with a high number is suspicious
            logger.debug(
                f"Long text '{cleaned_text}' with high number {number} - suspicious"
            )
            return False

        return True

    def _detect_ocr_artifacts(self, number: int, original_text: str) -> bool:
        """
        Detect common OCR artifacts that might lead to invalid number detection.

        Args:
            number: The detected number
            original_text: Original OCR text

        Returns:
            True if OCR artifacts are detected (number should be rejected)
        """
        cleaned_text = original_text.replace(" ", "").upper()

        # Common OCR misreads that could create false numbers
        ocr_artifacts = [
            # Letters that look like numbers
            ("O", "0"),  # O misread as 0
            ("I", "1"),  # I misread as 1
            ("L", "1"),  # L misread as 1
            ("S", "5"),  # S misread as 5
            ("G", "6"),  # G misread as 6
            ("B", "8"),  # B misread as 8
            ("Z", "2"),  # Z misread as 2
        ]

        # Check if the text contains suspicious letter-number combinations
        for letter, digit in ocr_artifacts:
            if letter in cleaned_text and digit in str(number):
                # Found a potential OCR misread
                logger.debug(
                    f"Potential OCR artifact: '{letter}' -> '{digit}' in text '{original_text}'"
                )

                # Be more strict for higher numbers as they're more likely to be artifacts
                if number > 70:
                    return True

        # Check for mixed alphanumeric patterns that suggest OCR confusion
        if re.search(r"[A-Z]\d|\d[A-Z]", cleaned_text):
            logger.debug(f"Mixed alphanumeric pattern detected: '{cleaned_text}'")
            if number > 50:  # Higher numbers with mixed patterns are suspicious
                return True

        return False

    def _clean_ocr_text(self, text: str) -> str:
        """
        Enhanced OCR text cleaning to improve number extraction accuracy.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text with common OCR artifacts corrected
        """
        cleaned = text.upper().strip()

        # Common OCR corrections - fix obvious letter-to-number misreads
        ocr_corrections = {
            "O": "0",  # O -> 0
            "I": "1",  # I -> 1
            "L": "1",  # L -> 1
            "S": "5",  # S -> 5 (sometimes)
            "G": "6",  # G -> 6 (sometimes)
            "B": "8",  # B -> 8 (sometimes)
            "Z": "2",  # Z -> 2 (sometimes)
        }

        # Apply corrections only if the result would be a valid jersey number
        # But be careful not to create false positives from obvious artifacts
        for letter, digit in ocr_corrections.items():
            if letter in cleaned:
                # Check if this looks like an OCR artifact pattern
                if self._is_likely_ocr_artifact_pattern(cleaned, letter, digit):
                    # Don't apply correction for obvious artifacts like "O1", "1O0", etc.
                    continue

                # Try the correction and see if it results in a valid number
                test_text = cleaned.replace(letter, digit)
                # Only apply if the correction creates a cleaner numeric pattern
                if re.search(r"\b\d{1,2}\b", test_text) and not re.search(
                    r"\d{3,}", test_text
                ):
                    cleaned = test_text

        # Remove common punctuation that might interfere
        cleaned = re.sub(r"[.,;:!?]", "", cleaned)

        # Remove extra whitespace and collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Final pass - keep only digits and spaces
        cleaned = re.sub(r"[^\d\s]", "", cleaned)

        # For simple cases with just digits and spaces, remove spaces if it creates a valid 1-2 digit number
        if re.match(r"^\d(\s\d)*$", cleaned):  # Pattern like "3 5" or "1 2 3"
            no_spaces = cleaned.replace(" ", "")
            if len(no_spaces) <= 2 and no_spaces.isdigit():
                cleaned = no_spaces

        return cleaned

    def _is_likely_ocr_artifact_pattern(
        self, text: str, letter: str, digit: str
    ) -> bool:
        """
        Check if a letter-to-digit correction would create an obvious OCR artifact.

        Args:
            text: The text containing the letter
            letter: The letter to be corrected
            digit: The digit it would become

        Returns:
            True if this looks like an OCR artifact that shouldn't be corrected
        """
        # Patterns that are likely OCR artifacts and should not be corrected
        # Be more selective - only prevent corrections for obvious artifacts

        # Special case: Only specific patterns are suspicious artifacts
        # "O1" is suspicious (O followed by 1), but "I5" -> "15" is a valid correction
        if re.match(rf"^{letter}1$", text.strip()) and letter in ["O", "I"]:
            # "O1" or "I1" are suspicious artifacts - don't correct these
            return True

        # Other suspicious patterns
        artifact_patterns = [
            rf"^{letter}{letter}$",  # e.g., "OO", "II" - double letters
            rf"^\d{letter}{letter}$",  # e.g., "1OO", "5II" - digit + double letters
            rf"^{letter}{letter}\d$",  # e.g., "OO1", "II5" - double letters + digit
        ]

        for pattern in artifact_patterns:
            if re.match(pattern, text.strip()):
                return True

        return False

    def _is_likely_misread_of_invalid_number(
        self, number: int, original_text: str
    ) -> bool:
        """
        Detect if a valid number might be a misread of an invalid 3-digit number.

        Args:
            number: The detected valid number
            original_text: Original OCR text

        Returns:
            True if this might be a misread of an invalid number like 100+
        """
        # Look for patterns that suggest the original was a 3-digit number
        cleaned_text = original_text.replace(" ", "").upper()

        # Pattern 1: If we detect "10" but the original text suggests "100"
        if number == 10:
            # Look for patterns that suggest "100" was misread as "10"
            # Include common OCR misreads of "100"
            suspicious_patterns = [
                "100",
                "10O",
                "1OO",
                "1O0",
                "10C",
                "1CC",
                "IOO",
                "I00",
            ]
            if any(pattern in cleaned_text for pattern in suspicious_patterns):
                return True

        # Pattern 2: If we detect a 2-digit number but text suggests 3+ digits
        if 10 <= number <= 99:
            # Check if the original text is longer and might contain 3+ digit patterns
            if len(cleaned_text) >= 3:
                # Look for patterns where our number might be part of a larger invalid number
                number_str = str(number)
                # Check if our number appears at the start of what might be a 3-digit sequence
                if re.search(rf"^{number_str}[O0]", cleaned_text):
                    return True

        return False

    def _is_mixed_alphanumeric_context(self, text: str) -> bool:
        """
        Check if the text contains mixed alphanumeric content that suggests the number
        is not a standalone jersey number.

        Args:
            text: Original OCR text

        Returns:
            True if the text appears to be mixed alphanumeric (should reject numbers from it)
        """
        # Remove spaces and common punctuation for analysis
        cleaned = re.sub(r"[\s.,;:!?-]", "", text.upper())

        # Check for patterns that suggest mixed content
        has_letters = bool(re.search(r"[A-Z]", cleaned))
        has_digits = bool(re.search(r"\d", cleaned))

        if has_letters and has_digits:
            # This is mixed alphanumeric content
            # Check if it's a simple correction case (like "1O" -> "10") vs complex mixed content
            if len(cleaned) > 3:  # Longer mixed content is suspicious
                return True

            # Check for patterns like "12ABC34" where numbers are embedded in letters
            if re.search(r"\d+[A-Z]+\d+", cleaned):
                return True

            # Check for patterns like "ABC12" or "12ABC" where it's clearly not just a jersey number
            if re.search(r"[A-Z]{2,}\d+|\d+[A-Z]{2,}", cleaned):
                return True

        return False

    def _validate_number_context(
        self, number: int, original_text: str, confidence: float
    ) -> bool:
        """
        Additional context-based validation for detected numbers.

        Args:
            number: Detected jersey number
            original_text: Original OCR text
            confidence: OCR confidence score

        Returns:
            True if number passes context validation
        """
        # Apply different confidence thresholds based on number characteristics

        # Very common numbers (1-11) - be more lenient with confidence
        if 1 <= number <= 11:
            # These are very common jersey numbers, use standard threshold
            min_confidence = self.confidence_threshold
        # Higher numbers (80+) need higher confidence as they're less common
        elif number > 80:
            min_confidence = self.confidence_threshold + 0.1
        # Uncommon numbers need even higher confidence
        elif self._is_uncommon_jersey_number(number):
            min_confidence = self.confidence_threshold + 0.15
        else:
            # Standard numbers (12-80) use standard threshold
            min_confidence = self.confidence_threshold

        if confidence < min_confidence:
            logger.debug(
                f"Number {number} rejected due to insufficient confidence: {confidence:.3f} < {min_confidence:.3f}"
            )
            if number > 80:
                self.validation_stats["context_validation_failed"] += 1
            elif self._is_uncommon_jersey_number(number):
                self.validation_stats["uncommon_numbers_rejected"] += 1
            else:
                self.validation_stats["context_validation_failed"] += 1
            return False

        # Check for common jersey number patterns
        if number == 0:
            # Jersey number 0 is very rare in football
            logger.debug(f"Jersey number 0 rejected - very uncommon in football")
            self.validation_stats["context_validation_failed"] += 1
            return False

        return True

    def _is_uncommon_jersey_number(self, number: int) -> bool:
        """
        Check if a jersey number is uncommon in football.

        Args:
            number: Jersey number to check

        Returns:
            True if the number is uncommon
        """
        # Very uncommon numbers in football (but still valid)
        uncommon_numbers = {0, 13, 69, 96, 97, 98, 99}
        return number in uncommon_numbers

    def _log_invalid_detection(self, number: Union[int, str], text: str, reason: str):
        """
        Log invalid detections for debugging and analysis.

        Args:
            number: Invalid number detected
            text: Original OCR text
            reason: Reason for rejection
        """
        self.invalid_detections[reason].append(
            {"number": number, "text": text, "timestamp": self.frame_count}
        )

        # Keep only recent invalid detections to prevent memory bloat
        if len(self.invalid_detections[reason]) > 100:
            self.invalid_detections[reason] = self.invalid_detections[reason][-50:]

        logger.debug(f"Invalid detection: {number} in '{text}' - {reason}")

    def detect_jersey_number(
        self, frame: np.ndarray, bbox: List[float], track_id: int
    ) -> Optional[int]:
        """
        Detect jersey number for a player using multi-frame consensus.

        Args:
            frame: Input video frame
            bbox: Player bounding box
            track_id: Player tracking ID

        Returns:
            Detected jersey number or None
        """
        # Check cache first - this is the most important optimization
        if track_id in self.player_jersey_cache:
            self.cache_hits += 1
            return self.player_jersey_cache[track_id]

        # Quick check: if we already have enough history for this player,
        # try to get consensus without new OCR
        if (
            track_id in self.player_jersey_history
            and len(self.player_jersey_history[track_id]) >= self.consensus_frames
        ):
            consensus_number = self._get_consensus_number(track_id)
            if consensus_number is not None:
                self.player_jersey_cache[track_id] = consensus_number
                logger.info(
                    f"Jersey number {consensus_number} confirmed for player {track_id} (from history)"
                )
                return consensus_number

        # Ultra-aggressive optimization: if we have ANY valid detection for this player
        # and we've already confirmed many other players, use the first valid detection
        if (
            track_id in self.player_jersey_history
            and len(self.player_jersey_history[track_id]) > 0
            and len(self.player_jersey_cache) > 10
        ):  # If we already have 10+ confirmed players

            # Use the most confident detection we have
            best_detection = max(
                self.player_jersey_history[track_id], key=lambda x: x[1]
            )
            number, confidence = best_detection

            if confidence > 0.5:  # Reasonable confidence threshold
                self.player_jersey_cache[track_id] = number
                logger.info(
                    f"Jersey number {number} fast-confirmed for player {track_id} (confidence: {confidence:.3f})"
                )
                return number

        # Preprocess jersey region
        preprocessed = self.preprocess_jersey_region(frame, bbox)
        if preprocessed is None:
            return None

        # Extract jersey number
        number, confidence = self.extract_jersey_number(preprocessed)

        if number is not None:
            # Additional post-processing validation before adding to history
            if self._post_process_validate(number, track_id):
                # Add to history
                self.player_jersey_history[track_id].append((number, confidence))

                # Keep only recent detections
                if (
                    len(self.player_jersey_history[track_id])
                    > self.consensus_frames * 2
                ):
                    self.player_jersey_history[track_id] = self.player_jersey_history[
                        track_id
                    ][-self.consensus_frames :]

                # Check for consensus
                consensus_number = self._get_consensus_number(track_id)
                if consensus_number is not None:
                    self.player_jersey_cache[track_id] = consensus_number
                    logger.info(
                        f"Jersey number {consensus_number} confirmed for player {track_id}"
                    )
                    return consensus_number
            else:
                logger.debug(
                    f"Number {number} failed post-processing validation for player {track_id}"
                )

        return None

    def _post_process_validate(self, number: int, track_id: int) -> bool:
        """
        Additional validation after OCR processing and before consensus.

        Args:
            number: Detected jersey number
            track_id: Player tracking ID

        Returns:
            True if number passes validation, False otherwise
        """
        # Ensure number is still in valid range (double-check)
        if not (self.valid_number_range[0] <= number <= self.valid_number_range[1]):
            self._log_invalid_detection(
                number, f"track_{track_id}", "post_process_range"
            )
            return False

        # Check for consistency with previous detections for this player
        if track_id in self.player_jersey_history:
            history = self.player_jersey_history[track_id]
            if history:
                # Get the most common number so far
                numbers = [n for n, _ in history]
                if numbers:
                    most_common = max(set(numbers), key=numbers.count)
                    # If this number is very different from the most common, be more cautious
                    if abs(number - most_common) > 20:  # Large difference
                        logger.debug(
                            f"Large difference detected: {number} vs {most_common} for player {track_id}"
                        )
                        # Still allow it, but log for analysis

        return True

    def _get_consensus_number(self, track_id: int) -> Optional[int]:
        """
        Get consensus jersey number from detection history with enhanced validation.

        Args:
            track_id: Player tracking ID

        Returns:
            Consensus jersey number or None
        """
        history = self.player_jersey_history[track_id]

        if len(history) < 3:  # Need at least 3 detections
            return None

        # Count occurrences of each number
        number_counts: Counter[int] = Counter()
        total_confidence: defaultdict[int, float] = defaultdict(float)

        for number, confidence in history[-self.consensus_frames :]:
            # Additional validation - ensure all numbers in consensus are valid
            if self.valid_number_range[0] <= number <= self.valid_number_range[1]:
                number_counts[number] += 1
                total_confidence[number] += confidence

        # Find most frequent number
        if not number_counts:
            return None

        most_common = number_counts.most_common(1)[0]
        number, count = most_common

        # Enhanced consensus requirements
        avg_confidence = total_confidence[number] / count

        # Stricter requirements for higher numbers (more likely to be false positives)
        min_detections = 2
        min_confidence = self.confidence_threshold

        if number > 50:  # Higher numbers need more evidence
            min_detections = 3
            min_confidence = self.confidence_threshold + 0.1

        if count >= min_detections and avg_confidence > min_confidence:
            # Final validation - ensure this number hasn't been rejected too many times
            if not self._is_frequently_rejected(number, track_id):
                return number

        return None

    def _is_frequently_rejected(self, number: int, track_id: int) -> bool:
        """
        Check if a number has been frequently rejected for this player.

        Args:
            number: Jersey number to check
            track_id: Player tracking ID

        Returns:
            True if frequently rejected, False otherwise
        """
        # Check invalid detections log
        rejection_count = 0
        for reason, detections in self.invalid_detections.items():
            for detection in detections:
                if detection.get("number") == number:
                    rejection_count += 1

        # If rejected more than 3 times, be very cautious
        return rejection_count > 3

    def get_jersey_number_for_player(self, track_id: int) -> Optional[int]:
        """
        Get the confirmed jersey number for a player.

        Args:
            track_id: Player tracking ID

        Returns:
            Jersey number or None if not detected
        """
        return self.player_jersey_cache.get(track_id)

    def get_detection_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive performance statistics for the detector.

        Returns:
            Dictionary with performance metrics and validation statistics
        """
        total_requests = self.ocr_calls + self.cache_hits
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        # Calculate validation rates
        total_ocr = self.validation_stats["total_ocr_results"]
        valid_rate = (
            (self.validation_stats["valid_numbers"] / total_ocr)
            if total_ocr > 0
            else 0.0
        )

        return {
            "ocr_calls": self.ocr_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "confirmed_players": len(self.player_jersey_cache),
            "players_in_progress": len(self.player_jersey_history)
            - len(self.player_jersey_cache),
            "validation_stats": self.validation_stats.copy(),
            "valid_detection_rate": valid_rate,
            "total_invalid_detections": sum(
                len(detections) for detections in self.invalid_detections.values()
            ),
        }

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get detailed validation report for debugging.

        Returns:
            Dictionary with detailed validation information
        """
        return {
            "validation_stats": self.validation_stats.copy(),
            "invalid_detections_summary": {
                reason: len(detections)
                for reason, detections in self.invalid_detections.items()
            },
            "recent_invalid_detections": {
                reason: detections[-5:] if detections else []
                for reason, detections in self.invalid_detections.items()
            },
        }

    def reset_cache(self):
        """Reset all cached data and statistics (useful for new videos)."""
        self.player_jersey_history.clear()
        self.player_jersey_cache.clear()
        self.invalid_detections.clear()
        self.frame_count = 0
        self.ocr_calls = 0
        self.cache_hits = 0

        # Reset validation statistics
        self.validation_stats = {
            "total_ocr_results": 0,
            "valid_numbers": 0,
            "invalid_range": 0,
            "invalid_format": 0,
            "filtered_multi_digit": 0,
            "exceeds_100": 0,  # Track numbers >100 specifically
            "high_number_rejected": 0,  # Track 90-99 numbers rejected
            "ocr_artifacts_detected": 0,  # Track OCR artifact rejections
            "suspicious_sequences": 0,  # Track suspicious sequence rejections
            "context_validation_failed": 0,  # Track context validation failures
            "uncommon_numbers_rejected": 0,  # Track uncommon numbers rejected
            "mixed_alphanumeric_rejected": 0,  # Track mixed alphanumeric rejections
        }

        logger.info("Jersey number detector cache and statistics reset")
