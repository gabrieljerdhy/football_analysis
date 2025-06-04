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

        # Enhanced preprocessing pipeline
        # 1. Gaussian blur to reduce noise before enhancement
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 2. Adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # 3. Morphological operations to clean up text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 4. Bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(morphed, 7, 50, 50)

        # 5. Unsharp masking for better text clarity
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

        # 6. Final contrast adjustment
        sharpened = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)

        return sharpened

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

        # Clean the text - remove common OCR artifacts
        cleaned_text = re.sub(r"[^\d\s]", "", text)  # Keep only digits and spaces

        # Multiple regex patterns to catch different number formats
        patterns = [
            r"\b(\d{1,2})\b",  # Standalone 1-2 digit numbers
            r"^(\d{1,2})$",  # Entire string is 1-2 digits
            r"(\d{1,2})(?=\s|$)",  # 1-2 digits followed by space or end
        ]

        found_numbers = set()  # Use set to avoid duplicates

        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            for match in matches:
                found_numbers.add(match)

        # Also try to extract from longer digit sequences
        long_sequences = re.findall(r"\d{3,}", cleaned_text)
        for seq in long_sequences:
            # Try to extract valid 1-2 digit numbers from longer sequences
            extracted = self._extract_from_long_sequence(seq)
            found_numbers.update(extracted)
            if extracted:
                self.validation_stats["filtered_multi_digit"] += 1

        # Validate each found number
        for num_str in found_numbers:
            try:
                number = int(num_str)

                # Strict validation
                if self._is_valid_jersey_number(number, text):
                    valid_numbers.append((number, confidence))
                else:
                    # Log invalid detections for debugging
                    self._log_invalid_detection(number, text, "range_validation")

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

    def _is_valid_jersey_number(self, number: int, original_text: str) -> bool:
        """
        Comprehensive validation for jersey numbers.

        Args:
            number: Detected number
            original_text: Original OCR text for context

        Returns:
            True if number is valid, False otherwise
        """
        # Range validation
        if not (self.valid_number_range[0] <= number <= self.valid_number_range[1]):
            self.validation_stats["invalid_range"] += 1
            return False

        # Additional validation rules
        # Rule 1: Reject numbers that are clearly part of longer sequences in suspicious contexts
        if len(original_text.strip()) > 3 and str(number) in original_text:
            # Check if this number appears as part of a longer sequence
            pattern = rf"\d*{number}\d+"
            if re.search(pattern, original_text.replace(" ", "")):
                # This number might be part of a longer sequence, be more cautious
                if number > 50:  # Higher numbers are more suspicious in long sequences
                    logger.debug(
                        f"Suspicious number {number} in context '{original_text}'"
                    )
                    return False

        # Rule 2: Common jersey number validation (optional - can be enabled if team rosters available)
        # For now, just ensure it's in the valid range

        return True

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
        # Check cache first
        if track_id in self.player_jersey_cache:
            self.cache_hits += 1
            return self.player_jersey_cache[track_id]

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
        }

        logger.info("Jersey number detector cache and statistics reset")
