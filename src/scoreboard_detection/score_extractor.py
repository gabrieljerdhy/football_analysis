"""
Score Extractor

This module implements OCR-based score extraction from detected scoreboard regions.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. Score extraction will be limited.")


class ScoreExtractor:
    """
    Extracts numerical scores from scoreboard regions using OCR and pattern matching.

    The extractor uses:
    1. Image preprocessing for better OCR results
    2. Tesseract OCR for text extraction
    3. Pattern matching for score formats (e.g., "2-1", "2:1", "2 1")
    4. Confidence scoring for extracted scores
    """

    def __init__(
        self,
        tesseract_config: str = "--psm 8 -c tessedit_char_whitelist=0123456789:-",
        min_score_confidence: float = 0.4,
    ):  # Lowered threshold
        """
        Initialize the score extractor.

        Args:
            tesseract_config: Tesseract configuration string
            min_score_confidence: Minimum confidence for score extraction
        """
        # Multiple Tesseract configurations for different scenarios
        self.tesseract_configs = [
            "--psm 8 -c tessedit_char_whitelist=0123456789:-",  # Single text line, digits only
            "--psm 7 -c tesseract_char_whitelist=0123456789:-",  # Single text block
            "--psm 6 -c tesseract_char_whitelist=0123456789:-",  # Single uniform block
            "--psm 13 -c tesseract_char_whitelist=0123456789:-",  # Raw line, no heuristics
            "--psm 8",  # Single text line, all characters
            "--psm 7",  # Single text block, all characters
        ]
        self.min_confidence = min_score_confidence

        # Enhanced score patterns with more variations
        self.score_patterns = [
            r"(\d+)\s*[-:–—]\s*(\d+)",  # "2-1", "2:1", "2 - 1" with various dashes
            r"(\d+)\s+(\d+)",  # "2 1"
            r"(\d+)\s*[^\d\s]\s*(\d+)",  # Any separator between digits
            r"(\d+).*?(\d+)",  # Any two numbers (last resort)
        ]

        self.logger = logging.getLogger(__name__)

        if not TESSERACT_AVAILABLE:
            self.logger.warning(
                "Tesseract not available. Using fallback pattern matching only."
            )

    def extract_scores(self, frame: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        Extract scores from detected scoreboard regions.

        Args:
            frame: Input video frame
            regions: List of detected scoreboard regions

        Returns:
            List of extracted scores with confidence and metadata
        """
        extracted_scores = []

        for region in regions:
            bbox = region["bbox"]
            x, y, w, h = bbox

            # Extract region from frame
            scoreboard_region = frame[y : y + h, x : x + w]

            if scoreboard_region.size == 0:
                continue

            # Preprocess the region (get multiple variants)
            processed_variants = self._preprocess_for_ocr(scoreboard_region)

            # Extract text using OCR on all variants
            all_extracted_texts = []
            for processed_region in processed_variants:
                extracted_text = self._extract_text_ocr(processed_region)
                if extracted_text:
                    all_extracted_texts.append(extracted_text)

            # Extract scores using pattern matching on all texts
            scores = []
            for text in all_extracted_texts:
                text_scores = self._extract_scores_from_text(text)
                scores.extend(text_scores)

            # Also try direct pattern matching on preprocessed images
            fallback_scores = []
            for processed_region in processed_variants:
                fallback = self._extract_scores_fallback(processed_region)
                fallback_scores.extend(fallback)

            # Combine and validate scores
            all_scores = scores + fallback_scores
            validated_scores = self._validate_and_score_extractions(all_scores, region)

            extracted_scores.extend(validated_scores)

        # Remove duplicates and sort by confidence
        unique_scores = self._deduplicate_scores(extracted_scores)

        return sorted(unique_scores, key=lambda x: x["confidence"], reverse=True)

    def _preprocess_for_ocr(self, region: np.ndarray) -> List[np.ndarray]:
        """
        Preprocess scoreboard region for better OCR results.

        Args:
            region: Scoreboard region image

        Returns:
            List of preprocessed image variants
        """
        if region.size == 0:
            return [region]

        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)

        # Resize for better OCR (make it larger)
        height, width = gray.shape
        scale_factor = max(
            3, 150 // min(height, width)
        )  # Larger scaling for better OCR
        new_width = width * scale_factor
        new_height = height * scale_factor

        preprocessed_variants = []

        # Process both original and equalized images
        for img in [gray, equalized]:
            resized = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

            # 1. OTSU threshold
            _, thresh1 = cv2.threshold(
                resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_variants.append(thresh1)

            # 2. Inverted OTSU threshold
            _, thresh2 = cv2.threshold(
                resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            preprocessed_variants.append(thresh2)

            # 3. Adaptive threshold (Gaussian)
            try:
                adaptive_gauss = cv2.adaptiveThreshold(
                    resized,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
                preprocessed_variants.append(adaptive_gauss)
            except:
                pass

            # 4. Adaptive threshold (Mean)
            try:
                adaptive_mean = cv2.adaptiveThreshold(
                    resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
                )
                preprocessed_variants.append(adaptive_mean)
            except:
                pass

            # 5. Manual threshold with different values
            for thresh_val in [100, 127, 150, 180]:
                _, manual_thresh = cv2.threshold(
                    resized, thresh_val, 255, cv2.THRESH_BINARY
                )
                preprocessed_variants.append(manual_thresh)

            # 6. Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph_close = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            morph_open = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
            preprocessed_variants.extend([morph_close, morph_open])

        return preprocessed_variants

    def _extract_text_ocr(self, processed_region: np.ndarray) -> str:
        """Extract text using Tesseract OCR with multiple configurations."""
        if not TESSERACT_AVAILABLE or processed_region.size == 0:
            return ""

        best_text = ""
        best_confidence = 0

        # Try multiple Tesseract configurations
        for config in self.tesseract_configs:
            try:
                # Extract text with current configuration
                text = pytesseract.image_to_string(processed_region, config=config)
                text = text.strip()

                if text:
                    # Try to get confidence data
                    try:
                        data = pytesseract.image_to_data(
                            processed_region,
                            config=config,
                            output_type=pytesseract.Output.DICT,
                        )
                        confidences = [
                            int(conf) for conf in data["conf"] if int(conf) > 0
                        ]
                        avg_confidence = (
                            sum(confidences) / len(confidences) if confidences else 0
                        )

                        # Keep the text with highest confidence
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_text = text
                    except:
                        # If confidence extraction fails, use the text if we don't have a better one
                        if not best_text:
                            best_text = text

            except Exception as e:
                self.logger.debug(f"OCR extraction failed with config {config}: {e}")
                continue

        return best_text

    def _extract_scores_from_text(self, text: str) -> List[Dict]:
        """Extract scores from OCR text using pattern matching."""
        scores = []

        if not text:
            return scores

        for pattern in self.score_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    team1_score = int(match.group(1))
                    team2_score = int(match.group(2))

                    # Enhanced score validation
                    if self._is_valid_football_score(
                        team1_score, team2_score, match.group(0)
                    ):
                        scores.append(
                            {
                                "team1_score": team1_score,
                                "team2_score": team2_score,
                                "raw_text": match.group(0),
                                "extraction_method": "ocr_pattern",
                                "pattern": pattern,
                            }
                        )
                except (ValueError, IndexError):
                    continue

        return scores

    def _extract_scores_fallback(self, processed_region: np.ndarray) -> List[Dict]:
        """
        Fallback score extraction using template matching for digits.
        This is a simplified approach when OCR is not available.
        """
        scores = []

        # This is a placeholder for template matching approach
        # In a full implementation, we would:
        # 1. Have templates for digits 0-9
        # 2. Use template matching to find digit locations
        # 3. Reconstruct scores from detected digits

        # For now, return empty list
        return scores

    def _validate_and_score_extractions(
        self, extractions: List[Dict], region: Dict
    ) -> List[Dict]:
        """Validate extracted scores and assign confidence scores."""
        validated_scores = []

        for extraction in extractions:
            confidence = self._calculate_extraction_confidence(extraction, region)

            if confidence >= self.min_confidence:
                extraction["confidence"] = confidence
                extraction["region_bbox"] = region["bbox"]
                extraction["region_confidence"] = region["confidence"]
                validated_scores.append(extraction)

        return validated_scores

    def _calculate_extraction_confidence(self, extraction: Dict, region: Dict) -> float:
        """Calculate confidence score for an extracted score."""
        confidence = 0.0

        # Base confidence from region detection
        confidence += region["confidence"] * 0.3

        # Confidence based on extraction method
        if extraction["extraction_method"] == "ocr_pattern":
            confidence += 0.4
        elif extraction["extraction_method"] == "template_matching":
            confidence += 0.3

        # Confidence based on score reasonableness
        team1_score = extraction["team1_score"]
        team2_score = extraction["team2_score"]

        # Reasonable score range
        if 0 <= team1_score <= 10 and 0 <= team2_score <= 10:
            confidence += 0.2
        elif 0 <= team1_score <= 20 and 0 <= team2_score <= 20:
            confidence += 0.1

        # Prefer lower scores (more common in football)
        if team1_score + team2_score <= 5:
            confidence += 0.1

        return min(confidence, 1.0)

    def _deduplicate_scores(self, scores: List[Dict]) -> List[Dict]:
        """Remove duplicate score extractions."""
        if not scores:
            return scores

        unique_scores = []
        seen_scores = set()

        for score in scores:
            score_tuple = (score["team1_score"], score["team2_score"])
            if score_tuple not in seen_scores:
                seen_scores.add(score_tuple)
                unique_scores.append(score)

        return unique_scores

    def _is_valid_football_score(
        self, team1_score: int, team2_score: int, raw_text: str
    ) -> bool:
        """
        Enhanced validation for football scores.

        Args:
            team1_score: First team's score
            team2_score: Second team's score
            raw_text: Original text that was matched

        Returns:
            True if the score is valid for football
        """
        # Basic range check (football scores are typically 0-10, rarely higher)
        if not (0 <= team1_score <= 15 and 0 <= team2_score <= 15):
            return False

        # Check for common OCR errors that produce unrealistic scores
        # 9-9 is very unlikely in football, might be misread 3-3
        if team1_score == 9 and team2_score == 9:
            self.logger.debug(
                f"Suspicious score 9-9 detected, might be misread 3-3: {raw_text}"
            )
            return False

        # Very high scores are suspicious (>8 goals per team is extremely rare)
        if team1_score > 8 or team2_score > 8:
            self.logger.debug(
                f"Unusually high score detected: {team1_score}-{team2_score}"
            )
            return False

        # Check for identical high scores (often OCR errors)
        if team1_score == team2_score and team1_score > 5:
            self.logger.debug(
                f"Suspicious identical high score: {team1_score}-{team2_score}"
            )
            return False

        # Check raw text for suspicious patterns
        if raw_text:
            # Look for non-score text that might have been misinterpreted
            suspicious_patterns = [
                r"\d{2}:\d{2}",  # Time format (e.g., "90:00")
                r"\d{4}",  # Year or other 4-digit numbers
                r"\d+\.\d+",  # Decimal numbers
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, raw_text):
                    self.logger.debug(f"Suspicious text pattern in score: {raw_text}")
                    return False

        return True

    def get_best_score(self, extracted_scores: List[Dict]) -> Optional[Dict]:
        """Get the most confident score extraction."""
        if not extracted_scores:
            return None

        # Sort by confidence and return the best
        best_scores = sorted(
            extracted_scores, key=lambda x: x["confidence"], reverse=True
        )
        return best_scores[0] if best_scores else None

    def format_score_for_display(self, score: Dict) -> str:
        """Format extracted score for display."""
        if not score:
            return "No score detected"

        team1 = score["team1_score"]
        team2 = score["team2_score"]
        confidence = score["confidence"]

        return f"{team1}-{team2} (confidence: {confidence:.2f})"

    def _is_valid_football_score(
        self, team1_score: int, team2_score: int, raw_text: str
    ) -> bool:
        """
        Enhanced validation for football scores.

        Args:
            team1_score: First team's score
            team2_score: Second team's score
            raw_text: Original text that was matched

        Returns:
            True if the score is valid for football
        """
        # Basic range check (football scores are typically 0-10, rarely higher)
        if not (0 <= team1_score <= 15 and 0 <= team2_score <= 15):
            return False

        # Check for common OCR errors that produce unrealistic scores
        # 9-9 is very unlikely in football, might be misread 3-3
        if team1_score == 9 and team2_score == 9:
            self.logger.debug(
                f"Suspicious score 9-9 detected, might be misread 3-3: {raw_text}"
            )
            return False

        # Very high scores are suspicious (>8 goals per team is extremely rare)
        if team1_score > 8 or team2_score > 8:
            self.logger.debug(
                f"Unusually high score detected: {team1_score}-{team2_score}"
            )
            return False

        # Check for identical high scores (often OCR errors)
        if team1_score == team2_score and team1_score > 5:
            self.logger.debug(
                f"Suspicious identical high score: {team1_score}-{team2_score}"
            )
            return False

        # Check raw text for suspicious patterns
        if raw_text:
            # Look for non-score text that might have been misinterpreted
            suspicious_patterns = [
                r"\d{2}:\d{2}",  # Time format (e.g., "90:00")
                r"\d{4}",  # Year or other 4-digit numbers
                r"\d+\.\d+",  # Decimal numbers
            ]

            for pattern in suspicious_patterns:
                if re.search(pattern, raw_text):
                    self.logger.debug(f"Suspicious text pattern in score: {raw_text}")
                    return False

        return True
