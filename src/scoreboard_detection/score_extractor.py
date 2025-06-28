"""
Score Extractor

This module implements OCR-based score extraction from detected scoreboard regions.
"""

import cv2
import numpy as np
import re
import logging
from typing import List, Tuple, Optional, Dict, Union
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
    
    def __init__(self, 
                 tesseract_config: str = '--psm 8 -c tessedit_char_whitelist=0123456789:-',
                 min_score_confidence: float = 0.6):
        """
        Initialize the score extractor.
        
        Args:
            tesseract_config: Tesseract configuration string
            min_score_confidence: Minimum confidence for score extraction
        """
        self.tesseract_config = tesseract_config
        self.min_confidence = min_score_confidence
        
        # Common score patterns
        self.score_patterns = [
            r'(\d+)\s*[-:]\s*(\d+)',  # "2-1", "2:1", "2 - 1"
            r'(\d+)\s+(\d+)',         # "2 1"
            r'(\d+).*?(\d+)',         # Any two numbers
        ]
        
        self.logger = logging.getLogger(__name__)
        
        if not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseract not available. Using fallback pattern matching only.")
    
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
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # Extract region from frame
            scoreboard_region = frame[y:y+h, x:x+w]
            
            if scoreboard_region.size == 0:
                continue
            
            # Preprocess the region
            processed_region = self._preprocess_for_ocr(scoreboard_region)
            
            # Extract text using OCR
            extracted_text = self._extract_text_ocr(processed_region)
            
            # Extract scores using pattern matching
            scores = self._extract_scores_from_text(extracted_text)
            
            # Also try direct pattern matching on preprocessed image
            fallback_scores = self._extract_scores_fallback(processed_region)
            
            # Combine and validate scores
            all_scores = scores + fallback_scores
            validated_scores = self._validate_and_score_extractions(all_scores, region)
            
            extracted_scores.extend(validated_scores)
        
        # Remove duplicates and sort by confidence
        unique_scores = self._deduplicate_scores(extracted_scores)
        
        return sorted(unique_scores, key=lambda x: x['confidence'], reverse=True)
    
    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess scoreboard region for better OCR results.
        
        Args:
            region: Scoreboard region image
            
        Returns:
            Preprocessed image
        """
        if region.size == 0:
            return region
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Resize for better OCR (make it larger)
        height, width = gray.shape
        scale_factor = max(2, 100 // min(height, width))  # Ensure minimum size
        new_width = width * scale_factor
        new_height = height * scale_factor
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply different preprocessing techniques and choose the best
        preprocessed_variants = []
        
        # 1. Simple threshold
        _, thresh1 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_variants.append(thresh1)
        
        # 2. Inverted threshold
        _, thresh2 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_variants.append(thresh2)
        
        # 3. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed_variants.append(adaptive)
        
        # 4. Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        preprocessed_variants.append(morph)
        
        # For now, return the OTSU threshold version
        # In a more sophisticated implementation, we could test all variants
        return thresh1
    
    def _extract_text_ocr(self, processed_region: np.ndarray) -> str:
        """Extract text using Tesseract OCR."""
        if not TESSERACT_AVAILABLE or processed_region.size == 0:
            return ""
        
        try:
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(processed_region, config=self.tesseract_config)
            return text.strip()
        except Exception as e:
            self.logger.debug(f"OCR extraction failed: {e}")
            return ""
    
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
                    
                    # Validate score range (0-20 is reasonable for football)
                    if 0 <= team1_score <= 20 and 0 <= team2_score <= 20:
                        scores.append({
                            'team1_score': team1_score,
                            'team2_score': team2_score,
                            'raw_text': match.group(0),
                            'extraction_method': 'ocr_pattern',
                            'pattern': pattern
                        })
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
    
    def _validate_and_score_extractions(self, extractions: List[Dict], region: Dict) -> List[Dict]:
        """Validate extracted scores and assign confidence scores."""
        validated_scores = []
        
        for extraction in extractions:
            confidence = self._calculate_extraction_confidence(extraction, region)
            
            if confidence >= self.min_confidence:
                extraction['confidence'] = confidence
                extraction['region_bbox'] = region['bbox']
                extraction['region_confidence'] = region['confidence']
                validated_scores.append(extraction)
        
        return validated_scores
    
    def _calculate_extraction_confidence(self, extraction: Dict, region: Dict) -> float:
        """Calculate confidence score for an extracted score."""
        confidence = 0.0
        
        # Base confidence from region detection
        confidence += region['confidence'] * 0.3
        
        # Confidence based on extraction method
        if extraction['extraction_method'] == 'ocr_pattern':
            confidence += 0.4
        elif extraction['extraction_method'] == 'template_matching':
            confidence += 0.3
        
        # Confidence based on score reasonableness
        team1_score = extraction['team1_score']
        team2_score = extraction['team2_score']
        
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
            score_tuple = (score['team1_score'], score['team2_score'])
            if score_tuple not in seen_scores:
                seen_scores.add(score_tuple)
                unique_scores.append(score)
        
        return unique_scores
    
    def get_best_score(self, extracted_scores: List[Dict]) -> Optional[Dict]:
        """Get the most confident score extraction."""
        if not extracted_scores:
            return None
        
        # Sort by confidence and return the best
        best_scores = sorted(extracted_scores, key=lambda x: x['confidence'], reverse=True)
        return best_scores[0] if best_scores else None
    
    def format_score_for_display(self, score: Dict) -> str:
        """Format extracted score for display."""
        if not score:
            return "No score detected"
        
        team1 = score['team1_score']
        team2 = score['team2_score']
        confidence = score['confidence']
        
        return f"{team1}-{team2} (confidence: {confidence:.2f})"
