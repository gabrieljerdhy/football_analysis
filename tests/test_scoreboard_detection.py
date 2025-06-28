"""
Unit tests for scoreboard detection functionality.

This module tests the scoreboard detection, score extraction, and analysis components.
"""

import unittest
import numpy as np
import cv2
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoreboard_detection import ScoreboardDetector, ScoreExtractor, ScoreboardAnalyzer


class TestScoreboardDetector(unittest.TestCase):
    """Test cases for ScoreboardDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ScoreboardDetector(
            min_scoreboard_width=100,
            min_scoreboard_height=30,
            max_scoreboard_width=400,
            max_scoreboard_height=100,
            confidence_threshold=0.5
        )
    
    def test_detector_initialization(self):
        """Test that detector initializes with correct parameters."""
        self.assertEqual(self.detector.min_width, 100)
        self.assertEqual(self.detector.min_height, 30)
        self.assertEqual(self.detector.max_width, 400)
        self.assertEqual(self.detector.max_height, 100)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    def test_detect_scoreboard_regions_empty_frame(self):
        """Test detection with empty frame."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = self.detector.detect_scoreboard_regions(empty_frame)
        self.assertIsInstance(regions, list)
    
    def test_detect_scoreboard_regions_none_frame(self):
        """Test detection with None frame."""
        regions = self.detector.detect_scoreboard_regions(None)
        self.assertEqual(regions, [])
    
    def test_calculate_text_confidence(self):
        """Test text confidence calculation."""
        # Create a simple text-like region
        text_region = np.ones((50, 200), dtype=np.uint8) * 128
        # Add some horizontal lines to simulate text
        text_region[10:15, :] = 255
        text_region[25:30, :] = 255
        text_region[40:45, :] = 255
        
        confidence = self.detector._calculate_text_confidence(text_region)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_rectangle_confidence(self):
        """Test rectangle confidence calculation."""
        # Create a simple rectangular contour
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]], dtype=np.int32)
        confidence = self.detector._calculate_rectangle_confidence(contour, (100, 50))
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_overlap(self):
        """Test bounding box overlap calculation."""
        bbox1 = (10, 10, 50, 30)  # x, y, w, h
        bbox2 = (30, 20, 50, 30)  # Overlapping box
        bbox3 = (100, 100, 50, 30)  # Non-overlapping box
        
        overlap1 = self.detector._calculate_overlap(bbox1, bbox2)
        overlap2 = self.detector._calculate_overlap(bbox1, bbox3)
        
        self.assertGreater(overlap1, 0.0)
        self.assertEqual(overlap2, 0.0)


class TestScoreExtractor(unittest.TestCase):
    """Test cases for ScoreExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ScoreExtractor(min_score_confidence=0.5)
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        self.assertEqual(self.extractor.min_confidence, 0.5)
        self.assertIsInstance(self.extractor.score_patterns, list)
        self.assertGreater(len(self.extractor.score_patterns), 0)
    
    def test_extract_scores_from_text(self):
        """Test score extraction from text."""
        test_cases = [
            ("Team A 2-1 Team B", [(2, 1)]),
            ("Score: 3:0", [(3, 0)]),
            ("Final 1 - 2", [(1, 2)]),
            ("4 5", [(4, 5)]),
            ("No scores here", []),
            ("25-30", []),  # Should be filtered out as unrealistic
        ]
        
        for text, expected_scores in test_cases:
            with self.subTest(text=text):
                scores = self.extractor._extract_scores_from_text(text)
                extracted_tuples = [(s['team1_score'], s['team2_score']) for s in scores]
                self.assertEqual(extracted_tuples, expected_scores)
    
    def test_validate_and_score_extractions(self):
        """Test extraction validation and scoring."""
        mock_region = {'bbox': (10, 10, 100, 50), 'confidence': 0.8}
        
        extractions = [
            {'team1_score': 2, 'team2_score': 1, 'extraction_method': 'ocr_pattern'},
            {'team1_score': 25, 'team2_score': 30, 'extraction_method': 'ocr_pattern'},  # Unrealistic
        ]
        
        validated = self.extractor._validate_and_score_extractions(extractions, mock_region)
        
        # Should have at least one valid extraction
        self.assertGreater(len(validated), 0)
        
        # All validated extractions should have confidence scores
        for extraction in validated:
            self.assertIn('confidence', extraction)
            self.assertGreaterEqual(extraction['confidence'], 0.0)
            self.assertLessEqual(extraction['confidence'], 1.0)
    
    def test_deduplicate_scores(self):
        """Test score deduplication."""
        scores = [
            {'team1_score': 2, 'team2_score': 1, 'confidence': 0.8},
            {'team1_score': 2, 'team2_score': 1, 'confidence': 0.7},  # Duplicate
            {'team1_score': 1, 'team2_score': 2, 'confidence': 0.9},
        ]
        
        unique_scores = self.extractor._deduplicate_scores(scores)
        
        self.assertEqual(len(unique_scores), 2)
        score_tuples = {(s['team1_score'], s['team2_score']) for s in unique_scores}
        self.assertEqual(score_tuples, {(2, 1), (1, 2)})
    
    def test_get_best_score(self):
        """Test getting the best score from extractions."""
        scores = [
            {'team1_score': 2, 'team2_score': 1, 'confidence': 0.7},
            {'team1_score': 1, 'team2_score': 2, 'confidence': 0.9},
            {'team1_score': 0, 'team2_score': 0, 'confidence': 0.5},
        ]
        
        best_score = self.extractor.get_best_score(scores)
        
        self.assertIsNotNone(best_score)
        self.assertEqual(best_score['confidence'], 0.9)
        self.assertEqual(best_score['team1_score'], 1)
        self.assertEqual(best_score['team2_score'], 2)
    
    def test_get_best_score_empty_list(self):
        """Test getting best score from empty list."""
        best_score = self.extractor.get_best_score([])
        self.assertIsNone(best_score)


class TestScoreboardAnalyzer(unittest.TestCase):
    """Test cases for ScoreboardAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ScoreboardAnalyzer(
            detection_interval=10,
            min_detection_confidence=0.6,
            min_extraction_confidence=0.6,
            score_stability_frames=3
        )
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        self.assertEqual(self.analyzer.detection_interval, 10)
        self.assertEqual(self.analyzer.min_detection_confidence, 0.6)
        self.assertEqual(self.analyzer.min_extraction_confidence, 0.6)
        self.assertEqual(self.analyzer.score_stability_frames, 3)
        self.assertIsNotNone(self.analyzer.detector)
        self.assertIsNotNone(self.analyzer.extractor)
    
    @patch('src.scoreboard_detection.scoreboard_detector.ScoreboardDetector.detect_scoreboard_regions')
    @patch('src.scoreboard_detection.score_extractor.ScoreExtractor.extract_scores')
    def test_analyze_frame(self, mock_extract_scores, mock_detect_regions):
        """Test frame analysis."""
        # Mock detection results
        mock_detect_regions.return_value = [
            {'bbox': (10, 10, 100, 50), 'confidence': 0.8, 'type': 'text'}
        ]
        
        # Mock extraction results
        mock_extract_scores.return_value = [
            {
                'team1_score': 2,
                'team2_score': 1,
                'confidence': 0.9,
                'detection_method': 'scoreboard'
            }
        ]
        
        # Create a dummy frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Analyze frame (should trigger analysis since frame 0 % 10 == 0)
        result = self.analyzer.analyze_frame(frame, 0)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['team1_score'], 2)
        self.assertEqual(result['team2_score'], 1)
        self.assertEqual(result['detection_method'], 'scoreboard')
    
    def test_score_stability(self):
        """Test score stability detection."""
        # Test that score is not stable initially
        self.assertFalse(self.analyzer._is_score_stable((2, 1)))
        
        # Add some score history
        for i in range(5):
            self.analyzer.score_history.append({
                'frame_number': i * 10,
                'score': (2, 1),
                'confidence': 0.8
            })
        
        # Now it should be stable
        self.assertTrue(self.analyzer._is_score_stable((2, 1)))
        
        # Different score should not be stable
        self.assertFalse(self.analyzer._is_score_stable((1, 2)))
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Process some frames to generate statistics
        self.analyzer.frames_processed = 100
        self.analyzer.total_detections = 10
        self.analyzer.successful_extractions = 8
        self.analyzer.scoreboard_detected = True
        
        stats = self.analyzer.get_statistics()
        
        self.assertEqual(stats['frames_processed'], 100)
        self.assertEqual(stats['total_detections'], 10)
        self.assertEqual(stats['successful_extractions'], 8)
        self.assertEqual(stats['detection_rate'], 0.8)
        self.assertTrue(stats['scoreboard_detected'])
    
    def test_reset(self):
        """Test analyzer reset functionality."""
        # Add some data
        self.analyzer.frames_processed = 100
        self.analyzer.total_detections = 10
        self.analyzer.scoreboard_detected = True
        self.analyzer.score_history.append({'test': 'data'})
        
        # Reset
        self.analyzer.reset()
        
        # Check that everything is reset
        self.assertEqual(self.analyzer.frames_processed, 0)
        self.assertEqual(self.analyzer.total_detections, 0)
        self.assertFalse(self.analyzer.scoreboard_detected)
        self.assertEqual(len(self.analyzer.score_history), 0)


if __name__ == '__main__':
    unittest.main()
