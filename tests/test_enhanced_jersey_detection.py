#!/usr/bin/env python3
"""
Test script for enhanced jersey number detection functionality.

This script tests the improved jersey number detection system to ensure:
1. Proper validation for jersey numbers in the 0-99 range
2. Correct filtering of invalid detections (negative, >99, non-numeric)
3. Accurate CSV output with jersey number data
4. Integration with the player tracking system
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jersey_number_detector.jersey_number_detector import JerseyNumberDetector
from trackers.tracker import Tracker
from utils.goal_utils import export_consolidated_goal_statistics


class TestEnhancedJerseyDetection(unittest.TestCase):
    """Test cases for enhanced jersey number detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = JerseyNumberDetector(
            confidence_threshold=0.3,
            consensus_frames=3,
            valid_number_range=(0, 99)
        )
        
        # Create a simple test image
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Mock bbox
        self.test_bbox = [10, 10, 90, 90]

    def test_valid_range_validation(self):
        """Test that jersey numbers are validated within 0-99 range."""
        # Test valid numbers
        valid_numbers = [0, 1, 10, 50, 99]
        for number in valid_numbers:
            result = self.detector._is_valid_jersey_number(number, str(number))
            self.assertTrue(result, f"Number {number} should be valid")

        # Test invalid numbers
        invalid_numbers = [-1, 100, 101, 999]
        for number in invalid_numbers:
            result = self.detector._is_valid_jersey_number(number, str(number))
            self.assertFalse(result, f"Number {number} should be invalid")

    def test_edge_case_validation(self):
        """Test validation of edge cases."""
        # Test negative numbers
        self.assertFalse(self.detector._is_valid_jersey_number(-5, "-5"))
        
        # Test numbers > 99
        self.assertFalse(self.detector._is_valid_jersey_number(100, "100"))
        self.assertFalse(self.detector._is_valid_jersey_number(123, "123"))
        
        # Test jersey number 0 (should be valid but require high confidence)
        self.assertTrue(self.detector._is_valid_jersey_number(0, "0"))

    def test_invalid_pattern_detection(self):
        """Test detection of invalid patterns in OCR text."""
        # Test negative number patterns
        self.assertTrue(self.detector._contains_invalid_patterns("-5"))
        self.assertTrue(self.detector._contains_invalid_patterns("player-10"))
        
        # Test decimal patterns
        self.assertTrue(self.detector._contains_invalid_patterns("12.5"))
        self.assertTrue(self.detector._contains_invalid_patterns("3.14"))
        
        # Test very long sequences
        self.assertTrue(self.detector._contains_invalid_patterns("12345"))
        self.assertTrue(self.detector._contains_invalid_patterns("9876543"))
        
        # Test timestamp patterns
        self.assertTrue(self.detector._contains_invalid_patterns("12:34"))
        self.assertTrue(self.detector._contains_invalid_patterns("2-1"))
        
        # Test valid patterns
        self.assertFalse(self.detector._contains_invalid_patterns("10"))
        self.assertFalse(self.detector._contains_invalid_patterns("99"))
        self.assertFalse(self.detector._contains_invalid_patterns("0"))

    def test_number_extraction_and_validation(self):
        """Test the complete number extraction and validation pipeline."""
        # Test valid single numbers
        valid_cases = [
            ("10", 0.8),
            ("99", 0.9),
            ("0", 0.95),  # Should require high confidence
            ("7", 0.7),
        ]
        
        for text, confidence in valid_cases:
            results = self.detector._extract_and_validate_numbers(text, confidence)
            self.assertGreater(len(results), 0, f"Should extract number from '{text}'")
            number, conf = results[0]
            self.assertGreaterEqual(number, 0, f"Number should be >= 0")
            self.assertLessEqual(number, 99, f"Number should be <= 99")

        # Test invalid cases
        invalid_cases = [
            ("-5", 0.8),  # Negative
            ("100", 0.8),  # Too high
            ("12.5", 0.8),  # Decimal
            ("12345", 0.8),  # Too long
            ("12:34", 0.8),  # Timestamp
        ]
        
        for text, confidence in invalid_cases:
            results = self.detector._extract_and_validate_numbers(text, confidence)
            self.assertEqual(len(results), 0, f"Should not extract number from '{text}'")

    def test_consensus_validation(self):
        """Test multi-frame consensus with enhanced validation."""
        track_id = 1
        
        # Add valid detections
        valid_detections = [(10, 0.8), (10, 0.9), (10, 0.85)]
        for number, confidence in valid_detections:
            self.detector.player_jersey_history[track_id].append((number, confidence))
        
        # Test consensus
        consensus = self.detector._get_consensus_number(track_id)
        self.assertEqual(consensus, 10, "Should reach consensus on number 10")
        
        # Test with invalid numbers mixed in
        track_id_2 = 2
        mixed_detections = [(15, 0.8), (100, 0.9), (15, 0.85), (15, 0.9)]  # 100 is invalid
        for number, confidence in mixed_detections:
            if 0 <= number <= 99:  # Only add valid numbers to history
                self.detector.player_jersey_history[track_id_2].append((number, confidence))
        
        consensus_2 = self.detector._get_consensus_number(track_id_2)
        self.assertEqual(consensus_2, 15, "Should reach consensus on valid number 15, ignoring invalid 100")

    def test_csv_export_with_jersey_numbers(self):
        """Test CSV export includes jersey number column."""
        # Create mock data
        mock_pass_counter = Mock()
        mock_pass_counter.player_passes = {1: {"team": 1, "passes": 5}}
        mock_pass_counter.player_goals = {1: {"team": 1, "goals": 1}}
        mock_pass_counter.team_passes = {1: 10, 2: 8}
        mock_pass_counter.team_goals = {1: 2, 2: 1}
        
        mock_tackle_counter = Mock()
        mock_tackle_counter.player_tackles = {1: {"team": 1, "tackles": 3}}
        mock_tackle_counter.player_interceptions = {1: {"team": 1, "interceptions": 2}}
        mock_tackle_counter.team_tackles = {1: 5, 2: 4}
        mock_tackle_counter.team_interceptions = {1: 3, 2: 2}
        
        # Create mock tracker with jersey detection
        mock_tracker = Mock()
        mock_tracker.jersey_detector = Mock()
        mock_tracker.jersey_detector.get_jersey_number_for_player.return_value = 10
        
        enhanced_goal_stats = {"team_goals": {1: 1, 2: 0}, "goal_events": []}
        final_team_goals = {1: 1, 2: 0}
        final_player_goals = {}
        
        # Test CSV export
        with tempfile.TemporaryDirectory() as temp_dir:
            team_csv, player_csv = export_consolidated_goal_statistics(
                "test_match",
                mock_pass_counter,
                enhanced_goal_stats,
                final_team_goals,
                final_player_goals,
                mock_tackle_counter,
                output_dir=temp_dir,
                tracker=mock_tracker
            )
            
            # Verify player CSV contains jersey_number column
            self.assertTrue(os.path.exists(player_csv), "Player CSV should be created")
            
            with open(player_csv, 'r') as f:
                content = f.read()
                self.assertIn("jersey_number", content, "CSV should contain jersey_number column")
                self.assertIn("10", content, "CSV should contain the detected jersey number")

    def test_detection_statistics(self):
        """Test that detection statistics are properly tracked."""
        # Get initial stats
        initial_stats = self.detector.get_detection_stats()
        
        # Simulate some detections
        track_id = 1
        self.detector.player_jersey_history[track_id] = [(10, 0.8), (10, 0.9)]
        self.detector.player_jersey_cache[track_id] = 10
        
        # Update stats
        self.detector.validation_stats["valid_numbers"] = 2
        self.detector.validation_stats["invalid_range"] = 1
        
        stats = self.detector.get_detection_stats()
        
        self.assertEqual(stats["confirmed_players"], 1)
        self.assertEqual(stats["validation_stats"]["valid_numbers"], 2)
        self.assertEqual(stats["validation_stats"]["invalid_range"], 1)

    def test_jersey_number_zero_handling(self):
        """Test special handling of jersey number 0."""
        # Jersey number 0 should be valid but require higher confidence
        result = self.detector._validate_number_context(0, "0", 0.3)  # Low confidence
        self.assertFalse(result, "Jersey number 0 should require higher confidence")
        
        result = self.detector._validate_number_context(0, "0", 0.6)  # High confidence
        self.assertTrue(result, "Jersey number 0 should be valid with high confidence")


def run_jersey_detection_tests():
    """Run all jersey detection tests."""
    print("🧪 Running Enhanced Jersey Number Detection Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedJerseyDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("✅ All jersey detection tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_jersey_detection_tests()
    sys.exit(0 if success else 1)
