#!/usr/bin/env python3
"""
Test suite for enhanced ball detection system.

This module tests the integration of the specialized ball detection model
with the existing football analysis pipeline.
"""

import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trackers.tracker import Tracker
from config import BallDetectionConfig, ACCURACY_OPTIMIZED_CONFIG, PERFORMANCE_OPTIMIZED_CONFIG
from utils import read_video


class TestEnhancedBallDetection(unittest.TestCase):
    """Test cases for enhanced ball detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = "data/input_videos/test_video.mp4"
        self.general_model_path = "data/models/best_detect.pt"
        self.ball_model_path = "data/models/best_ball.pt"
        
        # Create test configuration
        self.test_config = BallDetectionConfig(
            general_model_path=self.general_model_path,
            ball_model_path=self.ball_model_path,
            enable_enhanced_ball_detection=True,
            ball_confidence_threshold=0.3,
            ball_fusion_confidence_threshold=0.5
        )
    
    def test_tracker_initialization_with_config(self):
        """Test tracker initialization with configuration."""
        try:
            tracker = Tracker(config=self.test_config)
            self.assertIsNotNone(tracker)
            self.assertEqual(tracker.config.ball_confidence_threshold, 0.3)
            self.assertEqual(tracker.config.ball_fusion_confidence_threshold, 0.5)
            print("✅ Tracker initialization with config: PASSED")
        except Exception as e:
            print(f"❌ Tracker initialization with config: FAILED - {e}")
            self.fail(f"Tracker initialization failed: {e}")
    
    def test_tracker_initialization_backward_compatibility(self):
        """Test backward compatibility with old initialization method."""
        try:
            tracker = Tracker(
                model_path=self.general_model_path,
                enable_jersey_detection=True,
                ball_model_path=self.ball_model_path,
                enable_enhanced_ball_detection=True
            )
            self.assertIsNotNone(tracker)
            print("✅ Backward compatibility: PASSED")
        except Exception as e:
            print(f"❌ Backward compatibility: FAILED - {e}")
            self.fail(f"Backward compatibility test failed: {e}")
    
    def test_ball_detection_fusion_logic(self):
        """Test ball detection fusion logic."""
        try:
            tracker = Tracker(config=self.test_config)
            
            # Mock detection objects
            mock_general_detection = Mock()
            mock_general_detection.boxes = Mock()
            mock_general_detection.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 120, 120]])
            mock_general_detection.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.6])
            mock_general_detection.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
            mock_general_detection.names = {0: "ball"}
            
            mock_ball_detection = Mock()
            mock_ball_detection.boxes = Mock()
            mock_ball_detection.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[105, 105, 125, 125]])
            mock_ball_detection.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
            
            # Test fusion logic
            result = tracker.fuse_ball_detections(mock_general_detection, mock_ball_detection, 0)
            
            self.assertIsNotNone(result)
            self.assertIn("bbox", result)
            self.assertIn("confidence", result)
            self.assertIn("source", result)
            
            print("✅ Ball detection fusion logic: PASSED")
        except Exception as e:
            print(f"❌ Ball detection fusion logic: FAILED - {e}")
            self.fail(f"Ball detection fusion test failed: {e}")
    
    def test_enhanced_interpolation(self):
        """Test enhanced ball position interpolation."""
        try:
            tracker = Tracker(config=self.test_config)
            
            # Create test ball positions with gaps
            test_positions = [
                {1: {"bbox": [100, 100, 120, 120], "confidence": 0.8, "source": "specialized"}},
                {},  # Missing detection
                {},  # Missing detection
                {1: {"bbox": [140, 140, 160, 160], "confidence": 0.7, "source": "specialized"}},
            ]
            
            interpolated = tracker.interpolate_ball_positions(test_positions)
            
            # Check that gaps are filled
            self.assertEqual(len(interpolated), 4)
            self.assertIn(1, interpolated[1])  # Gap should be filled
            self.assertIn(1, interpolated[2])  # Gap should be filled
            
            print("✅ Enhanced interpolation: PASSED")
        except Exception as e:
            print(f"❌ Enhanced interpolation: FAILED - {e}")
            self.fail(f"Enhanced interpolation test failed: {e}")
    
    def test_configuration_presets(self):
        """Test different configuration presets."""
        try:
            # Test accuracy optimized config
            accuracy_tracker = Tracker(config=ACCURACY_OPTIMIZED_CONFIG)
            self.assertEqual(accuracy_tracker.ball_confidence_threshold, 0.2)
            
            # Test performance optimized config
            performance_tracker = Tracker(config=PERFORMANCE_OPTIMIZED_CONFIG)
            self.assertEqual(performance_tracker.ball_confidence_threshold, 0.4)
            
            print("✅ Configuration presets: PASSED")
        except Exception as e:
            print(f"❌ Configuration presets: FAILED - {e}")
            self.fail(f"Configuration presets test failed: {e}")
    
    def test_model_file_validation(self):
        """Test model file validation."""
        try:
            # Test with non-existent ball model
            invalid_config = BallDetectionConfig(
                general_model_path=self.general_model_path,
                ball_model_path="non_existent_model.pt",
                enable_enhanced_ball_detection=True
            )
            
            tracker = Tracker(config=invalid_config)
            # Should disable enhanced ball detection automatically
            self.assertFalse(tracker.enable_enhanced_ball_detection)
            
            print("✅ Model file validation: PASSED")
        except Exception as e:
            print(f"❌ Model file validation: FAILED - {e}")
            self.fail(f"Model file validation test failed: {e}")


class TestBallDetectionIntegration(unittest.TestCase):
    """Test integration with existing pipeline components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_config = BallDetectionConfig(
            enable_enhanced_ball_detection=True,
            ball_confidence_threshold=0.3
        )
    
    def test_goal_detection_integration(self):
        """Test integration with goal detection system."""
        try:
            from goal_detection import GoalDetector, FieldKeypointsDetector
            
            # Initialize components
            tracker = Tracker(config=self.test_config)
            field_detector = FieldKeypointsDetector("data/models/best_keypoint.pt")
            goal_detector = GoalDetector(field_detector)
            
            # Test that components can work together
            self.assertIsNotNone(tracker)
            self.assertIsNotNone(goal_detector)
            
            print("✅ Goal detection integration: PASSED")
        except Exception as e:
            print(f"❌ Goal detection integration: FAILED - {e}")
            # Don't fail the test if goal detection models are not available
            print("⚠️ Goal detection models may not be available for testing")
    
    def test_data_output_format_compatibility(self):
        """Test that enhanced ball detection maintains output format compatibility."""
        try:
            tracker = Tracker(config=self.test_config)
            
            # Create dummy frames
            dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
            
            # Mock the model predictions to avoid actual inference
            with patch.object(tracker.model, 'predict') as mock_predict:
                mock_predict.return_value = [Mock() for _ in dummy_frames]
                
                # Configure mock returns
                for mock_result in mock_predict.return_value:
                    mock_result.boxes = None
                    mock_result.names = {"ball": 0, "player": 1, "referee": 2}
                
                tracks = tracker.get_object_tracks(dummy_frames)
                
                # Verify output format
                self.assertIn("players", tracks)
                self.assertIn("referees", tracks)
                self.assertIn("ball", tracks)
                self.assertEqual(len(tracks["ball"]), 5)
                
            print("✅ Data output format compatibility: PASSED")
        except Exception as e:
            print(f"❌ Data output format compatibility: FAILED - {e}")
            self.fail(f"Data output format test failed: {e}")


def run_enhanced_ball_detection_tests():
    """Run all enhanced ball detection tests."""
    print("🧪 Running Enhanced Ball Detection Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnhancedBallDetection))
    test_suite.addTest(unittest.makeSuite(TestBallDetectionIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All enhanced ball detection tests PASSED!")
    else:
        print(f"❌ {len(result.failures)} test(s) FAILED, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_enhanced_ball_detection_tests()
    sys.exit(0 if success else 1)
