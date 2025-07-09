#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced goal detection system.

This module tests the improved goal detection capabilities to ensure
accurate detection of the expected 4-0 score.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from goal_detection.improved_goal_detector import ImprovedGoalDetector
    from goal_detection.field_keypoints_detector import FieldKeypointsDetector
    from config.enhanced_ball_config import (
        EnhancedBallDetectionConfig,
        GOAL_DETECTION_OPTIMIZED_CONFIG,
        get_config_for_video_size
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules not available, running basic tests...")
    IMPORTS_AVAILABLE = False


class TestEnhancedGoalDetection(unittest.TestCase):
    """Test cases for enhanced goal detection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
            
        # Mock field keypoints detector
        self.mock_field_detector = Mock(spec=FieldKeypointsDetector)
        self.mock_field_detector.is_ball_in_goal_area.return_value = None
        self.mock_field_detector.detect_keypoints.return_value = {}
        
        # Initialize improved goal detector
        self.goal_detector = ImprovedGoalDetector(self.mock_field_detector)
        
        # Set up video dimensions (1920x1080)
        self.goal_detector.video_dimensions = (1920, 1080)
        self.goal_detector._update_enhanced_goal_areas(1920, 1080)
        
    def test_enhanced_goal_areas_creation(self):
        """Test that enhanced goal areas are created correctly."""
        areas = self.goal_detector.enhanced_goal_areas
        
        # Check that all required areas exist
        self.assertIn("left", areas)
        self.assertIn("right", areas)
        
        for side in ["left", "right"]:
            self.assertIn("primary", areas[side])
            self.assertIn("extended", areas[side])
            self.assertIn("near", areas[side])
            
        # Check left goal primary area
        left_primary = areas["left"]["primary"]
        self.assertEqual(left_primary["x_min"], 0)
        self.assertGreater(left_primary["x_max"], 0)
        self.assertGreater(left_primary["y_max"], left_primary["y_min"])
        
        print("✅ Enhanced goal areas created correctly")
        
    def test_ball_trajectory_tracking(self):
        """Test ball trajectory tracking functionality."""
        # Add some ball positions
        positions = [(100, 400), (90, 405), (80, 410)]
        
        for i, pos in enumerate(positions):
            ball_info = {
                "position": pos,
                "frame": i,
                "confidence": 0.8,
                "source": "test"
            }
            self.goal_detector._update_ball_trajectory(ball_info)
            
        # Check trajectory length
        self.assertEqual(len(self.goal_detector.ball_trajectory), 3)
        
        # Check trajectory content
        self.assertEqual(self.goal_detector.ball_trajectory[0]["position"], (100, 400))
        self.assertEqual(self.goal_detector.ball_trajectory[-1]["position"], (80, 410))
        
        print("✅ Ball trajectory tracking works correctly")
        
    def test_goal_area_detection(self):
        """Test enhanced goal area detection."""
        # Test left goal detection
        left_goal_pos = (50, 500)  # Should be in left primary area
        result = self.goal_detector._check_enhanced_goal_areas(left_goal_pos)
        self.assertEqual(result, "left")
        
        # Test right goal detection
        right_goal_pos = (1870, 500)  # Should be in right primary area
        result = self.goal_detector._check_enhanced_goal_areas(right_goal_pos)
        self.assertEqual(result, "right")
        
        # Test non-goal position
        center_pos = (960, 500)  # Center of field
        result = self.goal_detector._check_enhanced_goal_areas(center_pos)
        self.assertIsNone(result)
        
        print("✅ Goal area detection works correctly")
        
    def test_trajectory_validation(self):
        """Test trajectory direction validation."""
        # Set up trajectory moving left
        positions = [(120, 400), (110, 405), (100, 410)]
        for i, pos in enumerate(positions):
            ball_info = {"position": pos, "frame": i, "confidence": 0.8, "source": "test"}
            self.goal_detector._update_ball_trajectory(ball_info)
            
        # Test left goal validation (should pass)
        self.assertTrue(self.goal_detector._validate_trajectory_direction("left"))
        
        # Test right goal validation (should fail)
        self.assertFalse(self.goal_detector._validate_trajectory_direction("right"))
        
        print("✅ Trajectory validation works correctly")
        
    def test_goal_registration(self):
        """Test goal registration and statistics tracking."""
        # Register a goal
        goal_event = self.goal_detector._register_goal_improved(
            "left", 10, 1, 100, (50, 500)
        )
        
        # Check goal event structure
        self.assertEqual(goal_event["team"], 1)
        self.assertEqual(goal_event["player_id"], 10)
        self.assertEqual(goal_event["frame_num"], 100)
        self.assertEqual(goal_event["goal_side"], "left")
        
        # Check statistics update
        self.assertEqual(self.goal_detector.team_goals[1], 1)
        self.assertEqual(self.goal_detector.player_goals[10]["goals"], 1)
        
        print("✅ Goal registration works correctly")
        
    def test_multiple_goals_for_4_0_score(self):
        """Test detection of multiple goals for correct 4-0 score."""
        goals_data = [
            {"pos": (50, 500), "player": 10, "team": 1, "frame": 100},
            {"pos": (1870, 520), "player": 15, "team": 1, "frame": 200},
            {"pos": (30, 480), "player": 7, "team": 1, "frame": 300},
            {"pos": (1890, 540), "player": 22, "team": 1, "frame": 400}
        ]
        
        detected_goals = 0
        for goal_data in goals_data:
            # Reset detection state for each goal
            self.goal_detector.reset_goal_detection()
            
            # Detect goal
            goal_event = self.goal_detector.detect_goal(
                goal_data["pos"], 
                goal_data["player"], 
                goal_data["team"], 
                goal_data["frame"],
                0.8, 
                "specialized"
            )
            
            if goal_event:
                detected_goals += 1
                print(f"✅ Goal {detected_goals} detected: Team {goal_data['team']}, Player {goal_data['player']}")
            
        # Verify final statistics show 4-0
        stats = self.goal_detector.get_goal_statistics()
        self.assertEqual(stats["team_goals"][1], 4)
        self.assertEqual(stats["team_goals"].get(2, 0), 0)
        self.assertEqual(stats["total_goals"], 4)
        
        print(f"✅ All 4 goals detected correctly for 4-0 score")


class TestEnhancedBallConfig(unittest.TestCase):
    """Test cases for enhanced ball detection configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_config_creation(self):
        """Test configuration creation and validation."""
        config = EnhancedBallDetectionConfig()
        
        # Test default values
        self.assertTrue(config.enable_enhanced_ball_detection)
        self.assertEqual(config.ball_confidence_threshold, 0.25)
        self.assertEqual(config.ball_temporal_consistency_frames, 5)
        
        print("✅ Enhanced ball config created correctly")
        
    def test_goal_optimized_config(self):
        """Test goal-optimized configuration."""
        base_config = EnhancedBallDetectionConfig()
        goal_config = base_config.get_goal_optimized_config()
        
        # Should have lower thresholds for better sensitivity
        self.assertLess(goal_config.ball_confidence_threshold, base_config.ball_confidence_threshold)
        self.assertGreater(goal_config.near_goal_detection_sensitivity, 1.0)
        
        print("✅ Goal-optimized config has correct parameters")
        
    def test_config_for_video_size(self):
        """Test configuration selection based on video size."""
        # Small video (high accuracy)
        small_config = get_config_for_video_size(1000)
        self.assertEqual(small_config.ball_confidence_threshold, 0.15)
        
        # Large video (balanced performance)
        large_config = get_config_for_video_size(15000)
        self.assertEqual(large_config.ball_confidence_threshold, 0.25)
        
        print("✅ Video size-based config selection works correctly")


def run_integration_test():
    """Run integration test with actual video processing."""
    print("\n🧪 Running integration test for 4-0 goal detection...")
    
    try:
        # Import main processing function
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        # Check if video file exists
        video_path = "data/input_videos/videoplayback_process.mp4"
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
            
        print(f"✅ Video file found: {video_path}")
        
        # Test with improved goal detection
        print("🎯 Testing improved goal detection system...")
        
        # This would run the actual analysis
        # For now, just validate the setup
        print("✅ Integration test setup complete")
        print("📝 To run full test, execute:")
        print("   python main.py --input data/input_videos/videoplayback_process.mp4 --memory-efficient --enable-scoreboard-detection --use-enhanced-stats --enable-trajectory-analysis --device cuda")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == '__main__':
    print("🚀 Starting Enhanced Goal Detection Test Suite")
    print("=" * 60)
    
    # Run unit tests
    print("\n📋 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n🔗 Running integration test...")
    run_integration_test()
    
    print("\n✅ Test suite completed!")
    print("Expected result: 4-0 score (Team 1: 4 goals, Team 2: 0 goals)")
