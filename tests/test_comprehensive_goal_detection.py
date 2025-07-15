#!/usr/bin/env python3
"""
Comprehensive Goal Detection Test Suite

This test suite verifies the comprehensive goal detection system including:
1. Goal detection accuracy with sample footage
2. CSV file generation and format validation
3. False positive detection (ball near goal but not scored)
4. Player kick vs other ball movement distinction
5. Temporal sequence validation
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.goal_detection import (
    ComprehensiveGoalDetector,
    FieldKeypointsDetector,
    GoalCSVGenerator,
)
from src.goal_detection.goal_detection_integration import (
    GoalDetectionIntegrator,
    create_goal_detection_integrator,
)


class TestGoalCSVGenerator(unittest.TestCase):
    """Test the CSV generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_generator = GoalCSVGenerator(self.temp_dir)
        
    def test_frame_csv_generation(self):
        """Test frame-by-frame CSV generation."""
        # Add sample frame data
        self.csv_generator.add_frame_data(
            frame_num=100,
            ball_position=(500, 300),
            ball_confidence=0.8,
            ball_in_goal_area=True,
            goal_side="left",
            player_id=7,
            player_team=1,
            player_ball_distance=25.5,
            kick_detected=True,
            goal_detected=True,
            goal_confidence=0.9,
            detection_method="comprehensive",
            temporal_validation=True,
            sequence_id=1
        )
        
        # Generate CSV
        csv_path = self.csv_generator.generate_frame_csv("test_video")
        
        # Verify file exists
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['frame_num'], '100')
        self.assertEqual(row['ball_x'], '500')
        self.assertEqual(row['ball_y'], '300')
        self.assertEqual(row['ball_confidence'], '0.8')
        self.assertEqual(row['ball_in_goal_area'], 'True')
        self.assertEqual(row['goal_side'], 'left')
        self.assertEqual(row['player_id'], '7')
        self.assertEqual(row['player_team'], '1')
        self.assertEqual(row['kick_detected'], 'True')
        self.assertEqual(row['goal_detected'], 'True')
        
    def test_scoreboard_csv_generation(self):
        """Test scoreboard CSV generation."""
        # Add sample goal event
        self.csv_generator.add_goal_event(
            goal_id=1,
            frame_start=95,
            frame_end=105,
            frame_goal=100,
            team=1,
            player_id=7,
            goal_side="left",
            ball_final_position=(480, 290),
            confidence=0.9,
            validation_score=0.85,
            kick_frame=98,
            sequence_frames=[95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
            detection_method="comprehensive"
        )
        
        # Generate CSV
        csv_path = self.csv_generator.generate_scoreboard_csv("test_video")
        
        # Verify file exists
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['goal_id'], '1')
        self.assertEqual(row['frame_start'], '95')
        self.assertEqual(row['frame_end'], '105')
        self.assertEqual(row['frame_goal'], '100')
        self.assertEqual(row['team'], '1')
        self.assertEqual(row['player_id'], '7')
        self.assertEqual(row['goal_side'], 'left')
        self.assertEqual(row['ball_final_x'], '480')
        self.assertEqual(row['ball_final_y'], '290')
        self.assertEqual(row['confidence'], '0.9')
        self.assertEqual(row['kick_frame'], '98')
        self.assertEqual(row['sequence_length'], '11')
        
    def test_statistics_generation(self):
        """Test statistics generation."""
        # Add multiple frame data and goal events
        for i in range(10):
            self.csv_generator.add_frame_data(
                frame_num=i,
                ball_position=(500 + i, 300),
                ball_confidence=0.7,
                ball_in_goal_area=False,
                goal_side=None,
                player_id=None,
                player_team=None,
                player_ball_distance=None,
                kick_detected=False,
                goal_detected=False,
                goal_confidence=0.0,
                detection_method="none",
                temporal_validation=False
            )
            
        # Add one goal event
        self.csv_generator.add_goal_event(
            goal_id=1,
            frame_start=5,
            frame_end=8,
            frame_goal=7,
            team=1,
            player_id=10,
            goal_side="right",
            ball_final_position=(600, 300),
            confidence=0.8,
            validation_score=0.75,
            kick_frame=6,
            sequence_frames=[5, 6, 7, 8],
            detection_method="comprehensive"
        )
        
        stats = self.csv_generator.get_statistics()
        
        self.assertEqual(stats['total_frames_processed'], 10)
        self.assertEqual(stats['total_goals_detected'], 1)
        self.assertEqual(stats['team_goals'], {1: 1})
        self.assertEqual(stats['average_goal_confidence'], 0.8)


class TestComprehensiveGoalDetector(unittest.TestCase):
    """Test the comprehensive goal detection logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock field keypoints detector
        self.mock_keypoints_detector = MagicMock(spec=FieldKeypointsDetector)
        self.mock_keypoints_detector.is_ball_in_goal_area.return_value = None
        
        # Create CSV generator
        self.csv_generator = GoalCSVGenerator(self.temp_dir)
        
        # Create comprehensive detector
        self.detector = ComprehensiveGoalDetector(
            self.mock_keypoints_detector,
            self.csv_generator
        )
        
    def test_ball_trajectory_tracking(self):
        """Test ball trajectory tracking."""
        # Process multiple frames with ball movement
        ball_positions = [(100, 200), (110, 205), (120, 210), (130, 215)]
        
        for i, pos in enumerate(ball_positions):
            self.detector._update_ball_trajectory(pos, i)
            
        # Verify trajectory is stored
        self.assertEqual(len(self.detector.ball_trajectory), 4)
        self.assertEqual(self.detector.ball_trajectory[0], (100, 200, 0))
        self.assertEqual(self.detector.ball_trajectory[-1], (130, 215, 3))
        
    def test_player_kick_detection(self):
        """Test player kick detection logic."""
        # Set up ball trajectory with speed increase
        ball_positions = [(100, 200), (105, 202), (120, 210), (140, 220)]
        for i, pos in enumerate(ball_positions):
            self.detector._update_ball_trajectory(pos, i)
            
        # Set up player positions
        player_detections = [{
            'player_id': 7,
            'team': 1,
            'bbox': [95, 195, 115, 215],  # Close to ball
            'confidence': 0.8
        }]
        
        # Update player positions
        self.detector._update_player_positions(player_detections, 2)
        
        # Test kick detection
        kick_detected = self.detector._detect_player_kick(7, (120, 210), 3, player_detections)
        
        # Should detect kick due to speed increase and player proximity
        self.assertTrue(kick_detected)
        
    def test_false_positive_prevention(self):
        """Test that false positives are prevented."""
        # Mock ball in goal area but no kick detected
        self.mock_keypoints_detector.is_ball_in_goal_area.return_value = "left"
        
        player_detections = [{
            'player_id': 7,
            'team': 1,
            'bbox': [200, 300, 220, 320],  # Far from ball
            'confidence': 0.8
        }]
        
        # Process frame - should not detect goal without kick
        goal_detected = self.detector.process_frame(
            frame_num=10,
            ball_position=(50, 100),  # In goal area
            ball_confidence=0.8,
            player_detections=player_detections
        )
        
        self.assertFalse(goal_detected)
        
    def test_temporal_validation(self):
        """Test temporal sequence validation."""
        # Set up trajectory moving towards left goal
        trajectory_points = [
            (200, 300), (180, 295), (160, 290), (140, 285), (120, 280), (100, 275)
        ]
        
        for i, pos in enumerate(trajectory_points):
            self.detector._update_ball_trajectory(pos, i)
            
        # Test validation for left goal
        is_valid = self.detector._validate_goal_sequence((100, 275), "left", 5)
        self.assertTrue(is_valid)
        
        # Test validation for right goal (should fail)
        is_valid = self.detector._validate_goal_sequence((100, 275), "right", 5)
        self.assertFalse(is_valid)


class TestGoalDetectionIntegration(unittest.TestCase):
    """Test the goal detection integration with main pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    @patch('src.goal_detection.goal_detection_integration.FieldKeypointsDetector')
    def test_integrator_creation(self, mock_keypoints_class):
        """Test integrator creation."""
        integrator = create_goal_detection_integrator(
            output_dir=self.temp_dir,
            device="cpu"
        )
        
        self.assertIsInstance(integrator, GoalDetectionIntegrator)
        self.assertTrue(integrator.csv_generator.output_dir.exists())
        
    @patch('src.goal_detection.goal_detection_integration.FieldKeypointsDetector')
    def test_video_processing_workflow(self, mock_keypoints_class):
        """Test complete video processing workflow."""
        integrator = create_goal_detection_integrator(
            output_dir=self.temp_dir,
            device="cpu"
        )
        
        # Start video processing
        integrator.start_video_processing("test_video")
        self.assertTrue(integrator.is_processing_active())
        
        # Process some frames
        for frame_num in range(10):
            ball_detections = [{'position': (100 + frame_num, 200), 'confidence': 0.8}]
            player_detections = [{
                'player_id': 7,
                'team': 1,
                'bbox': [95 + frame_num, 195, 115 + frame_num, 215],
                'confidence': 0.8
            }]
            
            integrator.process_frame_with_detections(
                frame_num, None, ball_detections, player_detections
            )
            
        # Finalize processing
        frame_csv, scoreboard_csv = integrator.finalize_video_processing()
        
        # Verify CSV files were created
        self.assertTrue(os.path.exists(frame_csv))
        self.assertTrue(os.path.exists(scoreboard_csv))
        
        # Verify CSV files have correct names
        self.assertTrue(frame_csv.endswith("test_video_goal_detected.csv"))
        self.assertTrue(scoreboard_csv.endswith("test_video_goal_detected_scoreboard.csv"))


class TestGoalDetectionAccuracy(unittest.TestCase):
    """Test goal detection accuracy with various scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    @patch('src.goal_detection.goal_detection_integration.FieldKeypointsDetector')
    def test_goal_sequence_detection(self, mock_keypoints_class):
        """Test detection of a complete goal sequence."""
        # Mock keypoints detector to return goal area for specific positions
        mock_detector = mock_keypoints_class.return_value
        mock_detector.is_ball_in_goal_area.side_effect = lambda pos: "left" if pos[0] < 50 else None
        mock_detector.detect_keypoints.return_value = {}
        
        integrator = create_goal_detection_integrator(
            output_dir=self.temp_dir,
            device="cpu"
        )
        
        integrator.start_video_processing("goal_sequence_test")
        
        # Simulate ball moving towards goal with player kick
        ball_trajectory = [
            (200, 300), (180, 295), (160, 290), (140, 285), 
            (120, 280), (100, 275), (80, 270), (40, 265)  # Last position in goal
        ]
        
        player_positions = [
            [190, 290, 210, 310],  # Player near ball initially
            [185, 285, 205, 305],
            [180, 280, 200, 300],
            [175, 275, 195, 295],
            [170, 270, 190, 290],  # Player kicks ball
            [165, 265, 185, 285],
            [160, 260, 180, 280],
            [155, 255, 175, 275]
        ]
        
        goal_detected_frames = []
        
        for frame_num, (ball_pos, player_bbox) in enumerate(zip(ball_trajectory, player_positions)):
            ball_detections = [{'position': ball_pos, 'confidence': 0.9}]
            player_detections = [{
                'player_id': 7,
                'team': 1,
                'bbox': player_bbox,
                'confidence': 0.8
            }]
            
            goal_detected = integrator.process_frame_with_detections(
                frame_num, None, ball_detections, player_detections
            )
            
            if goal_detected:
                goal_detected_frames.append(frame_num)
                
        # Should detect goal when ball enters goal area with proper sequence
        self.assertTrue(len(goal_detected_frames) > 0)
        
        # Finalize and check CSV output
        frame_csv, scoreboard_csv = integrator.finalize_video_processing()
        
        # Verify goal event in scoreboard CSV
        with open(scoreboard_csv, 'r') as f:
            reader = csv.DictReader(f)
            goals = list(reader)
            
        self.assertTrue(len(goals) > 0)
        if goals:
            goal = goals[0]
            self.assertEqual(goal['team'], '1')
            self.assertEqual(goal['player_id'], '7')
            self.assertEqual(goal['goal_side'], 'left')


def run_comprehensive_tests():
    """Run all comprehensive goal detection tests."""
    print("🧪 Running Comprehensive Goal Detection Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGoalCSVGenerator,
        TestComprehensiveGoalDetector,
        TestGoalDetectionIntegration,
        TestGoalDetectionAccuracy
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_comprehensive_tests()
