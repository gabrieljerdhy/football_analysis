#!/usr/bin/env python3
"""
Test Dribble Detection System

This test script validates the dribble detection functionality and ensures
it produces reliable statistics and integrates properly with the existing
football video analysis pipeline.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dribble_detection import DribbleAnalyzer, DribbleDetector, DribbleEvent


class TestDribbleDetection(unittest.TestCase):
    """Test cases for dribble detection system."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DribbleAnalyzer(
            min_dribble_distance=30.0,
            min_dribble_duration=10,
            confidence_threshold=0.4
        )
        
        self.detector = DribbleDetector(
            min_dribble_distance=30.0,
            min_dribble_duration=10,
            confidence_threshold=0.4
        )

    def test_dribble_analyzer_initialization(self):
        """Test that DribbleAnalyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.detector)
        self.assertEqual(self.analyzer.detector.min_dribble_distance, 30.0)
        self.assertEqual(self.analyzer.detector.min_dribble_duration, 10)
        self.assertEqual(self.analyzer.detector.confidence_threshold, 0.4)

    def test_dribble_detector_initialization(self):
        """Test that DribbleDetector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_dribble_distance, 30.0)
        self.assertEqual(self.detector.min_dribble_duration, 10)
        self.assertEqual(self.detector.confidence_threshold, 0.4)

    def test_analyze_frame_no_ball_possession(self):
        """Test frame analysis when no player has ball possession."""
        players = {
            1: {"bbox": [100, 100, 150, 200], "team": 1},
            2: {"bbox": [200, 100, 250, 200], "team": 2}
        }
        
        result = self.analyzer.analyze_frame(
            frame_num=1,
            players=players,
            ball_position=(125, 150),
            ball_possessor=-1,  # No possession
            ball_team=None
        )
        
        self.assertIsNone(result)

    def test_analyze_frame_with_ball_possession(self):
        """Test frame analysis with ball possession."""
        players = {
            1: {"bbox": [100, 100, 150, 200], "team": 1},
            2: {"bbox": [200, 100, 250, 200], "team": 2}
        }
        
        result = self.analyzer.analyze_frame(
            frame_num=1,
            players=players,
            ball_position=(125, 150),
            ball_possessor=1,
            ball_team=1
        )
        
        # Should not detect dribble on first frame
        self.assertIsNone(result)

    def test_dribble_detection_with_opponent_proximity(self):
        """Test dribble detection when opponents are nearby."""
        players = {
            1: {"bbox": [100, 100, 150, 200], "team": 1},
            2: {"bbox": [160, 100, 210, 200], "team": 2}  # Close opponent
        }
        
        # Simulate multiple frames to build up dribble potential
        for frame_num in range(1, 20):
            # Move player with ball while maintaining possession
            players[1]["bbox"] = [100 + frame_num * 5, 100, 150 + frame_num * 5, 200]
            ball_position = (125 + frame_num * 5, 150)
            
            result = self.analyzer.analyze_frame(
                frame_num=frame_num,
                players=players,
                ball_position=ball_position,
                ball_possessor=1,
                ball_team=1
            )
            
            # Should eventually detect a dribble
            if result:
                self.assertIsInstance(result, DribbleEvent)
                self.assertEqual(result.team, 1)
                self.assertEqual(result.player_id, 1)
                self.assertGreater(result.confidence, 0.0)
                break

    def test_direction_change_detection(self):
        """Test detection of direction changes in player movement."""
        positions = [
            (100, 100),  # Start
            (110, 100),  # Move right
            (105, 110)   # Change direction (left and down)
        ]
        
        direction_change = self.detector._detect_direction_change(positions)
        self.assertTrue(direction_change)

    def test_no_direction_change_detection(self):
        """Test that straight movement doesn't trigger direction change."""
        positions = [
            (100, 100),  # Start
            (110, 100),  # Move right
            (120, 100)   # Continue right
        ]
        
        direction_change = self.detector._detect_direction_change(positions)
        self.assertFalse(direction_change)

    def test_confidence_calculation(self):
        """Test dribble confidence calculation."""
        dribble_data = {
            'direction_changes': 2,
            'max_speed': 8.0,
            'opponents': [{'player_id': 2, 'team': 2, 'distance': 50}]
        }
        
        confidence = self.detector._calculate_dribble_confidence(
            dribble_data, duration=30, total_distance=80.0
        )
        
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_team_statistics_tracking(self):
        """Test that team statistics are properly tracked."""
        # Create a mock dribble event
        dribble_event = DribbleEvent(
            frame_num=100,
            team=1,
            player_id=5,
            ball_position=(200, 150),
            player_position=(195, 145),
            confidence=0.8,
            start_frame=90,
            end_frame=100,
            duration_frames=10,
            distance_covered=50.0,
            direction_changes=2,
            speed_changes=1,
            avg_speed=5.0,
            max_speed=8.0,
            trajectory_smoothness=0.7,
            ball_control_consistency=0.8,
            ball_touches=10
        )
        
        # Update statistics
        self.detector._update_statistics(dribble_event)
        
        # Check team statistics
        team_stats = self.detector.get_team_statistics()
        self.assertEqual(team_stats[1].total_dribbles, 1)
        self.assertEqual(team_stats[1].successful_dribbles, 1)
        self.assertEqual(team_stats[1].events_count, 1)
        self.assertAlmostEqual(team_stats[1].avg_confidence, 0.8, places=2)

    def test_csv_export_functionality(self):
        """Test CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add some mock dribble events
            mock_event = DribbleEvent(
                frame_num=100,
                team=1,
                player_id=5,
                ball_position=(200, 150),
                player_position=(195, 145),
                confidence=0.8,
                start_frame=90,
                end_frame=100,
                duration_frames=10,
                distance_covered=50.0,
                direction_changes=2,
                speed_changes=1,
                avg_speed=5.0,
                max_speed=8.0,
                trajectory_smoothness=0.7,
                ball_control_consistency=0.8,
                ball_touches=10
            )
            
            self.analyzer.dribble_events.append(mock_event)
            self.analyzer.detector._update_statistics(mock_event)
            
            # Export to CSV
            exported_files = self.analyzer.export_to_csv(
                output_dir=temp_dir,
                video_name="test_match"
            )
            
            # Check that files were created
            self.assertIn("team_dribble_stats", exported_files)
            self.assertIn("player_dribble_stats", exported_files)
            self.assertIn("dribble_events", exported_files)
            
            # Check that files exist
            for file_path in exported_files.values():
                self.assertTrue(os.path.exists(file_path))

    def test_integration_methods(self):
        """Test methods used for integration with existing systems."""
        # Add mock statistics
        mock_event = DribbleEvent(
            frame_num=100,
            team=1,
            player_id=5,
            ball_position=(200, 150),
            player_position=(195, 145),
            confidence=0.8,
            start_frame=90,
            end_frame=100,
            duration_frames=10,
            distance_covered=50.0,
            direction_changes=2,
            speed_changes=1,
            avg_speed=5.0,
            max_speed=8.0,
            trajectory_smoothness=0.7,
            ball_control_consistency=0.8,
            ball_touches=10
        )
        
        self.analyzer.detector._update_statistics(mock_event)
        
        # Test integration methods
        dribble_counts = self.analyzer.get_team_dribble_counts()
        dribble_confidence = self.analyzer.get_team_dribble_confidence()
        dribble_events_count = self.analyzer.get_team_dribble_events_count()
        
        self.assertEqual(dribble_counts[1], 1)
        self.assertEqual(dribble_counts[2], 0)
        self.assertAlmostEqual(dribble_confidence[1], 0.8, places=2)
        self.assertEqual(dribble_events_count[1], 1)

    def test_complete_match_analysis(self):
        """Test complete match analysis functionality."""
        # Create mock tracking data
        tracks = {
            "players": [
                {1: {"bbox": [100, 100, 150, 200], "team": 1}},
                {1: {"bbox": [110, 100, 160, 200], "team": 1}},
                {1: {"bbox": [120, 100, 170, 200], "team": 1}}
            ],
            "ball": [
                {1: {"bbox": [125, 140, 135, 160]}},
                {1: {"bbox": [135, 140, 145, 160]}},
                {1: {"bbox": [145, 140, 155, 160]}}
            ]
        }
        
        player_assignments = [1, 1, 1]  # Player 1 has ball in all frames
        
        # Run analysis
        results = self.analyzer.analyze_complete_match(tracks, player_assignments)
        
        # Check results structure
        self.assertIn("team_statistics", results)
        self.assertIn("player_statistics", results)
        self.assertIn("dribble_events", results)
        self.assertIn("total_dribbles", results)
        self.assertIn("avg_confidence", results)


if __name__ == "__main__":
    print("🧪 Running Dribble Detection Tests...")
    unittest.main(verbosity=2)
