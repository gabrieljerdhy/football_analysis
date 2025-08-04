#!/usr/bin/env python3
"""
Test Dribble Detection Integration

This test script validates that the dribble detection system integrates
properly with the main football video analysis pipeline and produces
the expected CSV output format.
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dribble_detection import DribbleAnalyzer
from src.pass_counter.pass_counter import PassCounter
from src.pass_counter.tackle_counter import TackleCounter
from src.utils.goal_utils import export_consolidated_goal_statistics


class TestDribbleIntegration(unittest.TestCase):
    """Test cases for dribble detection integration with main pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.dribble_analyzer = DribbleAnalyzer(
            min_dribble_distance=30.0,
            min_dribble_duration=10,
            confidence_threshold=0.4
        )
        
        # Create mock components
        self.pass_counter = PassCounter()
        self.tackle_counter = TackleCounter()
        
        # Add some mock data
        self.pass_counter.team_passes = {1: 50, 2: 45}
        self.pass_counter.team_goals = {1: 2, 2: 1}
        self.tackle_counter.team_tackles = {1: 15, 2: 18}
        self.tackle_counter.team_interceptions = {1: 8, 2: 12}

    def test_csv_export_integration(self):
        """Test that dribble detection integrates properly with CSV export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add mock dribble statistics
            from src.dribble_detection.dribble_event import DribbleEvent
            
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
            
            # Add event to analyzer
            self.dribble_analyzer.dribble_events.append(mock_event)
            self.dribble_analyzer.detector._update_statistics(mock_event)
            
            # Mock enhanced goal stats
            enhanced_goal_stats = {
                "team_goals": {1: 2, 2: 1},
                "total_goals": 3,
                "goal_events": []
            }
            
            final_team_goals = {1: 2, 2: 1}
            final_player_goals = {}
            
            # Test CSV export with dribble analyzer
            team_csv_path, player_csv_path = export_consolidated_goal_statistics(
                video_name="test_integration",
                pass_counter=self.pass_counter,
                enhanced_goal_stats=enhanced_goal_stats,
                final_team_goals=final_team_goals,
                final_player_goals=final_player_goals,
                tackle_counter=self.tackle_counter,
                output_dir=temp_dir,
                dribble_analyzer=self.dribble_analyzer
            )
            
            # Verify files were created
            self.assertTrue(os.path.exists(team_csv_path))
            self.assertTrue(os.path.exists(player_csv_path))
            
            # Verify team CSV contains dribble columns
            with open(team_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check for dribble columns
                self.assertIn("dribbles_detected", headers)
                self.assertIn("avg_dribble_confidence", headers)
                self.assertIn("dribble_events_count", headers)
                
                # Check data
                rows = list(reader)
                self.assertEqual(len(rows), 2)  # Two teams
                
                # Team 1 should have dribble data
                team1_row = next(row for row in rows if row["team"] == "1")
                self.assertEqual(team1_row["dribbles_detected"], "1")
                self.assertEqual(team1_row["avg_dribble_confidence"], "0.8")
                self.assertEqual(team1_row["dribble_events_count"], "1")
                
                # Team 2 should have zero dribbles
                team2_row = next(row for row in rows if row["team"] == "2")
                self.assertEqual(team2_row["dribbles_detected"], "0")
                self.assertEqual(team2_row["avg_dribble_confidence"], "0.0")
                self.assertEqual(team2_row["dribble_events_count"], "0")

    def test_csv_export_without_dribble_analyzer(self):
        """Test that CSV export works when dribble analyzer is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_goal_stats = {
                "team_goals": {1: 2, 2: 1},
                "total_goals": 3,
                "goal_events": []
            }
            
            final_team_goals = {1: 2, 2: 1}
            final_player_goals = {}
            
            # Test CSV export without dribble analyzer
            team_csv_path, player_csv_path = export_consolidated_goal_statistics(
                video_name="test_no_dribble",
                pass_counter=self.pass_counter,
                enhanced_goal_stats=enhanced_goal_stats,
                final_team_goals=final_team_goals,
                final_player_goals=final_player_goals,
                tackle_counter=self.tackle_counter,
                output_dir=temp_dir,
                dribble_analyzer=None  # No dribble analyzer
            )
            
            # Verify files were created
            self.assertTrue(os.path.exists(team_csv_path))
            self.assertTrue(os.path.exists(player_csv_path))
            
            # Verify team CSV contains dribble columns with zero values
            with open(team_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check for dribble columns
                self.assertIn("dribbles_detected", headers)
                self.assertIn("avg_dribble_confidence", headers)
                self.assertIn("dribble_events_count", headers)
                
                # Check data - should all be zeros
                rows = list(reader)
                for row in rows:
                    self.assertEqual(row["dribbles_detected"], "0")
                    self.assertEqual(row["avg_dribble_confidence"], "0.0")
                    self.assertEqual(row["dribble_events_count"], "0")

    def test_player_csv_integration(self):
        """Test that player CSV includes dribble statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add mock player dribble statistics
            from src.dribble_detection.dribble_event import DribbleEvent, DribbleStatistics
            
            # Create mock player statistics
            player_stats = DribbleStatistics(
                total_dribbles=3,
                successful_dribbles=3,
                events_count=3,
                avg_confidence=0.75,
                success_rate=1.0
            )
            
            # Add to analyzer
            self.dribble_analyzer.detector.player_statistics[5] = player_stats
            
            # Add some player data to other counters
            self.pass_counter.player_passes[5] = {"passes": 10, "team": 1}
            self.tackle_counter.player_tackles[5] = {"tackles": 2, "team": 1}
            
            enhanced_goal_stats = {
                "team_goals": {1: 2, 2: 1},
                "total_goals": 3,
                "goal_events": [],
                "player_goals": {}
            }
            
            final_team_goals = {1: 2, 2: 1}
            final_player_goals = {}
            
            # Test CSV export
            team_csv_path, player_csv_path = export_consolidated_goal_statistics(
                video_name="test_player_integration",
                pass_counter=self.pass_counter,
                enhanced_goal_stats=enhanced_goal_stats,
                final_team_goals=final_team_goals,
                final_player_goals=final_player_goals,
                tackle_counter=self.tackle_counter,
                output_dir=temp_dir,
                dribble_analyzer=self.dribble_analyzer
            )
            
            # Verify player CSV contains dribble data
            with open(player_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check for dribble columns
                self.assertIn("dribbles_detected", headers)
                self.assertIn("avg_dribble_confidence", headers)
                self.assertIn("dribble_events_count", headers)
                
                # Check data
                rows = list(reader)
                player5_row = next((row for row in rows if row["player_id"] == "5"), None)
                self.assertIsNotNone(player5_row)
                
                # Check dribble data for player 5
                self.assertEqual(player5_row["dribbles_detected"], "3")
                self.assertEqual(player5_row["avg_dribble_confidence"], "0.75")
                self.assertEqual(player5_row["dribble_events_count"], "3")

    def test_dribble_analyzer_methods_for_integration(self):
        """Test the methods used by the CSV export for integration."""
        # Add mock statistics
        from src.dribble_detection.dribble_event import DribbleStatistics
        
        team1_stats = DribbleStatistics(
            total_dribbles=5,
            successful_dribbles=4,
            events_count=5,
            avg_confidence=0.7,
            success_rate=0.8
        )
        
        team2_stats = DribbleStatistics(
            total_dribbles=3,
            successful_dribbles=2,
            events_count=3,
            avg_confidence=0.6,
            success_rate=0.67
        )
        
        self.dribble_analyzer.detector.team_statistics[1] = team1_stats
        self.dribble_analyzer.detector.team_statistics[2] = team2_stats
        
        # Test integration methods
        dribble_counts = self.dribble_analyzer.get_team_dribble_counts()
        dribble_confidence = self.dribble_analyzer.get_team_dribble_confidence()
        dribble_events_count = self.dribble_analyzer.get_team_dribble_events_count()
        
        # Verify results
        self.assertEqual(dribble_counts[1], 5)
        self.assertEqual(dribble_counts[2], 3)
        self.assertAlmostEqual(dribble_confidence[1], 0.7, places=2)
        self.assertAlmostEqual(dribble_confidence[2], 0.6, places=2)
        self.assertEqual(dribble_events_count[1], 5)
        self.assertEqual(dribble_events_count[2], 3)

    def test_csv_format_consistency(self):
        """Test that the CSV format is consistent with existing structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_goal_stats = {
                "team_goals": {1: 2, 2: 1},
                "total_goals": 3,
                "goal_events": []
            }
            
            final_team_goals = {1: 2, 2: 1}
            final_player_goals = {}
            
            # Test CSV export
            team_csv_path, player_csv_path = export_consolidated_goal_statistics(
                video_name="test_format",
                pass_counter=self.pass_counter,
                enhanced_goal_stats=enhanced_goal_stats,
                final_team_goals=final_team_goals,
                final_player_goals=final_player_goals,
                tackle_counter=self.tackle_counter,
                output_dir=temp_dir,
                dribble_analyzer=self.dribble_analyzer
            )
            
            # Check that the CSV has the expected structure
            with open(team_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Verify all expected columns are present
                expected_columns = [
                    "team", "passes", "goals_regular", "goals_enhanced", 
                    "goals_scoreboard", "goals_final", "tackles", "interceptions",
                    "goal_events_count", "avg_goal_confidence", "scoreboard_detected",
                    "scoreboard_confidence", "goal_detected_using_the_model",
                    "goal_detected_using_scoreboard_detection",
                    "dribbles_detected", "avg_dribble_confidence", "dribble_events_count"
                ]
                
                for column in expected_columns:
                    self.assertIn(column, headers, f"Missing column: {column}")


if __name__ == "__main__":
    print("🧪 Running Dribble Detection Integration Tests...")
    unittest.main(verbosity=2)
