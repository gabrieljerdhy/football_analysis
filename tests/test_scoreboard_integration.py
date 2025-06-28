"""
Integration tests for scoreboard detection with the main football analysis pipeline.

This module tests the integration of scoreboard detection with the existing
goal detection and analysis systems.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoreboard_detection import ScoreboardAnalyzer
from utils.goal_utils import calculate_final_goal_stats, export_consolidated_goal_statistics


class TestScoreboardIntegration(unittest.TestCase):
    """Integration tests for scoreboard detection with the main pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scoreboard_analyzer = ScoreboardAnalyzer()
        
        # Mock pass counter
        self.mock_pass_counter = Mock()
        self.mock_pass_counter.team_goals = {1: 1, 2: 2}
        self.mock_pass_counter.player_goals = {
            5: {'goals': 1, 'team': 1},
            10: {'goals': 2, 'team': 2}
        }
        self.mock_pass_counter.team_passes = {1: 50, 2: 45}
        self.mock_pass_counter.player_passes = {
            5: {'passes': 25, 'team': 1},
            10: {'passes': 20, 'team': 2}
        }
        
        # Mock enhanced goal stats
        self.mock_enhanced_stats = {
            'final_total_goals': 3,
            'total_goals': 3,
            'team_goals': {1: 1, 2: 2},
            'player_goals': {
                5: {'goals': 1, 'team': 1},
                10: {'goals': 2, 'team': 2}
            },
            'goal_events': [
                {'team': 1, 'player_id': 5, 'confidence_score': 0.8},
                {'team': 2, 'player_id': 10, 'confidence_score': 0.9},
                {'team': 2, 'player_id': 10, 'confidence_score': 0.7}
            ],
            'final_team_goals': {1: 1, 2: 2},
            'final_player_goals': {
                5: {'goals': 1, 'team': 1},
                10: {'goals': 2, 'team': 2}
            }
        }
        
        # Mock tackle counter
        self.mock_tackle_counter = Mock()
        self.mock_tackle_counter.team_tackles = {1: 5, 2: 7}
        self.mock_tackle_counter.team_interceptions = {1: 3, 2: 4}
        self.mock_tackle_counter.player_tackles = {
            5: {'tackles': 2, 'team': 1},
            10: {'tackles': 3, 'team': 2}
        }
        self.mock_tackle_counter.player_interceptions = {
            5: {'interceptions': 1, 'team': 1},
            10: {'interceptions': 2, 'team': 2}
        }
    
    def test_calculate_final_goal_stats_no_scoreboard(self):
        """Test goal stats calculation without scoreboard detection."""
        final_team_goals, final_player_goals = calculate_final_goal_stats(
            self.mock_pass_counter,
            self.mock_enhanced_stats,
            None,  # No manual goals
            None   # No scoreboard analyzer
        )
        
        # Should use enhanced stats
        self.assertEqual(final_team_goals, {1: 1, 2: 2})
        self.assertEqual(len(final_player_goals), 2)
    
    def test_calculate_final_goal_stats_with_scoreboard_low_confidence(self):
        """Test goal stats calculation with low-confidence scoreboard detection."""
        # Mock scoreboard with low confidence
        self.scoreboard_analyzer.scoreboard_detected = True
        self.scoreboard_analyzer.current_score = {
            'team1_score': 3,
            'team2_score': 1,
            'confidence': 0.5  # Low confidence
        }
        
        final_team_goals, final_player_goals = calculate_final_goal_stats(
            self.mock_pass_counter,
            self.mock_enhanced_stats,
            None,  # No manual goals
            self.scoreboard_analyzer
        )
        
        # Should fall back to enhanced stats due to low confidence
        self.assertEqual(final_team_goals, {1: 1, 2: 2})
    
    def test_calculate_final_goal_stats_with_scoreboard_high_confidence(self):
        """Test goal stats calculation with high-confidence scoreboard detection."""
        # Mock scoreboard with high confidence
        self.scoreboard_analyzer.scoreboard_detected = True
        self.scoreboard_analyzer.current_score = {
            'team1_score': 3,
            'team2_score': 1,
            'confidence': 0.8,  # High confidence
            'detection_rate': 0.9
        }
        
        final_team_goals, final_player_goals = calculate_final_goal_stats(
            self.mock_pass_counter,
            self.mock_enhanced_stats,
            None,  # No manual goals
            self.scoreboard_analyzer
        )
        
        # Should use scoreboard stats
        self.assertEqual(final_team_goals, {1: 3, 2: 1})
        # Player goals should be empty since scoreboard doesn't provide player info
        self.assertEqual(final_player_goals, {})
    
    def test_calculate_final_goal_stats_priority_manual_over_scoreboard(self):
        """Test that manual goals take priority over scoreboard detection."""
        # Mock high-confidence scoreboard
        self.scoreboard_analyzer.scoreboard_detected = True
        self.scoreboard_analyzer.current_score = {
            'team1_score': 3,
            'team2_score': 1,
            'confidence': 0.9
        }
        
        # Manual goals
        manual_goals = [
            {'team': 1, 'player_id': 5, 'frame_num': 1000},
            {'team': 1, 'player_id': 7, 'frame_num': 2000},
            {'team': 2, 'player_id': 10, 'frame_num': 3000},
        ]
        
        final_team_goals, final_player_goals = calculate_final_goal_stats(
            self.mock_pass_counter,
            self.mock_enhanced_stats,
            manual_goals,  # Manual goals provided
            self.scoreboard_analyzer
        )
        
        # Should use manual goals, not scoreboard
        self.assertEqual(final_team_goals, {1: 2, 2: 1})
        self.assertEqual(len(final_player_goals), 2)
        self.assertEqual(final_player_goals[5]['goals'], 1)
        self.assertEqual(final_player_goals[7]['goals'], 1)
        self.assertEqual(final_player_goals[10]['goals'], 1)
    
    @patch('builtins.open', create=True)
    @patch('csv.DictWriter')
    @patch('os.makedirs')
    def test_export_consolidated_goal_statistics_with_scoreboard(self, mock_makedirs, mock_csv_writer, mock_open):
        """Test CSV export with scoreboard information."""
        # Mock scoreboard analyzer with detection
        self.scoreboard_analyzer.scoreboard_detected = True
        self.scoreboard_analyzer.current_score = {
            'team1_score': 2,
            'team2_score': 1,
            'confidence': 0.85,
            'detection_rate': 0.8
        }
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_writer_instance = MagicMock()
        mock_csv_writer.return_value = mock_writer_instance
        
        final_team_goals = {1: 2, 2: 1}
        final_player_goals = {}
        
        # Call the function
        team_csv, player_csv = export_consolidated_goal_statistics(
            "test_video",
            self.mock_pass_counter,
            self.mock_enhanced_stats,
            final_team_goals,
            final_player_goals,
            self.mock_tackle_counter,
            self.scoreboard_analyzer
        )
        
        # Verify that CSV writer was called
        self.assertTrue(mock_csv_writer.called)
        self.assertTrue(mock_writer_instance.writeheader.called)
        self.assertTrue(mock_writer_instance.writerow.called)
        
        # Check that the correct file paths are returned
        self.assertIn("test_video_team_stats.csv", team_csv)
        self.assertIn("test_video_player_stats.csv", player_csv)
        
        # Verify that scoreboard data is included in the CSV rows
        writerow_calls = mock_writer_instance.writerow.call_args_list
        
        # Check that scoreboard fields are present in at least one call
        scoreboard_fields_found = False
        for call in writerow_calls:
            row_data = call[0][0]  # First argument of the call
            if 'scoreboard_detected' in row_data and 'scoreboard_confidence' in row_data:
                scoreboard_fields_found = True
                break
        
        self.assertTrue(scoreboard_fields_found, "Scoreboard fields not found in CSV output")
    
    def test_scoreboard_analyzer_frame_processing_integration(self):
        """Test that scoreboard analyzer integrates properly with frame processing."""
        # Create a dummy frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Process multiple frames
        results = []
        for frame_num in range(0, 100, 10):  # Every 10th frame
            result = self.scoreboard_analyzer.analyze_frame(frame, frame_num)
            results.append(result)
        
        # Verify that frames were processed
        self.assertGreater(self.scoreboard_analyzer.frames_processed, 0)
        
        # Verify that the analyzer maintains state
        stats = self.scoreboard_analyzer.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('frames_processed', stats)
        self.assertIn('total_detections', stats)
    
    def test_scoreboard_analyzer_batch_processing(self):
        """Test batch processing of frames."""
        # Create dummy frames
        frames = [np.ones((480, 640, 3), dtype=np.uint8) * 128 for _ in range(10)]
        
        # Process batch
        results = self.scoreboard_analyzer.analyze_video_batch(frames, start_frame_number=0)
        
        # Verify results
        self.assertEqual(len(results), 10)
        self.assertGreater(self.scoreboard_analyzer.frames_processed, 0)
    
    def test_scoreboard_analyzer_reset_functionality(self):
        """Test that analyzer reset works properly in integration context."""
        # Process some frames
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.scoreboard_analyzer.analyze_frame(frame, 0)
        
        # Verify state exists
        self.assertGreater(self.scoreboard_analyzer.frames_processed, 0)
        
        # Reset
        self.scoreboard_analyzer.reset()
        
        # Verify reset
        self.assertEqual(self.scoreboard_analyzer.frames_processed, 0)
        self.assertFalse(self.scoreboard_analyzer.scoreboard_detected)
        self.assertEqual(len(self.scoreboard_analyzer.score_history), 0)


class TestScoreboardPipelineIntegration(unittest.TestCase):
    """Test scoreboard detection integration with the complete pipeline."""
    
    def test_scoreboard_detection_parameter_flow(self):
        """Test that scoreboard detection parameters flow correctly through the pipeline."""
        # This test verifies that the enable_scoreboard_detection parameter
        # is properly passed through the main function calls
        
        # Mock the main components
        with patch('src.scoreboard_detection.ScoreboardAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            
            # Import and test the parameter flow
            from utils.goal_utils import calculate_final_goal_stats
            
            # Test with scoreboard analyzer
            mock_pass_counter = Mock()
            mock_pass_counter.team_goals = {1: 0, 2: 0}
            mock_pass_counter.player_goals = {}
            
            result = calculate_final_goal_stats(
                mock_pass_counter,
                None,  # No enhanced stats
                None,  # No manual goals
                mock_analyzer  # Scoreboard analyzer provided
            )
            
            # Should return valid results
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
