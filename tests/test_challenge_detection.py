#!/usr/bin/env python3
"""
Test script for challenge detection system.

This script validates the challenge detection functionality by:
1. Testing the ChallengeDetector class initialization
2. Testing challenge detection with mock data
3. Testing statistics calculation and export
4. Testing integration with the main pipeline
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pass_counter.challenge_detector import ChallengeDetector, ChallengeEvent, PlayerChallengeStats


class TestChallengeDetection(unittest.TestCase):
    """Test cases for challenge detection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.challenge_detector = ChallengeDetector(frame_rate=24.0)
        
        # Mock player data
        self.mock_players = {
            1: {
                "bbox": [100, 100, 150, 200],  # x1, y1, x2, y2
                "team": 1
            },
            2: {
                "bbox": [120, 110, 170, 210],  # Close to player 1
                "team": 2
            },
            3: {
                "bbox": [300, 300, 350, 400],  # Far from others
                "team": 1
            }
        }
        
        # Mock ball data
        self.mock_ball_data = {
            "position": (125, 150),  # Near players 1 and 2
            "confidence": 0.8,
            "velocity": 2.5
        }
    
    def test_challenge_detector_initialization(self):
        """Test that ChallengeDetector initializes correctly."""
        detector = ChallengeDetector(frame_rate=30.0)
        
        self.assertEqual(detector.frame_rate, 30.0)
        self.assertEqual(detector.challenge_distance_threshold, 80)
        self.assertEqual(detector.ball_proximity_threshold, 120)
        self.assertEqual(len(detector.challenge_events), 0)
        self.assertEqual(len(detector.active_challenges), 0)
        self.assertEqual(detector.team_challenges, {1: 0, 2: 0})
    
    def test_challenge_detection_basic(self):
        """Test basic challenge detection functionality."""
        # Test with player 1 having the ball and player 2 challenging
        result = self.challenge_detector.detect_challenge(
            current_frame_players=self.mock_players,
            ball_possessor=1,
            ball_team=1,
            frame_num=100,
            ball_data=self.mock_ball_data
        )
        
        # Should detect a challenge being started
        self.assertEqual(len(self.challenge_detector.active_challenges), 1)
        
        # Check that the challenge is properly recorded
        challenge_key = list(self.challenge_detector.active_challenges.keys())[0]
        self.assertEqual(challenge_key, (2, 1))  # Player 2 challenging Player 1
    
    def test_challenge_completion_success(self):
        """Test challenge completion when challenger wins the ball."""
        # Start a challenge
        self.challenge_detector.detect_challenge(
            current_frame_players=self.mock_players,
            ball_possessor=1,
            ball_team=1,
            frame_num=100,
            ball_data=self.mock_ball_data
        )
        
        # Simulate possession change to challenger
        result = self.challenge_detector.detect_challenge(
            current_frame_players=self.mock_players,
            ball_possessor=2,
            ball_team=2,
            frame_num=105,
            ball_data=self.mock_ball_data
        )
        
        # Should complete the challenge successfully
        self.assertIsInstance(result, ChallengeEvent)
        self.assertTrue(result.success)
        self.assertEqual(result.challenger_id, 2)
        self.assertEqual(result.challenged_player_id, 1)
        self.assertEqual(len(self.challenge_detector.active_challenges), 0)
        self.assertEqual(len(self.challenge_detector.challenge_events), 1)
    
    def test_challenge_completion_failure(self):
        """Test challenge completion when challenger fails to win the ball."""
        # Start a challenge
        self.challenge_detector.detect_challenge(
            current_frame_players=self.mock_players,
            ball_possessor=1,
            ball_team=1,
            frame_num=100,
            ball_data=self.mock_ball_data
        )
        
        # Simulate possession change to different player (not challenger)
        result = self.challenge_detector.detect_challenge(
            current_frame_players=self.mock_players,
            ball_possessor=3,
            ball_team=1,
            frame_num=105,
            ball_data=self.mock_ball_data
        )
        
        # Should complete the challenge as failed
        self.assertIsInstance(result, ChallengeEvent)
        self.assertFalse(result.success)
        self.assertEqual(result.challenger_id, 2)
        self.assertEqual(result.challenged_player_id, 1)
    
    def test_challenge_type_classification(self):
        """Test challenge type classification."""
        # Test ground challenge (close distance, low velocity)
        challenge_type = self.challenge_detector._classify_challenge_type(
            challenger_pos=(100, 100),
            challenged_pos=(120, 110),
            distance=25,
            challenger_velocity=1.0
        )
        self.assertEqual(challenge_type, "ground")
        
        # Test sliding challenge (close distance, high velocity)
        challenge_type = self.challenge_detector._classify_challenge_type(
            challenger_pos=(100, 100),
            challenged_pos=(120, 110),
            distance=25,
            challenger_velocity=4.0
        )
        self.assertEqual(challenge_type, "sliding")
        
        # Test pressing challenge (medium distance, medium velocity)
        challenge_type = self.challenge_detector._classify_challenge_type(
            challenger_pos=(100, 100),
            challenged_pos=(150, 130),
            distance=50,
            challenger_velocity=2.5
        )
        self.assertEqual(challenge_type, "pressing")
    
    def test_statistics_calculation(self):
        """Test challenge statistics calculation."""
        # Create some mock challenge events
        event1 = ChallengeEvent(
            frame_num=100, challenger_id=2, challenged_player_id=1,
            challenger_team=2, challenged_team=1, challenge_distance=30.0,
            ball_distance_to_challenge=25.0, challenge_type="ground",
            success=True, confidence_score=0.8, ball_velocity_before=1.0,
            ball_velocity_after=2.0, challenge_duration_frames=5
        )
        
        event2 = ChallengeEvent(
            frame_num=200, challenger_id=2, challenged_player_id=3,
            challenger_team=2, challenged_team=1, challenge_distance=45.0,
            ball_distance_to_challenge=35.0, challenge_type="pressing",
            success=False, confidence_score=0.6, ball_velocity_before=2.0,
            ball_velocity_after=1.5, challenge_duration_frames=3
        )
        
        # Record the events
        self.challenge_detector._record_challenge_event(event1)
        self.challenge_detector._record_challenge_event(event2)
        
        # Check team statistics
        team_stats = self.challenge_detector.get_team_challenge_statistics()
        self.assertEqual(team_stats[2]["challenges_attempted"], 2)
        self.assertEqual(team_stats[2]["challenges_successful"], 1)
        self.assertEqual(team_stats[2]["challenges_failed"], 1)
        self.assertEqual(team_stats[2]["challenge_success_rate"], 50.0)
        
        # Check player statistics
        player_stats = self.challenge_detector.get_player_challenge_statistics()
        self.assertIn(2, player_stats)
        player_2_stats = player_stats[2]
        self.assertEqual(player_2_stats.challenges_attempted, 2)
        self.assertEqual(player_2_stats.challenges_successful, 1)
        self.assertEqual(player_2_stats.challenges_failed, 1)
        self.assertEqual(player_2_stats.challenge_success_rate, 50.0)
        self.assertEqual(player_2_stats.ground_challenges, 1)
        self.assertEqual(player_2_stats.pressing_challenges, 1)
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        # Create a test event
        event = ChallengeEvent(
            frame_num=100, challenger_id=2, challenged_player_id=1,
            challenger_team=2, challenged_team=1, challenge_distance=30.0,
            ball_distance_to_challenge=25.0, challenge_type="ground",
            success=True, confidence_score=0.8, ball_velocity_before=1.0,
            ball_velocity_after=2.0, challenge_duration_frames=5
        )
        
        self.challenge_detector._record_challenge_event(event)
        
        # Test CSV export
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            events_path, player_stats_path = self.challenge_detector.export_challenge_statistics_to_csv(output_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(events_path))
            self.assertTrue(os.path.exists(player_stats_path))
            
            # Check file contents
            with open(events_path, 'r') as f:
                content = f.read()
                self.assertIn("frame_num", content)
                self.assertIn("challenger_id", content)
                self.assertIn("100", content)  # frame number
                self.assertIn("2", content)    # challenger id
            
            with open(player_stats_path, 'r') as f:
                content = f.read()
                self.assertIn("player_id", content)
                self.assertIn("challenges_attempted", content)
                self.assertIn("2", content)  # player id
                self.assertIn("1", content)  # challenges attempted
        
        finally:
            # Clean up test files
            if os.path.exists(events_path):
                os.remove(events_path)
            if os.path.exists(player_stats_path):
                os.remove(player_stats_path)
            if os.path.exists(output_dir):
                os.rmdir(output_dir)


def run_challenge_detection_tests():
    """Run all challenge detection tests."""
    print("🥊 Running Challenge Detection Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChallengeDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("✅ All challenge detection tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_challenge_detection_tests()
    sys.exit(0 if success else 1)
