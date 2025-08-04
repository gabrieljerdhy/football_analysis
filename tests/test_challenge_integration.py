#!/usr/bin/env python3
"""
Integration test for challenge detection with the main pipeline.

This script tests the integration of challenge detection with:
1. Main pipeline imports
2. CSV export integration
3. Statistics calculation
"""

import os
import sys
import tempfile
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_main_pipeline_imports():
    """Test that main.py can import the challenge detector without errors."""
    print("🔍 Testing main pipeline imports...")
    
    try:
        # Test importing the challenge detector
        from src.pass_counter.challenge_detector import ChallengeDetector
        print("✅ ChallengeDetector import successful")
        
        # Test importing main.py (this will test all imports)
        import main
        print("✅ Main pipeline imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_csv_export_integration():
    """Test that the CSV export system includes challenge fields."""
    print("🔍 Testing CSV export integration...")
    
    try:
        from src.utils.goal_utils import export_consolidated_goal_statistics
        from src.pass_counter.challenge_detector import ChallengeDetector
        from src.pass_counter.pass_counter import PassCounter
        from src.pass_counter.tackle_counter import TackleCounter
        
        # Create temporary directory for test output
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create mock objects
            pass_counter = PassCounter()
            tackle_counter = TackleCounter()
            challenge_detector = ChallengeDetector()
            
            # Add some mock data
            pass_counter.team_passes = {1: 10, 2: 8}
            tackle_counter.team_tackles = {1: 3, 2: 5}
            
            # Add mock challenge data
            challenge_detector.team_challenges = {1: 4, 2: 6}
            challenge_detector.team_successful_challenges = {1: 2, 2: 4}
            
            # Test export function
            team_csv, player_csv = export_consolidated_goal_statistics(
                video_name="test_video",
                pass_counter=pass_counter,
                enhanced_goal_stats={},
                final_team_goals={1: 1, 2: 0},
                final_player_goals={},
                tackle_counter=tackle_counter,
                output_dir=temp_dir,
                challenge_detector=challenge_detector
            )
            
            # Check that files were created
            if not os.path.exists(team_csv):
                print(f"❌ Team CSV not created: {team_csv}")
                return False
            
            if not os.path.exists(player_csv):
                print(f"❌ Player CSV not created: {player_csv}")
                return False
            
            # Check team CSV content
            with open(team_csv, 'r') as f:
                team_content = f.read()
                if "challenges_attempted" not in team_content:
                    print("❌ Team CSV missing challenges_attempted column")
                    return False
                if "challenges_successful" not in team_content:
                    print("❌ Team CSV missing challenges_successful column")
                    return False
                if "challenges_failed" not in team_content:
                    print("❌ Team CSV missing challenges_failed column")
                    return False
                if "challenge_success_rate" not in team_content:
                    print("❌ Team CSV missing challenge_success_rate column")
                    return False
            
            # Check player CSV content
            with open(player_csv, 'r') as f:
                player_content = f.read()
                if "challenges_attempted" not in player_content:
                    print("❌ Player CSV missing challenges_attempted column")
                    return False
                if "challenges_successful" not in player_content:
                    print("❌ Player CSV missing challenges_successful column")
                    return False
                if "challenges_failed" not in player_content:
                    print("❌ Player CSV missing challenges_failed column")
                    return False
                if "challenge_success_rate" not in player_content:
                    print("❌ Player CSV missing challenge_success_rate column")
                    return False
            
            print("✅ CSV export integration successful")
            print(f"   Team CSV: {team_csv}")
            print(f"   Player CSV: {player_csv}")
            
            return True
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"❌ CSV export integration error: {e}")
        return False

def test_challenge_detector_functionality():
    """Test basic challenge detector functionality."""
    print("🔍 Testing challenge detector functionality...")
    
    try:
        from src.pass_counter.challenge_detector import ChallengeDetector
        
        # Create detector
        detector = ChallengeDetector(frame_rate=24.0)
        
        # Test basic functionality
        mock_players = {
            1: {"bbox": [100, 100, 150, 200], "team": 1},
            2: {"bbox": [120, 110, 170, 210], "team": 2}
        }
        
        mock_ball_data = {
            "position": (125, 150),
            "confidence": 0.8,
            "velocity": 2.5
        }
        
        # Test challenge detection
        result = detector.detect_challenge(
            current_frame_players=mock_players,
            ball_possessor=1,
            ball_team=1,
            frame_num=100,
            ball_data=mock_ball_data
        )
        
        # Should have started a challenge
        if len(detector.active_challenges) != 1:
            print(f"❌ Expected 1 active challenge, got {len(detector.active_challenges)}")
            return False
        
        # Test statistics
        team_stats = detector.get_team_challenge_statistics()
        if not isinstance(team_stats, dict):
            print("❌ Team statistics should be a dictionary")
            return False
        
        player_stats = detector.get_player_challenge_statistics()
        if not isinstance(player_stats, dict):
            print("❌ Player statistics should be a dictionary")
            return False
        
        print("✅ Challenge detector functionality test successful")
        return True
        
    except Exception as e:
        print(f"❌ Challenge detector functionality error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("🥊 Running Challenge Detection Integration Tests...")
    print("=" * 60)
    
    tests = [
        test_main_pipeline_imports,
        test_csv_export_integration,
        test_challenge_detector_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All integration tests passed!")
        return True
    else:
        print(f"❌ {total - passed} integration test(s) failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
