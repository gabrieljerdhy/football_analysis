#!/usr/bin/env python3
"""
Test script for the enhanced goal detection and counting system.
This script validates the improvements made to goal detection, CSV export, and data integration.
"""

import os
import sys
import csv
from pathlib import Path

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_detection import FieldKeypointsDetector, GoalDetector
from pass_counter.pass_counter import PassCounter
from pass_counter.tackle_counter import TackleCounter
from utils.goal_utils import calculate_final_goal_stats, export_consolidated_goal_statistics


def test_enhanced_goal_detector():
    """Test the enhanced GoalDetector functionality."""
    print("🧪 Testing Enhanced GoalDetector...")
    
    # Initialize detector
    keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
    goal_detector = GoalDetector(keypoints_detector)
    
    # Test initialization
    assert goal_detector.final_team_goals == {1: 0, 2: 0}
    assert goal_detector.final_player_goals == {}
    assert goal_detector.goal_validation_history == []
    assert goal_detector.goal_confidence_scores == []
    print("✓ Enhanced initialization successful")
    
    # Test goal detection with trajectory validation
    test_positions = [
        (50, 400),   # Left goal area
        (45, 400),   # Moving towards left goal
        (40, 400),   # Continuing towards left goal
        (35, 400),   # Final position in left goal
    ]
    
    for i, pos in enumerate(test_positions):
        goal_event = goal_detector.detect_goal(pos, 5, 2, i * 10)
        if goal_event:
            print(f"✓ Goal detected with confidence: {goal_event.get('confidence_score', 0):.2f}")
            break
    
    # Test statistics
    stats = goal_detector.get_goal_statistics()
    assert "final_team_goals" in stats
    assert "final_player_goals" in stats
    assert "average_confidence" in stats
    assert "detection_accuracy" in stats
    print("✓ Enhanced statistics retrieval successful")
    
    # Test final goal count setting
    goal_detector.set_final_goal_counts({1: 2, 2: 1}, {5: {"goals": 1, "team": 2}})
    assert goal_detector.final_team_goals[1] == 2
    assert goal_detector.final_team_goals[2] == 1
    print("✓ Final goal count setting successful")
    
    print("✅ Enhanced GoalDetector tests passed!\n")
    return True


def test_trajectory_validation():
    """Test the enhanced trajectory validation system."""
    print("🧪 Testing Trajectory Validation...")
    
    keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
    goal_detector = GoalDetector(keypoints_detector)
    
    # Test direction consistency
    goal_detector.ball_trajectory = [(100, 400), (90, 400), (80, 400), (70, 400)]
    direction_score = goal_detector._calculate_direction_consistency("left")
    assert direction_score > 0.5, f"Direction score too low: {direction_score}"
    print(f"✓ Direction consistency: {direction_score:.2f}")
    
    # Test speed consistency
    speed_score = goal_detector._calculate_speed_consistency()
    assert speed_score >= 0, f"Speed score invalid: {speed_score}"
    print(f"✓ Speed consistency: {speed_score:.2f}")
    
    # Test trajectory smoothness
    smoothness_score = goal_detector._calculate_trajectory_smoothness()
    assert smoothness_score >= 0, f"Smoothness score invalid: {smoothness_score}"
    print(f"✓ Trajectory smoothness: {smoothness_score:.2f}")
    
    # Test goal approach angle
    angle_score = goal_detector._calculate_goal_approach_angle("left")
    assert 0 <= angle_score <= 1, f"Angle score out of range: {angle_score}"
    print(f"✓ Goal approach angle: {angle_score:.2f}")
    
    print("✅ Trajectory validation tests passed!\n")
    return True


def test_consolidated_csv_export():
    """Test the consolidated CSV export functionality."""
    print("🧪 Testing Consolidated CSV Export...")
    
    # Create mock data
    pass_counter = PassCounter()
    pass_counter.team_passes = {1: 15, 2: 12}
    pass_counter.player_passes = {5: {"passes": 8, "team": 1}, 7: {"passes": 6, "team": 2}}
    pass_counter.team_goals = {1: 1, 2: 0}
    pass_counter.player_goals = {5: {"goals": 1, "team": 1}}
    
    tackle_counter = TackleCounter()
    tackle_counter.team_tackles = {1: 3, 2: 5}
    tackle_counter.player_tackles = {5: {"tackles": 2, "team": 1}}
    tackle_counter.team_interceptions = {1: 1, 2: 2}
    tackle_counter.player_interceptions = {7: {"interceptions": 1, "team": 2}}
    
    enhanced_goal_stats = {
        "team_goals": {1: 1, 2: 1},
        "player_goals": {5: {"goals": 1, "team": 1}, 7: {"goals": 1, "team": 2}},
        "final_team_goals": {1: 1, 2: 1},
        "final_player_goals": {5: {"goals": 1, "team": 1}, 7: {"goals": 1, "team": 2}},
        "goal_events": [
            {"team": 1, "player_id": 5, "confidence_score": 0.85},
            {"team": 2, "player_id": 7, "confidence_score": 0.72}
        ],
        "average_confidence": 0.785,
        "detection_accuracy": 1.0
    }
    
    final_team_goals = {1: 1, 2: 1}
    final_player_goals = {5: {"goals": 1, "team": 1}, 7: {"goals": 1, "team": 2}}
    
    # Test export
    output_dir = "debug_and_tests/test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    team_csv_path, player_csv_path = export_consolidated_goal_statistics(
        "test_video",
        pass_counter,
        enhanced_goal_stats,
        final_team_goals,
        final_player_goals,
        tackle_counter,
        output_dir
    )
    
    # Verify files were created
    assert os.path.exists(team_csv_path), f"Team CSV not created: {team_csv_path}"
    assert os.path.exists(player_csv_path), f"Player CSV not created: {player_csv_path}"
    print("✓ CSV files created successfully")
    
    # Verify team CSV content
    with open(team_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        team_rows = list(reader)
        assert len(team_rows) == 2, f"Expected 2 team rows, got {len(team_rows)}"
        
        team1_row = next(row for row in team_rows if row['team'] == '1')
        assert team1_row['goals_final'] == '1', f"Team 1 final goals incorrect: {team1_row['goals_final']}"
        assert team1_row['passes'] == '15', f"Team 1 passes incorrect: {team1_row['passes']}"
        print("✓ Team CSV content verified")
    
    # Verify player CSV content
    with open(player_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        player_rows = list(reader)
        assert len(player_rows) >= 2, f"Expected at least 2 player rows, got {len(player_rows)}"
        
        player5_row = next(row for row in player_rows if row['player_id'] == '5')
        assert player5_row['goals_final'] == '1', f"Player 5 final goals incorrect: {player5_row['goals_final']}"
        assert player5_row['team'] == '1', f"Player 5 team incorrect: {player5_row['team']}"
        print("✓ Player CSV content verified")
    
    # Cleanup
    os.remove(team_csv_path)
    os.remove(player_csv_path)
    os.rmdir(output_dir)
    
    print("✅ Consolidated CSV export tests passed!\n")
    return True


def test_goal_priority_system():
    """Test the goal priority system (manual > enhanced > regular)."""
    print("🧪 Testing Goal Priority System...")
    
    # Create mock components
    pass_counter = PassCounter()
    pass_counter.team_goals = {1: 1, 2: 0}  # Regular detection
    pass_counter.player_goals = {5: {"goals": 1, "team": 1}}
    
    enhanced_goal_stats = {
        "team_goals": {1: 2, 2: 1},  # Enhanced detection
        "player_goals": {5: {"goals": 1, "team": 1}, 7: {"goals": 1, "team": 2}},
        "final_team_goals": {1: 2, 2: 1},
        "final_player_goals": {5: {"goals": 1, "team": 1}, 7: {"goals": 1, "team": 2}},
        "final_total_goals": 3
    }
    
    # Test 1: No manual goals - should use enhanced
    final_team, final_player = calculate_final_goal_stats(
        pass_counter, enhanced_goal_stats, None
    )
    assert final_team[1] == 2, f"Expected enhanced goals (2), got {final_team[1]}"
    assert final_team[2] == 1, f"Expected enhanced goals (1), got {final_team[2]}"
    print("✓ Enhanced goals used when no manual goals")
    
    # Test 2: Manual goals provided - should override enhanced
    manual_goals = [
        {"team": 1, "player_id": 5, "frame_num": 1000},
        {"team": 1, "player_id": 3, "frame_num": 2000},
        {"team": 2, "player_id": 8, "frame_num": 3000}
    ]
    
    final_team, final_player = calculate_final_goal_stats(
        pass_counter, enhanced_goal_stats, manual_goals
    )
    assert final_team[1] == 2, f"Expected manual goals (2), got {final_team[1]}"
    assert final_team[2] == 1, f"Expected manual goals (1), got {final_team[2]}"
    assert len(final_player) == 3, f"Expected 3 players, got {len(final_player)}"
    print("✓ Manual goals override enhanced goals")
    
    print("✅ Goal priority system tests passed!\n")
    return True


def run_all_tests():
    """Run all enhanced goal detection tests."""
    print("🚀 Running Enhanced Goal Detection System Tests...\n")
    
    tests = [
        test_enhanced_goal_detector,
        test_trajectory_validation,
        test_consolidated_csv_export,
        test_goal_priority_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Enhanced goal detection system is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
