#!/usr/bin/env python3
"""
Integration test for jersey number CSV output.

This script tests that the CSV output correctly includes jersey numbers
and validates the format matches the expected schema.
"""

import os
import sys
import tempfile
import csv
from unittest.mock import Mock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.goal_utils import export_consolidated_goal_statistics


def test_csv_jersey_integration():
    """Test that CSV export correctly includes jersey numbers."""
    print("🧪 Testing CSV Jersey Number Integration...")
    
    # Create mock data
    mock_pass_counter = Mock()
    mock_pass_counter.player_passes = {
        1: {"team": 1, "passes": 5},
        2: {"team": 1, "passes": 3},
        3: {"team": 2, "passes": 7}
    }
    mock_pass_counter.player_goals = {
        1: {"team": 1, "goals": 1},
        3: {"team": 2, "goals": 2}
    }
    mock_pass_counter.team_passes = {1: 8, 2: 7}
    mock_pass_counter.team_goals = {1: 1, 2: 2}
    
    mock_tackle_counter = Mock()
    mock_tackle_counter.player_tackles = {
        1: {"team": 1, "tackles": 3},
        2: {"team": 1, "tackles": 1},
        3: {"team": 2, "tackles": 4}
    }
    mock_tackle_counter.player_interceptions = {
        1: {"team": 1, "interceptions": 2},
        3: {"team": 2, "interceptions": 1}
    }
    mock_tackle_counter.team_tackles = {1: 4, 2: 4}
    mock_tackle_counter.team_interceptions = {1: 2, 2: 1}
    
    # Create mock tracker with jersey detection
    mock_tracker = Mock()
    mock_tracker.jersey_detector = Mock()
    
    # Mock jersey number mappings: track_id -> jersey_number
    jersey_mappings = {1: 10, 2: 7, 3: 99}
    mock_tracker.jersey_detector.get_jersey_number_for_player.side_effect = lambda player_id: jersey_mappings.get(player_id)
    
    enhanced_goal_stats = {
        "team_goals": {1: 1, 2: 2}, 
        "goal_events": [
            {"team": 1, "player_id": 1, "confidence_score": 0.8},
            {"team": 2, "player_id": 3, "confidence_score": 0.9},
            {"team": 2, "player_id": 3, "confidence_score": 0.85}
        ]
    }
    final_team_goals = {1: 1, 2: 2}
    final_player_goals = {1: {"goals": 1, "team": 1}, 3: {"goals": 2, "team": 2}}
    
    # Test CSV export
    with tempfile.TemporaryDirectory() as temp_dir:
        team_csv, player_csv = export_consolidated_goal_statistics(
            "test_integration",
            mock_pass_counter,
            enhanced_goal_stats,
            final_team_goals,
            final_player_goals,
            mock_tackle_counter,
            output_dir=temp_dir,
            tracker=mock_tracker
        )
        
        # Verify files exist
        assert os.path.exists(team_csv), "Team CSV should be created"
        assert os.path.exists(player_csv), "Player CSV should be created"
        
        # Read and validate player CSV
        with open(player_csv, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Check headers
            expected_headers = [
                "player_id", "jersey_number", "team", "passes", "goals_regular",
                "goals_enhanced", "goals_final", "tackles", "interceptions",
                "goal_events_count", "avg_goal_confidence", "scoreboard_detected",
                "scoreboard_confidence"
            ]
            
            for header in expected_headers:
                assert header in headers, f"Header '{header}' should be in CSV"
            
            # Check data
            rows = list(reader)
            assert len(rows) == 3, f"Should have 3 player rows, got {len(rows)}"
            
            # Validate jersey numbers
            for row in rows:
                player_id = int(row["player_id"])
                jersey_number = int(row["jersey_number"])
                expected_jersey = jersey_mappings.get(player_id, player_id)
                
                assert jersey_number == expected_jersey, \
                    f"Player {player_id} should have jersey {expected_jersey}, got {jersey_number}"
                
                # Validate jersey number is in valid range
                assert 0 <= jersey_number <= 99, \
                    f"Jersey number {jersey_number} should be in range 0-99"
            
            print("✅ Player CSV format validation passed")
            
            # Print sample data for verification
            print("\n📊 Sample Player CSV Data:")
            print(f"Headers: {', '.join(headers)}")
            for i, row in enumerate(rows[:2]):  # Show first 2 rows
                print(f"Row {i+1}: Player {row['player_id']}, Jersey {row['jersey_number']}, Team {row['team']}")
        
        # Read and validate team CSV
        with open(team_csv, 'r') as f:
            reader = csv.DictReader(f)
            team_headers = reader.fieldnames
            team_rows = list(reader)
            
            assert len(team_rows) == 2, f"Should have 2 team rows, got {len(team_rows)}"
            print("✅ Team CSV format validation passed")
            
            print("\n📊 Sample Team CSV Data:")
            print(f"Headers: {', '.join(team_headers[:5])}...")  # Show first 5 headers
            for row in team_rows:
                print(f"Team {row['team']}: {row['passes']} passes, {row['goals_final']} goals")
    
    return True


def test_fallback_jersey_numbers():
    """Test that fallback jersey numbers work when no tracker is provided."""
    print("\n🧪 Testing Fallback Jersey Numbers...")
    
    # Create minimal mock data
    mock_pass_counter = Mock()
    mock_pass_counter.player_passes = {100: {"team": 1, "passes": 1}}
    mock_pass_counter.player_goals = {}
    mock_pass_counter.team_passes = {1: 1, 2: 0}
    mock_pass_counter.team_goals = {1: 0, 2: 0}
    
    mock_tackle_counter = Mock()
    mock_tackle_counter.player_tackles = {}
    mock_tackle_counter.player_interceptions = {}
    mock_tackle_counter.team_tackles = {1: 0, 2: 0}
    mock_tackle_counter.team_interceptions = {1: 0, 2: 0}
    
    enhanced_goal_stats = {"team_goals": {1: 0, 2: 0}, "goal_events": []}
    final_team_goals = {1: 0, 2: 0}
    final_player_goals = {}
    
    # Test without tracker (should use player_id as jersey_number)
    with tempfile.TemporaryDirectory() as temp_dir:
        team_csv, player_csv = export_consolidated_goal_statistics(
            "test_fallback",
            mock_pass_counter,
            enhanced_goal_stats,
            final_team_goals,
            final_player_goals,
            mock_tackle_counter,
            output_dir=temp_dir,
            tracker=None  # No tracker provided
        )
        
        # Read player CSV
        with open(player_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 1, "Should have 1 player row"
            row = rows[0]
            
            # Should use player_id as jersey_number when no tracker
            assert row["player_id"] == "100", "Player ID should be 100"
            assert row["jersey_number"] == "100", "Jersey number should fallback to player_id (100)"
            
            print("✅ Fallback jersey number validation passed")
            print(f"   Player {row['player_id']} uses fallback jersey {row['jersey_number']}")
    
    return True


if __name__ == "__main__":
    print("🚀 Running CSV Jersey Integration Tests...")
    
    try:
        success1 = test_csv_jersey_integration()
        success2 = test_fallback_jersey_numbers()
        
        if success1 and success2:
            print("\n✅ All CSV integration tests passed!")
            print("🎯 Jersey number detection is properly integrated into CSV output")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
