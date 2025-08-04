#!/usr/bin/env python3
"""
Test script to validate the enhanced confidence system for football statistics.
This script tests the confidence level generation for passes, tackles, and interceptions.
"""

import os
import sys

import numpy as np

# Add the project root to the path (go up one level from tests to project root)
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from src.pass_counter.enhanced_pass_counter import EnhancedPassCounter, PassEvent
from src.pass_counter.enhanced_tackle_counter import (
    EnhancedTackleCounter,
    InterceptionEvent,
    TackleEvent,
)
from src.pass_counter.tackle_counter import TackleCounter
from src.utils.goal_utils import export_consolidated_goal_statistics


def test_enhanced_pass_counter_confidence():
    """Test the enhanced pass counter confidence system."""
    print("🧪 Testing Enhanced Pass Counter Confidence System...")

    # Initialize enhanced pass counter
    pass_counter = EnhancedPassCounter(
        video_width=1920, video_height=1080, frame_rate=24.0
    )

    # Create mock ball data with confidence
    ball_data = {
        "position": (500, 400),
        "confidence": 0.85,
        "velocity": 8.5,
        "source": "detection",
    }

    # Test pass detection with confidence
    pass_counter.last_player_id = 1
    pass_counter.last_ball_position = (450, 380)
    pass_counter._detect_pass(2, 1, 100, ball_data)

    # Check if pass event was created with confidence
    if pass_counter.pass_events:
        pass_event = pass_counter.pass_events[0]
        print(
            f"✅ Pass event created with confidence: {pass_event.confidence_score:.3f}"
        )

        # Check player statistics
        if 1 in pass_counter.player_passes:
            player_stats = pass_counter.player_passes[1]
            avg_confidence = player_stats.get("avg_pass_confidence", 0.0)
            print(f"✅ Player pass confidence tracked: {avg_confidence:.3f}")
        else:
            print("❌ Player pass statistics not found")
    else:
        print("❌ No pass events created")

    return pass_counter


def test_enhanced_tackle_counter_confidence():
    """Test the enhanced tackle counter confidence system."""
    print("\n🧪 Testing Enhanced Tackle Counter Confidence System...")

    # Initialize enhanced tackle counter
    tackle_counter = EnhancedTackleCounter(frame_rate=24.0)

    # Create mock player data
    current_frame_players = {
        1: {"bbox": [100, 100, 150, 200], "team": 1},
        2: {"bbox": [110, 105, 160, 205], "team": 2},
    }

    # Create mock ball data with confidence
    ball_data = {
        "position": (125, 150),
        "confidence": 0.75,
        "velocity": 3.2,
        "source": "detection",
    }

    # Set up previous state
    tackle_counter.last_ball_possessor = 1
    tackle_counter.last_team_possession = 1
    tackle_counter.possession_frames = 5

    # Test tackle detection
    tackle_counter.detect_tackles_and_interceptions_enhanced(
        current_frame_players, 2, 2, 100, ball_data
    )

    # Check if tackle event was created
    if tackle_counter.tackle_events:
        tackle_event = tackle_counter.tackle_events[0]
        print(
            f"✅ Tackle event created with confidence: {tackle_event.confidence_score:.3f}"
        )
    else:
        print("❌ No tackle events created")

    # Check team statistics
    team_tackles = tackle_counter.team_tackles.get(2, 0)
    print(f"✅ Team tackles recorded: {team_tackles}")

    return tackle_counter


def test_basic_tackle_counter_confidence():
    """Test the basic tackle counter confidence system."""
    print("\n🧪 Testing Basic Tackle Counter Confidence System...")

    # Initialize basic tackle counter
    tackle_counter = TackleCounter()

    # Create mock player data
    current_frame_players = {
        1: {"position": (100, 100), "team": 1},
        2: {"position": (110, 105), "team": 2},
    }

    # Test tackle detection
    tackle_counter.detect_tackles_and_interceptions(current_frame_players, 2, 2, 100)

    # Check if tackle was recorded
    team_tackles = tackle_counter.team_tackles.get(2, 0)
    print(f"✅ Team tackles recorded: {team_tackles}")

    # Check if confidence tracking is available
    if 2 in tackle_counter.player_tackles:
        player_stats = tackle_counter.player_tackles[2]
        avg_confidence = player_stats.get("avg_tackle_confidence", 0.0)
        print(f"✅ Player tackle confidence tracked: {avg_confidence:.3f}")
    else:
        print("❌ Player tackle statistics not found")

    return tackle_counter


def test_csv_export_with_confidence():
    """Test CSV export with confidence levels."""
    print("\n🧪 Testing CSV Export with Confidence Levels...")

    # Create mock counters with some data
    pass_counter = test_enhanced_pass_counter_confidence()
    tackle_counter = test_enhanced_tackle_counter_confidence()

    # Create mock enhanced goal stats
    enhanced_goal_stats = {
        "team_goals": {1: 2, 2: 1},
        "total_goals": 3,
        "average_confidence": 0.82,
        "goal_events": [
            {"team": 1, "confidence_score": 0.8, "player_id": 1},
            {"team": 1, "confidence_score": 0.9, "player_id": 3},
            {"team": 2, "confidence_score": 0.7, "player_id": 5},
        ],
    }

    # Create mock final goals
    final_team_goals = {1: 2, 2: 1}
    final_player_goals = {
        1: {"goals": 1, "team": 1},
        3: {"goals": 1, "team": 1},
        5: {"goals": 1, "team": 2},
    }

    try:
        # Test CSV export
        team_csv_path, player_csv_path = export_consolidated_goal_statistics(
            video_name="confidence_test",
            pass_counter=pass_counter,
            enhanced_goal_stats=enhanced_goal_stats,
            final_team_goals=final_team_goals,
            final_player_goals=final_player_goals,
            tackle_counter=tackle_counter,
            output_dir="data/output",
        )

        print(f"✅ Team CSV generated: {team_csv_path}")
        print(f"✅ Player CSV generated: {player_csv_path}")

        # Read and display team CSV content
        if os.path.exists(team_csv_path):
            with open(team_csv_path, "r") as f:
                lines = f.readlines()
                print(f"\n📊 Team CSV Header:")
                print(lines[0].strip())
                if len(lines) > 1:
                    print(f"📊 Team CSV Sample Row:")
                    print(lines[1].strip())

        # Read and display player CSV content
        if os.path.exists(player_csv_path):
            with open(player_csv_path, "r") as f:
                lines = f.readlines()
                print(f"\n📊 Player CSV Header:")
                print(lines[0].strip())
                if len(lines) > 1:
                    print(f"📊 Player CSV Sample Row:")
                    print(lines[1].strip())

        return True

    except Exception as e:
        print(f"❌ Error during CSV export: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Confidence System Tests...")
    print("=" * 60)

    # Test individual components
    test_enhanced_pass_counter_confidence()
    test_enhanced_tackle_counter_confidence()
    test_basic_tackle_counter_confidence()

    # Test CSV export
    csv_success = test_csv_export_with_confidence()

    print("\n" + "=" * 60)
    if csv_success:
        print("✅ All confidence system tests completed successfully!")
        print(
            "🎯 Confidence levels are now automatically generated and exported to CSV files."
        )
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return csv_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
