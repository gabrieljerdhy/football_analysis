#!/usr/bin/env python3
"""
Validate Dribble Detection System

This script provides a simple validation of the dribble detection system
with sample data to ensure it works correctly and produces expected results.
"""

import os
import sys
import tempfile

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dribble_detection import DribbleAnalyzer
from src.dribble_detection.dribble_event import DribbleEvent


def create_sample_tracking_data():
    """Create sample tracking data for testing."""
    # Simulate a player dribbling past an opponent
    tracks = {"players": [], "ball": []}

    # Create 30 frames of data
    for frame_num in range(30):
        # Player 1 (dribbling player) moves forward with direction changes
        if frame_num < 10:
            # Initial movement
            player1_x = 100 + frame_num * 3
            player1_y = 200
        elif frame_num < 20:
            # Direction change (dribble move)
            player1_x = 130 + (frame_num - 10) * 2
            player1_y = 200 + (frame_num - 10) * 2
        else:
            # Continue forward after dribble
            player1_x = 150 + (frame_num - 20) * 4
            player1_y = 220 + (frame_num - 20) * 1

        # Player 2 (opponent) stays relatively stationary
        player2_x = 140 + frame_num * 0.5  # Slight movement
        player2_y = 210

        # Ball follows player 1 closely
        ball_x = player1_x + 5
        ball_y = player1_y + 10

        # Create frame data
        players_frame = {
            1: {
                "bbox": [player1_x, player1_y, player1_x + 50, player1_y + 100],
                "team": 1,
            },
            2: {
                "bbox": [player2_x, player2_y, player2_x + 50, player2_y + 100],
                "team": 2,
            },
        }

        ball_frame = {1: {"bbox": [ball_x, ball_y, ball_x + 10, ball_y + 10]}}

        tracks["players"].append(players_frame)
        tracks["ball"].append(ball_frame)

    # Player 1 has ball possession throughout
    player_assignments = [1] * 30

    return tracks, player_assignments


def validate_basic_functionality():
    """Validate basic dribble detection functionality."""
    print("🧪 Testing basic dribble detection functionality...")

    analyzer = DribbleAnalyzer(
        min_dribble_distance=20.0,  # Lower threshold for test
        min_dribble_duration=8,  # Lower threshold for test
        confidence_threshold=0.3,  # Lower threshold for test
    )

    # Adjust detector thresholds for testing
    analyzer.detector.opponent_proximity_threshold = 100.0  # More lenient for testing

    # Test frame-by-frame analysis
    players = {
        1: {"bbox": [100, 100, 150, 200], "team": 1},
        2: {"bbox": [160, 100, 210, 200], "team": 2},  # Close opponent
    }

    dribble_detected = False

    # Simulate movement over multiple frames
    for frame_num in range(1, 25):
        # Move player 1 with direction changes
        if frame_num < 10:
            x_offset = frame_num * 5
            y_offset = 0
        else:
            x_offset = 50 + (frame_num - 10) * 3
            y_offset = (frame_num - 10) * 2

        players[1]["bbox"] = [
            100 + x_offset,
            100 + y_offset,
            150 + x_offset,
            200 + y_offset,
        ]
        ball_position = (125 + x_offset, 150 + y_offset)

        result = analyzer.analyze_frame(
            frame_num=frame_num,
            players=players,
            ball_position=ball_position,
            ball_possessor=1,
            ball_team=1,
        )

        if result:
            print(f"✅ Dribble detected at frame {frame_num}:")
            print(f"   Player: {result.player_id}, Team: {result.team}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Duration: {result.duration_frames} frames")
            print(f"   Distance: {result.distance_covered:.1f} pixels")
            print(f"   Direction changes: {result.direction_changes}")
            dribble_detected = True
            break

    if not dribble_detected:
        print("⚠️  No dribble detected in basic test")

    return dribble_detected


def validate_complete_match_analysis():
    """Validate complete match analysis functionality."""
    print("\n🧪 Testing complete match analysis...")

    analyzer = DribbleAnalyzer(
        min_dribble_distance=15.0,  # Lower threshold for test
        min_dribble_duration=5,  # Lower threshold for test
        confidence_threshold=0.2,  # Lower threshold for test
    )

    # Adjust detector thresholds for testing
    analyzer.detector.opponent_proximity_threshold = 100.0  # More lenient for testing

    # Create sample data
    tracks, player_assignments = create_sample_tracking_data()

    # Run complete analysis
    results = analyzer.analyze_complete_match(tracks, player_assignments)

    print(f"✅ Analysis complete:")
    print(f"   Total dribbles detected: {results['total_dribbles']}")
    print(f"   Average confidence: {results['avg_confidence']:.3f}")

    # Check team statistics
    team_stats = results["team_statistics"]
    for team_id, stats in team_stats.items():
        print(
            f"   Team {team_id}: {stats['total_dribbles']} dribbles, "
            f"{stats['avg_confidence']:.3f} avg confidence"
        )

    # Check player statistics
    player_stats = results["player_statistics"]
    for player_id, stats in player_stats.items():
        print(
            f"   Player {player_id}: {stats['total_dribbles']} dribbles, "
            f"{stats['avg_confidence']:.3f} avg confidence"
        )

    return results["total_dribbles"] > 0


def validate_csv_export():
    """Validate CSV export functionality."""
    print("\n🧪 Testing CSV export functionality...")

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = DribbleAnalyzer()

        # Create mock dribble event
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
            ball_touches=10,
        )

        # Add to analyzer
        analyzer.dribble_events.append(mock_event)
        analyzer.detector._update_statistics(mock_event)

        # Export to CSV
        exported_files = analyzer.export_to_csv(
            output_dir=temp_dir, video_name="validation_test"
        )

        print(f"✅ CSV files exported:")
        for file_type, file_path in exported_files.items():
            if os.path.exists(file_path):
                print(f"   {file_type}: {file_path} ✓")

                # Read and display first few lines
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    print(f"     Header: {lines[0].strip()}")
                    if len(lines) > 1:
                        print(f"     Data: {lines[1].strip()}")
            else:
                print(f"   {file_type}: {file_path} ✗")
                return False

        return True


def validate_integration_methods():
    """Validate methods used for integration with main pipeline."""
    print("\n🧪 Testing integration methods...")

    analyzer = DribbleAnalyzer()

    # Create mock statistics
    from src.dribble_detection.dribble_event import DribbleStatistics

    team1_stats = DribbleStatistics(
        total_dribbles=5,
        successful_dribbles=4,
        events_count=5,
        avg_confidence=0.7,
        success_rate=0.8,
    )

    team2_stats = DribbleStatistics(
        total_dribbles=3,
        successful_dribbles=2,
        events_count=3,
        avg_confidence=0.6,
        success_rate=0.67,
    )

    analyzer.detector.team_statistics[1] = team1_stats
    analyzer.detector.team_statistics[2] = team2_stats

    # Test integration methods
    dribble_counts = analyzer.get_team_dribble_counts()
    dribble_confidence = analyzer.get_team_dribble_confidence()
    dribble_events_count = analyzer.get_team_dribble_events_count()

    print(f"✅ Integration methods working:")
    print(f"   Team dribble counts: {dribble_counts}")
    print(f"   Team dribble confidence: {dribble_confidence}")
    print(f"   Team dribble events count: {dribble_events_count}")

    # Validate results
    expected_counts = {1: 5, 2: 3}
    expected_confidence = {1: 0.7, 2: 0.6}
    expected_events = {1: 5, 2: 3}

    return (
        dribble_counts == expected_counts
        and abs(dribble_confidence[1] - expected_confidence[1]) < 0.01
        and abs(dribble_confidence[2] - expected_confidence[2]) < 0.01
        and dribble_events_count == expected_events
    )


def main():
    """Run all validation tests."""
    print("🏃 Validating Dribble Detection System")
    print("=" * 50)

    tests = [
        ("Basic Functionality", validate_basic_functionality),
        ("Complete Match Analysis", validate_complete_match_analysis),
        ("CSV Export", validate_csv_export),
        ("Integration Methods", validate_integration_methods),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR: {test_name} - {e}")

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Validation Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All validation tests passed! Dribble detection system is working correctly."
        )
        return True
    else:
        print("⚠️  Some validation tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
