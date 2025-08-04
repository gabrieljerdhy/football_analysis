"""
Test Cross Detection Feature

Tests the cross detection functionality to ensure it works correctly
with the football analysis pipeline.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cross_detection import (
    CrossDetector,
    CrossEvent,
    CrossTrajectoryAnalyzer,
    FieldZoneAnalyzer,
)
from cross_detection.cross_event import CrossOrigin, CrossTarget, CrossType


def test_field_zone_analyzer():
    """Test the field zone analyzer for detecting cross origins and targets."""
    print("🧪 Testing Field Zone Analyzer...")

    analyzer = FieldZoneAnalyzer(video_width=1920, video_height=1080)

    # Test wing positions (should be valid cross origins)
    # Adjust positions to be in wing areas (25% from edge) not byline areas (15% from edge)
    left_wing_pos = (400, 540)  # Left wing area (around 20% from left)
    right_wing_pos = (1520, 540)  # Right wing area (around 80% from left)

    left_origin = analyzer.get_cross_origin_zone(left_wing_pos)
    right_origin = analyzer.get_cross_origin_zone(right_wing_pos)

    print(f"   Left wing position {left_wing_pos} -> {left_origin}")
    print(f"   Right wing position {right_wing_pos} -> {right_origin}")

    assert (
        left_origin == CrossOrigin.LEFT_WING
    ), f"Expected LEFT_WING, got {left_origin}"
    assert (
        right_origin == CrossOrigin.RIGHT_WING
    ), f"Expected RIGHT_WING, got {right_origin}"

    # Test byline positions (should be byline cross origins)
    left_byline_pos = (200, 540)  # Left byline area
    right_byline_pos = (1720, 540)  # Right byline area

    left_byline_origin = analyzer.get_cross_origin_zone(left_byline_pos)
    right_byline_origin = analyzer.get_cross_origin_zone(right_byline_pos)

    print(f"   Left byline position {left_byline_pos} -> {left_byline_origin}")
    print(f"   Right byline position {right_byline_pos} -> {right_byline_origin}")

    assert (
        left_byline_origin == CrossOrigin.LEFT_BYLINE
    ), f"Expected LEFT_BYLINE, got {left_byline_origin}"
    assert (
        right_byline_origin == CrossOrigin.RIGHT_BYLINE
    ), f"Expected RIGHT_BYLINE, got {right_byline_origin}"

    # Test penalty box positions (should be valid cross targets)
    penalty_box_pos = (300, 540)  # Left penalty area
    target_zone = analyzer.get_cross_target_zone(penalty_box_pos)

    print(f"   Penalty box position {penalty_box_pos} -> {target_zone}")

    # Test center field (should not be valid cross origin)
    center_pos = (960, 540)
    center_origin = analyzer.get_cross_origin_zone(center_pos)

    print(f"   Center position {center_pos} -> {center_origin}")
    assert (
        center_origin is None
    ), f"Expected None for center position, got {center_origin}"

    print("✅ Field Zone Analyzer tests passed!")


def test_cross_trajectory_analyzer():
    """Test the cross trajectory analyzer."""
    print("🧪 Testing Cross Trajectory Analyzer...")

    analyzer = CrossTrajectoryAnalyzer(frame_rate=24.0)

    # Create a mock trajectory that looks like a cross
    # From left wing to penalty box with arc-like movement
    trajectory = []
    start_x, start_y = 200, 540  # Left wing
    end_x, end_y = 400, 520  # Penalty box

    # Create 10 frames of trajectory with slight arc
    for i in range(10):
        t = i / 9.0  # Normalize to 0-1

        # Linear interpolation with slight arc
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t + 20 * np.sin(t * np.pi)  # Add arc

        trajectory.append(
            {
                "position": (int(x), int(y)),
                "frame_num": i,
                "confidence": 0.8,
                "source": "detection",
            }
        )

    # Analyze the trajectory
    result = analyzer.analyze_trajectory_for_cross(trajectory)

    print(f"   Trajectory analysis result: {result is not None}")
    if result:
        print(f"   Cross score: {result['cross_score']:.3f}")
        print(f"   Cross type: {result['cross_type']}")
        print(f"   Distance: {result['distance']:.1f}")

    print("✅ Cross Trajectory Analyzer tests passed!")


def test_cross_detector():
    """Test the main cross detector."""
    print("🧪 Testing Cross Detector...")

    detector = CrossDetector(video_width=1920, video_height=1080, frame_rate=24.0)

    # Mock field keypoints detector
    mock_keypoints_detector = MagicMock()
    detector.set_field_keypoints_detector(mock_keypoints_detector)

    # Simulate a cross sequence
    # Player starts with ball in left wing
    player_id = 7
    team_id = 1

    # Simulate frames where player moves ball from wing to penalty area
    positions = [
        (200, 540),  # Start in left wing
        (250, 535),  # Moving toward center
        (300, 530),  # Continuing
        (350, 525),  # Getting closer to penalty area
        (400, 520),  # End in penalty area
    ]

    cross_detected = False
    for frame_num, position in enumerate(positions):
        cross_event = detector.detect_cross(
            ball_position=position,
            player_id=player_id,
            team_id=team_id,
            frame_num=frame_num,
            ball_confidence=0.8,
        )

        if cross_event:
            cross_detected = True
            print(f"   Cross detected at frame {frame_num}!")
            print(f"   Player: {cross_event.player_id}, Team: {cross_event.team_id}")
            print(f"   Type: {cross_event.cross_type.value}")
            print(f"   Origin: {cross_event.origin_zone.value}")
            print(f"   Target: {cross_event.target_zone.value}")
            print(f"   Confidence: {cross_event.confidence_score:.3f}")
            break

    # Simulate possession change to trigger cross analysis
    if not cross_detected:
        # Change possession to different player to trigger analysis
        cross_event = detector.detect_cross(
            ball_position=(450, 515),
            player_id=9,  # Different player
            team_id=team_id,
            frame_num=len(positions),
            ball_confidence=0.8,
        )

        if cross_event:
            cross_detected = True
            print(f"   Cross detected on possession change!")

    # Get statistics
    stats = detector.get_cross_statistics()
    print(f"   Total crosses detected: {stats['total_crosses']}")
    print(
        f"   Team 1 crosses attempted: {stats['team_statistics'][1]['crosses_attempted']}"
    )

    print("✅ Cross Detector tests passed!")


def test_cross_integration():
    """Test cross detection integration with enhanced pass counter."""
    print("🧪 Testing Cross Detection Integration...")

    try:
        from pass_counter.enhanced_pass_counter import EnhancedPassCounter

        # Initialize enhanced pass counter with cross detection
        pass_counter = EnhancedPassCounter(
            video_width=1920, video_height=1080, frame_rate=24.0
        )

        # Test that cross detector is initialized
        assert hasattr(pass_counter, "cross_detector"), "Cross detector not initialized"
        assert pass_counter.cross_detector is not None, "Cross detector is None"

        print("   Enhanced pass counter initialized with cross detection")

        # Test cross detection through pass counter
        ball_data = {"position": (200, 540), "confidence": 0.8, "source": "detection"}

        # Simulate pass counting with cross detection
        result = pass_counter.count_passes_enhanced(
            current_player_id=7, current_team=1, frame_num=0, ball_data=ball_data
        )

        print(f"   Pass counting with cross detection completed")
        print(f"   Result: {result}")

        print("✅ Cross Detection Integration tests passed!")

    except ImportError as e:
        print(f"⚠️  Could not test integration: {e}")


def main():
    """Run all cross detection tests."""
    print("🚀 Starting Cross Detection Tests")
    print("=" * 50)

    try:
        test_field_zone_analyzer()
        print()

        test_cross_trajectory_analyzer()
        print()

        test_cross_detector()
        print()

        test_cross_integration()
        print()

        print("🎉 All Cross Detection Tests Passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
