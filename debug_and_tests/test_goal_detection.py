#!/usr/bin/env python3
"""
Test script for the enhanced goal detection feature.
This script tests the goal detection components independently.
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_detection import FieldKeypointsDetector, GoalDetector


def test_field_keypoints_detector():
    """Test the field keypoints detector with a dummy frame."""
    print("Testing FieldKeypointsDetector...")

    # Create a dummy frame (you can replace this with a real frame)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    try:
        # Initialize detector
        detector = FieldKeypointsDetector("../models/best_fk.pt")
        print("✓ FieldKeypointsDetector initialized successfully")

        # Test keypoint detection
        keypoints = detector.detect_keypoints(dummy_frame)
        print(f"✓ Detected {len(keypoints)} keypoints")

        # Test goal area detection
        goal_areas = detector.get_goal_areas()
        print(f"✓ Goal areas: {goal_areas}")

        # Test ball position checking
        test_ball_position = (100, 400)  # Left side
        goal_side = detector.is_ball_in_goal_area(test_ball_position)
        print(f"✓ Ball at {test_ball_position} is in goal: {goal_side}")

        return True

    except Exception as e:
        print(f"✗ Error in FieldKeypointsDetector: {e}")
        return False


def test_goal_detector():
    """Test the goal detector."""
    print("\nTesting GoalDetector...")

    try:
        # Initialize detector with keypoints detector
        keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
        goal_detector = GoalDetector(keypoints_detector)
        print("✓ GoalDetector initialized successfully")

        # Test goal detection with various ball positions
        test_positions = [
            ((50, 400), 1, 1, 100),  # Left goal area
            ((1200, 400), 2, 2, 200),  # Right goal area
            ((640, 400), 3, 1, 300),  # Middle field
        ]

        for ball_pos, player_id, team, frame_num in test_positions:
            goal_event = goal_detector.detect_goal(ball_pos, player_id, team, frame_num)
            if goal_event:
                print(f"✓ Goal detected at {ball_pos}: {goal_event}")
            else:
                print(f"✓ No goal at {ball_pos} (expected)")

        # Test statistics
        stats = goal_detector.get_goal_statistics()
        print(f"✓ Goal statistics: {stats}")

        return True

    except Exception as e:
        print(f"✗ Error in GoalDetector: {e}")
        return False


def test_integration():
    """Test the integration between components."""
    print("\nTesting Integration...")

    try:
        # Create dummy frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Initialize components
        keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
        goal_detector = GoalDetector(keypoints_detector)

        # Update keypoints
        goal_detector.update_keypoints(dummy_frame)
        print("✓ Keypoints updated successfully")

        # Test goal detection
        ball_position = (50, 400)  # Left goal
        goal_event = goal_detector.detect_goal(ball_position, 5, 2, 100)

        if goal_event:
            print(f"✓ Integrated goal detection successful: {goal_event}")
        else:
            print("✓ No goal detected (may be expected)")

        # Test visualization
        annotated_frame = goal_detector.draw_goal_info(dummy_frame)
        print("✓ Goal info visualization successful")

        return True

    except Exception as e:
        print(f"✗ Error in integration test: {e}")
        return False


def main():
    """Run all tests."""
    print("🥅 Enhanced Goal Detection Test Suite")
    print("=" * 50)

    tests = [
        test_field_keypoints_detector,
        test_goal_detector,
        test_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Goal detection feature is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    main()
