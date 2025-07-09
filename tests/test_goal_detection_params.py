#!/usr/bin/env python3
"""
Simple test script to check goal detection parameters and identify issues.
"""

import os
import sys

# Add the parent directory (project root) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.goal_detection import FieldKeypointsDetector, GoalDetector


def analyze_goal_detection_config():
    """Analyze current goal detection configuration."""
    print("🔍 Goal Detection Configuration Analysis")
    print("=" * 50)

    # Initialize goal detector
    keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
    goal_detector = GoalDetector(keypoints_detector)

    print(f"📊 Current Goal Detection Parameters:")
    print(f"   Goal cooldown frames: {goal_detector.goal_cooldown_frames}")
    print(f"   Min trajectory for goal: {goal_detector.min_trajectory_for_goal}")
    print(f"   Goal direction threshold: {goal_detector.goal_direction_threshold}")
    print(f"   Goal speed threshold: {goal_detector.goal_speed_threshold}")
    print(f"   Max trajectory length: {goal_detector.max_trajectory_length}")

    print(f"\n🎯 Fallback Goal Areas:")
    print(f"   Left goal: {goal_detector.fallback_goal_areas['left']}")
    print(f"   Right goal: {goal_detector.fallback_goal_areas['right']}")

    # Calculate cooldown time
    fps = 25  # Assuming 25 fps from video info
    cooldown_seconds = goal_detector.goal_cooldown_frames / fps
    print(f"\n⏱️  Goal Detection Timing:")
    print(f"   Cooldown period: {cooldown_seconds:.1f} seconds")
    print(f"   Min trajectory frames: {goal_detector.min_trajectory_for_goal}")
    print(
        f"   Min trajectory time: {goal_detector.min_trajectory_for_goal / fps:.2f} seconds"
    )

    print(f"\n🚨 Potential Issues Identified:")

    issues = []
    if goal_detector.goal_cooldown_frames > 300:  # > 10 seconds at 30fps
        issues.append(
            f"Very long cooldown period ({cooldown_seconds:.1f}s) may prevent multiple goals"
        )

    if goal_detector.min_trajectory_for_goal > 8:
        issues.append(
            f"High minimum trajectory requirement ({goal_detector.min_trajectory_for_goal} frames)"
        )

    if goal_detector.goal_direction_threshold > 0.7:
        issues.append(
            f"Very strict direction threshold ({goal_detector.goal_direction_threshold})"
        )

    if goal_detector.goal_speed_threshold > 12:
        issues.append(
            f"High speed threshold ({goal_detector.goal_speed_threshold} pixels)"
        )

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   No obvious parameter issues found")

    print(f"\n💡 Recommended Adjustments for Better Sensitivity:")
    print(
        f"   1. Reduce goal_cooldown_frames from {goal_detector.goal_cooldown_frames} to 300 (10 seconds)"
    )
    print(
        f"   2. Reduce min_trajectory_for_goal from {goal_detector.min_trajectory_for_goal} to 6-8 frames"
    )
    print(
        f"   3. Reduce goal_direction_threshold from {goal_detector.goal_direction_threshold} to 0.6"
    )
    print(
        f"   4. Reduce goal_speed_threshold from {goal_detector.goal_speed_threshold} to 8-10 pixels"
    )
    print(
        f"   5. Consider reducing validation threshold in _validate_goal_trajectory from 0.75 to 0.6"
    )

    return goal_detector


def test_goal_area_coverage():
    """Test goal area coverage for typical video dimensions."""
    print(f"\n🥅 Goal Area Coverage Analysis")
    print("=" * 50)

    # Typical video dimensions
    video_width = 1920
    video_height = 1080

    keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
    goal_detector = GoalDetector(keypoints_detector)

    # Update goal areas based on video dimensions
    goal_detector._update_fallback_goal_areas(video_width, video_height)

    print(f"📺 Video dimensions: {video_width}x{video_height}")
    print(f"🎯 Updated goal areas:")
    print(f"   Left goal: {goal_detector.fallback_goal_areas['left']}")
    print(f"   Right goal: {goal_detector.fallback_goal_areas['right']}")

    # Calculate goal area percentages
    left_area = goal_detector.fallback_goal_areas["left"]
    right_area = goal_detector.fallback_goal_areas["right"]

    left_width = left_area["x_max"] - left_area["x_min"]
    left_height = left_area["y_max"] - left_area["y_min"]
    left_percentage = (left_width * left_height) / (video_width * video_height) * 100

    right_width = right_area["x_max"] - right_area["x_min"]
    right_height = right_area["y_max"] - right_area["y_min"]
    right_percentage = (right_width * right_height) / (video_width * video_height) * 100

    print(f"\n📏 Goal Area Coverage:")
    print(
        f"   Left goal: {left_width}x{left_height} pixels ({left_percentage:.2f}% of frame)"
    )
    print(
        f"   Right goal: {right_width}x{right_height} pixels ({right_percentage:.2f}% of frame)"
    )

    if left_percentage < 1.0 or right_percentage < 1.0:
        print(f"⚠️  Warning: Goal areas may be too small for reliable detection")

    return goal_detector


if __name__ == "__main__":
    print("🔧 Goal Detection Parameter Analysis Tool")
    print("=" * 60)

    try:
        goal_detector = analyze_goal_detection_config()
        test_goal_area_coverage()

        print(f"\n✅ Analysis complete. Review the recommendations above.")
        print(
            f"💡 Consider adjusting parameters in src/goal_detection/goal_detector.py"
        )

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
