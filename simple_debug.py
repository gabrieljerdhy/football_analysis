#!/usr/bin/env python3
"""
Simple debug script to check ball positions and goal detection.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(".")

from goal_detection import FieldKeypointsDetector, GoalDetector
from trackers import Tracker
from utils import read_video


def simple_ball_debug(video_path, max_frames=50):
    """Simple ball position debugging."""
    print(f"Debugging ball positions in: {video_path}")

    # Read video
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")

    # Get video dimensions
    if video_frames:
        height, width = video_frames[0].shape[:2]
        print(f"Video dimensions: {width}x{height}")

    # Initialize tracker
    tracker = Tracker("models/best.pt")

    # Get tracks for first few frames
    debug_frames = video_frames[: min(max_frames, len(video_frames))]
    print(f"Processing {len(debug_frames)} frames...")

    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Analyze ball positions
    print("\nBall position analysis:")
    ball_positions = []

    for frame_num in range(min(20, len(tracks["ball"]))):  # Check first 20 frames
        ball_track = tracks["ball"][frame_num]
        if 1 in ball_track and ball_track[1]:
            ball_position = ball_track[1].get("position", None)
            if ball_position:
                ball_positions.append(ball_position)
                print(f"Frame {frame_num}: Ball at {ball_position}")

    if ball_positions:
        x_coords = [pos[0] for pos in ball_positions]
        y_coords = [pos[1] for pos in ball_positions]

        print(f"\nBall position summary:")
        print(f"X range: {min(x_coords)} - {max(x_coords)}")
        print(f"Y range: {min(y_coords)} - {max(y_coords)}")

        # Test current goal areas
        print(f"\nTesting current goal areas:")
        print(f"Left goal area: x < 100, 300 < y < 600")
        print(f"Right goal area: x > 1180, 300 < y < 600")

        left_hits = 0
        right_hits = 0

        for x, y in ball_positions:
            if x < 100 and 300 < y < 600:
                left_hits += 1
                print(f"  Ball in LEFT goal area: ({x}, {y})")
            elif x > 1180 and 300 < y < 600:
                right_hits += 1
                print(f"  Ball in RIGHT goal area: ({x}, {y})")

        print(f"\nGoal area hits: Left={left_hits}, Right={right_hits}")

        # Suggest better goal areas based on video dimensions
        if width and height:
            suggested_left = {
                "x_min": 0,
                "x_max": int(width * 0.1),
                "y_min": int(height * 0.3),
                "y_max": int(height * 0.7),
            }
            suggested_right = {
                "x_min": int(width * 0.9),
                "x_max": width,
                "y_min": int(height * 0.3),
                "y_max": int(height * 0.7),
            }

            print(f"\nSuggested goal areas for {width}x{height} video:")
            print(f"Left: {suggested_left}")
            print(f"Right: {suggested_right}")

    return tracks


def test_goal_detection_with_manual_positions(video_path):
    """Test goal detection with manual ball positions."""
    print(f"\nTesting goal detection with manual positions...")

    # Read a frame to initialize video dimensions
    video_frames = read_video(video_path)

    # Initialize goal detector
    keypoints_detector = FieldKeypointsDetector("models/best_fk.pt")
    goal_detector = GoalDetector(keypoints_detector)

    # Update with first frame to set video dimensions
    if video_frames:
        goal_detector.update_keypoints(video_frames[0])

    # Test positions that should trigger goals (updated for 1920x1080)
    test_positions = [
        (75, 400, "Should be LEFT goal (updated)"),
        (1850, 400, "Should be RIGHT goal (updated)"),
        (960, 400, "Should be MIDFIELD"),
        (50, 450, "Should be LEFT goal edge"),
        (1870, 350, "Should be RIGHT goal edge"),
        (100, 540, "Should be LEFT goal center"),
        (1820, 540, "Should be RIGHT goal center"),
    ]

    for i, (x, y, description) in enumerate(test_positions):
        # Reset goal detection state for each test
        goal_detector.goal_detected = False
        goal_detector.goal_cooldown = 0

        goal_event = goal_detector.detect_goal(
            (x, y), 1, 1, i * 100
        )  # Use different frame numbers
        if goal_event:
            print(f"✅ {description}: GOAL DETECTED - {goal_event}")
        else:
            print(f"❌ {description}: No goal detected at ({x}, {y})")

            # Debug: Check if ball is in goal area manually
            goal_side = goal_detector._check_ball_in_goal((x, y))
            if goal_side:
                print(
                    f"   🔍 Ball IS in {goal_side} goal area, but goal not detected (check trajectory validation)"
                )

    # Get statistics
    stats = goal_detector.get_goal_statistics()
    print(f"\nGoal detection stats: {stats}")


def main():
    # Test with your video
    video_path = "input_videos/08fd33_4.mp4"

    print("🔍 Simple Goal Detection Debug")
    print("=" * 40)

    # Debug ball tracking
    tracks = simple_ball_debug(video_path, 100)

    # Test manual goal detection
    test_goal_detection_with_manual_positions(video_path)

    print("\n" + "=" * 40)
    print("Debug complete!")


if __name__ == "__main__":
    main()
