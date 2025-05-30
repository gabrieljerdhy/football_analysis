#!/usr/bin/env python3
"""
Debug script to understand why goals aren't being detected in real videos.
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_detection import FieldKeypointsDetector, GoalDetector
from player_ball_assigner import PlayerBallAssigner
from trackers import Tracker
from utils import read_video


def analyze_ball_positions_in_video(video_path, max_frames=200):
    """Analyze ball positions throughout the video to find potential goals."""
    print(f"🔍 Analyzing ball positions in: {video_path}")

    # Read video
    video_frames = read_video(video_path)
    print(f"📹 Loaded {len(video_frames)} frames")

    if not video_frames:
        print("❌ No frames loaded!")
        return

    # Get video dimensions
    height, width = video_frames[0].shape[:2]
    print(f"📐 Video dimensions: {width}x{height}")

    # Initialize tracker
    tracker = Tracker("../models/best.pt")

    # Process frames
    debug_frames = video_frames[: min(max_frames, len(video_frames))]
    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize goal detector
    keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
    goal_detector = GoalDetector(keypoints_detector)

    # Update goal detector with first frame to set dimensions
    goal_detector.update_keypoints(video_frames[0])

    print(f"\n📊 Goal areas calculated:")
    print(f"   Left goal: {goal_detector.fallback_goal_areas['left']}")
    print(f"   Right goal: {goal_detector.fallback_goal_areas['right']}")

    # Analyze ball positions
    ball_positions = []
    goal_area_hits = {"left": [], "right": []}

    for frame_num in range(len(tracks["ball"])):
        ball_track = tracks["ball"][frame_num]
        if 1 in ball_track and ball_track[1]:
            ball_position = ball_track[1].get("position", None)
            if ball_position:
                ball_positions.append((frame_num, ball_position))

                # Check if ball is in goal areas
                x, y = ball_position

                # Check left goal
                left_area = goal_detector.fallback_goal_areas["left"]
                if (
                    left_area["x_min"] <= x <= left_area["x_max"]
                    and left_area["y_min"] <= y <= left_area["y_max"]
                ):
                    goal_area_hits["left"].append((frame_num, ball_position))
                    print(
                        f"🥅 Ball in LEFT goal area at frame {frame_num}: {ball_position}"
                    )

                # Check right goal
                right_area = goal_detector.fallback_goal_areas["right"]
                if (
                    right_area["x_min"] <= x <= right_area["x_max"]
                    and right_area["y_min"] <= y <= right_area["y_max"]
                ):
                    goal_area_hits["right"].append((frame_num, ball_position))
                    print(
                        f"🥅 Ball in RIGHT goal area at frame {frame_num}: {ball_position}"
                    )

    print(f"\n📈 Ball tracking summary:")
    print(f"   Total ball positions detected: {len(ball_positions)}")
    print(f"   Left goal area hits: {len(goal_area_hits['left'])}")
    print(f"   Right goal area hits: {len(goal_area_hits['right'])}")

    if ball_positions:
        x_coords = [pos[1][0] for pos in ball_positions]
        y_coords = [pos[1][1] for pos in ball_positions]

        print(f"\n📍 Ball position range:")
        print(f"   X: {min(x_coords)} - {max(x_coords)}")
        print(f"   Y: {min(y_coords)} - {max(y_coords)}")

        # Show some sample positions
        print(f"\n📋 Sample ball positions:")
        for i, (frame_num, pos) in enumerate(ball_positions[:10]):
            print(f"   Frame {frame_num}: {pos}")

    # Test goal detection with actual ball positions
    print(f"\n🧪 Testing goal detection with real ball positions:")

    player_assigner = PlayerBallAssigner()
    goals_detected = 0

    for frame_num, ball_position in ball_positions[:50]:  # Test first 50 ball positions
        # Reset goal detector state for each test
        goal_detector.goal_detected = False
        goal_detector.goal_cooldown = 0

        # Update keypoints
        if frame_num < len(video_frames):
            goal_detector.update_keypoints(video_frames[frame_num])

        # Try to assign ball to player
        if frame_num < len(tracks["players"]):
            player_track = tracks["players"][frame_num]
            ball_bbox = (
                tracks["ball"][frame_num][1]["bbox"]
                if 1 in tracks["ball"][frame_num]
                else None
            )

            if ball_bbox is not None:
                assigned_player = player_assigner.assign_ball_to_player(
                    player_track, ball_bbox
                )

                # Test goal detection
                goal_event = goal_detector.detect_goal(
                    ball_position, assigned_player, 1, frame_num
                )

                if goal_event:
                    goals_detected += 1
                    print(f"🎉 GOAL DETECTED at frame {frame_num}: {goal_event}")

    print(f"\n🏁 Final results:")
    print(f"   Goals detected by enhanced system: {goals_detected}")

    return ball_positions, goal_area_hits


def create_visual_debug_video(video_path, output_path, max_frames=100):
    """Create a visual debug video showing goal areas and ball positions."""
    print(f"\n🎬 Creating visual debug video...")

    # Read video
    video_frames = read_video(video_path)
    debug_frames = video_frames[: min(max_frames, len(video_frames))]

    # Get tracking data
    tracker = Tracker("../models/best.pt")
    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize goal detection
    keypoints_detector = FieldKeypointsDetector("../models/best_fk.pt")
    goal_detector = GoalDetector(keypoints_detector)
    goal_detector.update_keypoints(video_frames[0])

    output_frames = []

    for frame_num, frame in enumerate(debug_frames):
        debug_frame = frame.copy()

        # Draw goal areas
        left_area = goal_detector.fallback_goal_areas["left"]
        right_area = goal_detector.fallback_goal_areas["right"]

        # Draw left goal (green)
        cv2.rectangle(
            debug_frame,
            (left_area["x_min"], left_area["y_min"]),
            (left_area["x_max"], left_area["y_max"]),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            debug_frame,
            "LEFT GOAL",
            (left_area["x_min"], left_area["y_min"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Draw right goal (red)
        cv2.rectangle(
            debug_frame,
            (right_area["x_min"], right_area["y_min"]),
            (right_area["x_max"], right_area["y_max"]),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            debug_frame,
            "RIGHT GOAL",
            (right_area["x_min"], right_area["y_min"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Draw ball position
        if 1 in tracks["ball"][frame_num] and tracks["ball"][frame_num][1]:
            ball_position = tracks["ball"][frame_num][1].get("position", None)
            if ball_position:
                cv2.circle(debug_frame, ball_position, 15, (255, 255, 0), -1)
                cv2.putText(
                    debug_frame,
                    f"BALL",
                    (ball_position[0] + 20, ball_position[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        # Draw frame info
        cv2.putText(
            debug_frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Draw goal area coordinates
        cv2.putText(
            debug_frame,
            f"Left: {left_area['x_min']}-{left_area['x_max']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            f"Right: {right_area['x_min']}-{right_area['x_max']}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        output_frames.append(debug_frame)

    # Save debug video
    from utils import save_video

    save_video(output_frames, output_path)
    print(f"✅ Debug video saved to: {output_path}")


def main():
    video_path = "../input_videos/goal_test_1.mp4"

    print("🔍 Goal Detection Debug Analysis")
    print("=" * 50)

    # Analyze ball positions
    ball_positions, goal_hits = analyze_ball_positions_in_video(video_path, 300)

    # Create visual debug video
    create_visual_debug_video(video_path, "../output_videos/goal_debug_visual.avi", 100)

    print("\n" + "=" * 50)
    print("🏁 Debug analysis complete!")

    if len(goal_hits["left"]) == 0 and len(goal_hits["right"]) == 0:
        print("\n💡 Recommendations:")
        print("1. The ball may not be entering the goal areas in this video")
        print("2. Goal areas might need adjustment for this camera angle")
        print("3. Check the debug video to see goal areas visually")
        print("4. Try with a different video that definitely contains goals")


if __name__ == "__main__":
    main()
