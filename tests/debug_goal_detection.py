#!/usr/bin/env python3
"""
Debug script for goal detection issues.
This script helps identify why goals are not being detected.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_detection import FieldKeypointsDetector, GoalDetector
from player_ball_assigner import PlayerBallAssigner
from trackers import Tracker
from utils import read_video


def debug_ball_tracking(video_path, max_frames=100):
    """Debug ball tracking and positions."""
    print(f"🔍 Debugging ball tracking for: {video_path}")

    # Read video
    video_frames = read_video(video_path)
    print(f"📹 Loaded {len(video_frames)} frames")

    # Initialize tracker
    tracker = Tracker("../models/best.pt")

    # Get tracks for first few frames
    debug_frames = video_frames[: min(max_frames, len(video_frames))]
    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )

    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print(f"\n📊 Ball tracking analysis:")
    ball_positions = []
    frames_with_ball = 0

    for frame_num, ball_track in enumerate(tracks["ball"]):
        if 1 in ball_track and ball_track[1]:
            ball_bbox = ball_track[1]["bbox"]
            ball_position = ball_track[1].get("position", None)

            if ball_position:
                ball_positions.append(ball_position)
                frames_with_ball += 1

                if frame_num < 10:  # Show first 10 frames
                    print(
                        f"  Frame {frame_num}: Ball at {ball_position}, bbox: {ball_bbox}"
                    )

    print(f"\n📈 Ball tracking summary:")
    print(f"  Frames with ball detected: {frames_with_ball}/{len(debug_frames)}")
    print(f"  Ball detection rate: {frames_with_ball/len(debug_frames)*100:.1f}%")

    if ball_positions:
        x_coords = [pos[0] for pos in ball_positions]
        y_coords = [pos[1] for pos in ball_positions]

        print(f"  Ball position range:")
        print(
            f"    X: {min(x_coords)} - {max(x_coords)} (width: {max(x_coords) - min(x_coords)})"
        )
        print(
            f"    Y: {min(y_coords)} - {max(y_coords)} (height: {max(y_coords) - min(y_coords)})"
        )

        # Check if ball ever reaches goal areas
        left_goal_hits = sum(1 for x in x_coords if x < 100)
        right_goal_hits = sum(1 for x in x_coords if x > 1180)

        print(f"  Ball near goals:")
        print(f"    Left goal area (x < 100): {left_goal_hits} frames")
        print(f"    Right goal area (x > 1180): {right_goal_hits} frames")

    return tracks, ball_positions


def debug_field_keypoints(video_frames, max_frames=10):
    """Debug field keypoints detection."""
    print(f"\n🎯 Debugging field keypoints detection:")

    # Initialize detector
    keypoints_detector = FieldKeypointsDetector("../models/best_keypoint.pt")

    total_keypoints = 0
    goal_keypoints_found = 0

    for frame_num in range(min(max_frames, len(video_frames))):
        frame = video_frames[frame_num]
        keypoints = keypoints_detector.detect_keypoints(frame)

        total_keypoints += len(keypoints)

        # Check for goal-related keypoints
        goal_related = [kp for kp in keypoints.keys() if "goal" in kp.lower()]
        goal_keypoints_found += len(goal_related)

        if frame_num < 3:  # Show first 3 frames
            print(f"  Frame {frame_num}: {len(keypoints)} keypoints detected")
            if keypoints:
                for name, data in keypoints.items():
                    print(
                        f"    {name}: {data['position']} (conf: {data['confidence']:.2f})"
                    )

    print(f"\n📊 Keypoints summary:")
    print(f"  Total keypoints detected: {total_keypoints}")
    print(f"  Goal-related keypoints: {goal_keypoints_found}")

    # Check goal areas
    goal_areas = keypoints_detector.get_goal_areas()
    print(f"  Goal areas: {goal_areas}")

    return keypoints_detector


def debug_goal_detection(video_path, max_frames=100):
    """Debug the complete goal detection system."""
    print(f"\n🥅 Debugging goal detection system:")

    # Get ball tracking data
    tracks, ball_positions = debug_ball_tracking(video_path, max_frames)

    # Get field keypoints
    video_frames = read_video(video_path)
    keypoints_detector = debug_field_keypoints(video_frames, 10)

    # Initialize goal detector
    goal_detector = GoalDetector(keypoints_detector)

    # Test goal detection with actual ball positions
    print(f"\n🔍 Testing goal detection with real ball positions:")

    player_assigner = PlayerBallAssigner()
    goals_detected = 0

    for frame_num, ball_track in enumerate(tracks["ball"][:max_frames]):
        if 1 in ball_track and ball_track[1]:
            ball_bbox = ball_track[1]["bbox"]
            ball_position = ball_track[1].get("position", None)

            if ball_position:
                # Update keypoints for this frame
                goal_detector.update_keypoints(video_frames[frame_num])

                # Try to assign ball to player
                player_track = tracks["players"][frame_num]
                assigned_player = player_assigner.assign_ball_to_player(
                    player_track, ball_bbox
                )

                # Test goal detection
                goal_event = goal_detector.detect_goal(
                    ball_position, assigned_player, 1, frame_num  # Assume team 1
                )

                if goal_event:
                    goals_detected += 1
                    print(f"  🎉 Goal detected at frame {frame_num}: {goal_event}")

                # Check if ball is in goal areas manually
                x, y = ball_position

                # Check fallback areas
                if x < 100 and 300 < y < 600:
                    print(
                        f"  ⚠️  Ball in left goal area at frame {frame_num}: {ball_position}"
                    )
                elif x > 1180 and 300 < y < 600:
                    print(
                        f"  ⚠️  Ball in right goal area at frame {frame_num}: {ball_position}"
                    )

    print(f"\n📊 Goal detection summary:")
    print(f"  Goals detected: {goals_detected}")

    # Get final statistics
    stats = goal_detector.get_goal_statistics()
    print(f"  Final stats: {stats}")

    return goal_detector


def create_debug_video(video_path, output_path, max_frames=100):
    """Create a debug video showing ball positions and goal areas."""
    print(f"\n🎬 Creating debug video: {output_path}")

    # Read video
    video_frames = read_video(video_path)
    debug_frames = video_frames[: min(max_frames, len(video_frames))]

    # Get tracking data
    tracker = Tracker("../models/best_detect.pt")
    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize goal detection
    keypoints_detector = FieldKeypointsDetector("../models/best_keypoint.pt")
    goal_detector = GoalDetector(keypoints_detector)

    output_frames = []

    for frame_num, frame in enumerate(debug_frames):
        debug_frame = frame.copy()

        # Update keypoints
        goal_detector.update_keypoints(frame)

        # Draw goal areas (fallback)
        cv2.rectangle(debug_frame, (0, 300), (100, 600), (0, 255, 0), 3)  # Left goal
        cv2.rectangle(
            debug_frame, (1180, 300), (1280, 600), (0, 0, 255), 3
        )  # Right goal

        # Draw ball position
        if 1 in tracks["ball"][frame_num] and tracks["ball"][frame_num][1]:
            ball_position = tracks["ball"][frame_num][1].get("position", None)
            if ball_position:
                cv2.circle(debug_frame, ball_position, 10, (255, 255, 0), -1)
                cv2.putText(
                    debug_frame,
                    f"Ball: {ball_position}",
                    (ball_position[0] + 15, ball_position[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

        # Draw frame number
        cv2.putText(
            debug_frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Draw goal areas info
        cv2.putText(
            debug_frame,
            "Left Goal",
            (10, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            "Right Goal",
            (1100, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        output_frames.append(debug_frame)

    # Save debug video
    from utils import save_video

    save_video(output_frames, output_path)
    print(f"✅ Debug video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug goal detection issues")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output debug video path")
    parser.add_argument(
        "--frames", "-f", type=int, default=100, help="Max frames to analyze"
    )

    args = parser.parse_args()

    print("🔧 Goal Detection Debug Tool")
    print("=" * 50)

    # Run debugging
    goal_detector = debug_goal_detection(args.input, args.frames)

    # Create debug video if requested
    if args.output:
        create_debug_video(args.input, args.output, args.frames)

    print("\n" + "=" * 50)
    print("🏁 Debug analysis complete!")
    print("\nRecommendations:")
    print("1. Check if ball positions are in expected coordinate ranges")
    print("2. Verify goal area coordinates match your video dimensions")
    print("3. Ensure field keypoints model is detecting goal posts")
    print("4. Consider adjusting goal area boundaries if needed")


if __name__ == "__main__":
    main()
