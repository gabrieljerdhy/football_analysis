#!/usr/bin/env python3
"""
Complete the full analysis pipeline for your large video.
This script will generate the output video and CSV files.
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from goal_detection import FieldKeypointsDetector, GoalDetector
from pass_counter.pass_counter import PassCounter
from pass_counter.tackle_counter import TackleCounter
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import VideoFrameIterator, get_video_info, monitor_memory_usage, save_video
from utils.goal_utils import (
    calculate_final_goal_stats,
    export_consolidated_goal_statistics,
    load_manual_goals,
)
from view_transformer import ViewTransformer


def complete_large_video_analysis(
    input_video_path: str,
    output_video_path: str = None,
    tracks_pickle_path: str = None,
    goals_config: str = None,
    batch_size: int = 50,
):
    """
    Complete the full analysis pipeline using existing tracks data.
    """

    print("🎬 COMPLETING FULL ANALYSIS PIPELINE")
    print("=" * 60)

    # Setup paths
    video_name = Path(input_video_path).stem
    if output_video_path is None:
        output_video_path = f"output_videos/{video_name}_output.avi"

    if tracks_pickle_path is None:
        tracks_pickle_path = f"stubs/{video_name}_tracks.pkl"

    # Create output directories
    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print(f"📹 Input video: {input_video_path}")
    print(f"📁 Tracks file: {tracks_pickle_path}")
    print(f"🎬 Output video: {output_video_path}")
    print(f"💾 Initial memory: {monitor_memory_usage():.2f} GB")

    # Load tracks
    if not os.path.exists(tracks_pickle_path):
        print(f"❌ Tracks file not found: {tracks_pickle_path}")
        return False

    print("📂 Loading tracks...")
    with open(tracks_pickle_path, "rb") as f:
        tracks = pickle.load(f)

    print(f"✅ Loaded tracks for {len(tracks['players'])} frames")

    # Get video info
    video_info = get_video_info(input_video_path)
    print(
        f"📹 Video: {video_info['total_frames']} frames, {video_info['duration_seconds']/60:.1f} min"
    )

    # Initialize components
    print("\n🔧 Initializing analysis components...")

    # Initialize Goal Detection System
    field_keypoints_detector = FieldKeypointsDetector("models/best_keypoint.pt")
    goal_detector = GoalDetector(field_keypoints_detector)

    # Load manual goals if provided
    manual_goals = load_manual_goals(goals_config)

    # Team assignment (already done, but we need the team_assigner object)
    team_assigner = TeamAssigner()
    # Set default team colors since we already processed teams
    team_assigner.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracker = Tracker(
        "models/best_detect.pt", enable_jersey_detection=False
    )  # Just for interpolation
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize counters
    pass_counter = PassCounter()
    tackle_counter = TackleCounter()
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    print("\n🔄 Processing game analysis...")

    # Process each frame for game analysis
    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num % 1000 == 0:
            progress = (frame_num / len(tracks["players"])) * 100
            print(
                f"  📊 Analysis progress: {progress:.1f}% ({frame_num}/{len(tracks['players'])})"
            )

        # Get ball info
        ball_info = tracks["ball"][frame_num].get(1, {})
        ball_bbox = ball_info.get("bbox", [])
        ball_position = ball_info.get("position", None)

        # Assign ball to player
        if ball_bbox and len(ball_bbox) == 4:
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
        else:
            assigned_player = -1

        if assigned_player != -1:
            current_team = tracks["players"][frame_num][assigned_player].get("team", 1)
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(current_team)

            # Count passes
            pass_counter.count_passes(assigned_player, current_team, frame_num)

            # Detect goals
            if ball_position:
                pass_counter.detect_goal(
                    ball_position, assigned_player, current_team, frame_num
                )

            # Detect tackles
            tackle_counter.detect_tackles_and_interceptions(
                player_track, assigned_player, current_team, frame_num
            )
        else:
            # No player has ball
            pass_counter.count_passes(-1, None, frame_num)
            tackle_counter.detect_tackles_and_interceptions(
                player_track, -1, None, frame_num
            )

            if ball_position:
                pass_counter.detect_goal(ball_position, -1, None, frame_num)

            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    print(f"✅ Game analysis complete")
    print(f"💾 Memory usage: {monitor_memory_usage():.2f} GB")

    # Generate statistics
    print("\n📊 Generating statistics...")

    enhanced_goal_stats = goal_detector.get_goal_statistics()
    final_team_goals, final_player_goals = calculate_final_goal_stats(
        pass_counter, enhanced_goal_stats, manual_goals
    )

    # Export CSV files
    team_csv_path, player_csv_path = export_consolidated_goal_statistics(
        video_name,
        pass_counter,
        enhanced_goal_stats,
        final_team_goals,
        final_player_goals,
        tackle_counter,
    )

    print(f"✅ CSV files generated:")
    print(f"   📊 {team_csv_path}")
    print(f"   📊 {player_csv_path}")

    # Generate output video
    print(f"\n🎬 Generating output video...")
    print(f"   This will take time for {video_info['total_frames']} frames...")

    # Process video in batches for memory efficiency
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    frames_written = 0

    with VideoFrameIterator(input_video_path, batch_size) as frame_iterator:
        for batch_frames in frame_iterator:
            batch_output_frames = []

            for i, frame in enumerate(batch_frames):
                frame_num = frames_written + i
                if frame_num >= len(tracks["players"]):
                    break

                # Initialize video writer with first frame
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(
                        output_video_path, fourcc, 24.0, (width, height)
                    )
                    print(f"📝 Video writer initialized: {width}x{height}")

                # Draw annotations
                annotated_frame = tracker.draw_annotations(
                    [frame],
                    {
                        "players": [tracks["players"][frame_num]],
                        "referees": [tracks["referees"][frame_num]],
                        "ball": [tracks["ball"][frame_num]],
                    },
                    team_ball_control[frame_num : frame_num + 1],
                )[0]

                # Draw pass and goal counts
                annotated_frame = pass_counter.draw_pass_counts([annotated_frame])[0]
                annotated_frame = pass_counter.draw_goal_counts([annotated_frame])[0]

                batch_output_frames.append(annotated_frame)

            # Write batch to video
            for frame in batch_output_frames:
                out.write(frame)
                frames_written += 1

            # Progress update
            if frames_written % 1000 == 0:
                progress = (frames_written / video_info["total_frames"]) * 100
                print(
                    f"  🎬 Video progress: {progress:.1f}% ({frames_written}/{video_info['total_frames']})"
                )

    if out:
        out.release()

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Final statistics:")
    print(f"   Team 1 goals: {final_team_goals.get(1, 0)}")
    print(f"   Team 2 goals: {final_team_goals.get(2, 0)}")
    print(f"   Total frames processed: {frames_written}")
    print(f"💾 Final memory usage: {monitor_memory_usage():.2f} GB")

    print(f"\n📁 Output files:")
    print(f"   🎬 Video: {output_video_path}")
    print(f"   📊 Team stats: {team_csv_path}")
    print(f"   📊 Player stats: {player_csv_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete analysis for large video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--tracks", help="Tracks pickle file path")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for video processing"
    )
    parser.add_argument("--goals-config", help="Goals configuration file")

    args = parser.parse_args()

    success = complete_large_video_analysis(
        input_video_path=args.input,
        output_video_path=args.output,
        tracks_pickle_path=args.tracks,
        goals_config=args.goals_config,
        batch_size=args.batch_size,
    )

    if success:
        print("\n✅ SUCCESS: Full analysis pipeline completed!")
    else:
        print("\n❌ FAILED: Analysis pipeline encountered errors")
