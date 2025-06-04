import argparse
import csv
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from goal_detection import FieldKeypointsDetector, GoalDetector
from pass_counter.pass_counter import PassCounter
from pass_counter.tackle_counter import TackleCounter
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from utils.goal_utils import (
    calculate_final_goal_stats,
    export_consolidated_goal_statistics,
    load_manual_goals,
)
from view_transformer import ViewTransformer


def main(
    input_video_path,
    output_video_path=None,
    use_stubs=True,
    force_regenerate=False,
    goals_config=None,
    enable_camera_movement=False,
    enable_speed_distance=False,
):
    """
    Process a football video to track players, detect passes, and analyze the game.

    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str, optional): Path to save the output video. If None,
                                          a default path will be generated
        use_stubs (bool): Whether to use stub files for faster processing
        force_regenerate (bool): Whether to force regeneration of stub files even if they exist
        goals_config (str, optional): Path to a CSV file with manual goal information
        enable_camera_movement (bool): Whether to enable camera movement estimation (disabled by default for memory optimization)
        enable_speed_distance (bool): Whether to enable speed and distance estimation (disabled by default for memory optimization)
    """
    # Create output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("stubs", exist_ok=True)

    # Generate default output path if not provided
    if output_video_path is None:
        video_name = Path(input_video_path).stem
        output_video_path = f"output_videos/{video_name}_output.avi"

    # Generate stub paths based on input video name
    video_name = Path(input_video_path).stem
    tracks_stub_path = f"stubs/{video_name}_tracks.pkl"

    # Only generate camera movement stub path if camera movement is enabled
    if enable_camera_movement:
        camera_movement_stub_path = f"stubs/{video_name}_camera_movement.pkl"

    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")

    # Read Video
    video_frames = read_video(input_video_path)
    print(f"Loaded {len(video_frames)} frames")

    # Initialize Tracker with jersey number detection
    tracker = Tracker("models/best.pt", enable_jersey_detection=True)

    # Initialize Goal Detection System
    field_keypoints_detector = FieldKeypointsDetector("models/best_fk.pt")
    goal_detector = GoalDetector(field_keypoints_detector)

    # Load manual goals if provided
    manual_goals = load_manual_goals(goals_config)

    # Check if tracks stub exists and should be used
    read_tracks_from_stub = (
        use_stubs and os.path.exists(tracks_stub_path) and not force_regenerate
    )

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=read_tracks_from_stub, stub_path=tracks_stub_path
    )

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Add jersey numbers to tracks using OCR
    tracker.add_jersey_numbers_to_tracks(tracks, video_frames, frame_sampling=5)

    # Camera movement estimator (optional - disabled by default for memory optimization)
    if enable_camera_movement:
        print("🎥 Camera movement estimation enabled")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])

        # Check if camera movement stub exists and should be used
        read_camera_from_stub = (
            use_stubs
            and os.path.exists(camera_movement_stub_path)
            and not force_regenerate
        )

        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames,
            read_from_stub=read_camera_from_stub,
            stub_path=camera_movement_stub_path,
        )

        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement_per_frame
        )
    else:
        print("🎥 Camera movement estimation disabled (memory optimization)")
        camera_movement_estimator = None
        camera_movement_per_frame = None

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator (optional - disabled by default for memory optimization)
    if enable_speed_distance:
        print("🏃 Speed and distance estimation enabled")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    else:
        print("🏃 Speed and distance estimation disabled (memory optimization)")
        speed_and_distance_estimator = None

    # Assign Player Teams
    team_assigner = TeamAssigner()

    # Make sure there are player detections in the first frame
    first_frame_with_players = 0
    for i, player_frame in enumerate(tracks["players"]):
        if player_frame:  # If there are player detections in this frame
            first_frame_with_players = i
            break

    # Use the first frame that has player detections
    if first_frame_with_players < len(video_frames):
        team_assigner.assign_team_color(
            video_frames[first_frame_with_players],
            tracks["players"][first_frame_with_players],
        )
    else:
        # If no frames have player detections, use default colors
        team_assigner.assign_team_color(video_frames[0], {})

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # Initialize pass counter
    pass_counter = PassCounter()

    # Initialize tackle counter
    tackle_counter = TackleCounter()

    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # Process each frame
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        ball_position = tracks["ball"][frame_num][1].get("position", None)
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            current_team = tracks["players"][frame_num][assigned_player]["team"]
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(current_team)

            # Count passes with improved accuracy, passing frame number
            pass_counter.count_passes(assigned_player, current_team, frame_num)

            # Enhanced goal detection using field keypoints
            if ball_position:
                # Update field keypoints for current frame
                goal_detector.update_keypoints(video_frames[frame_num])

                # Detect goals using enhanced system
                goal_event = goal_detector.detect_goal(
                    ball_position, assigned_player, current_team, frame_num
                )

                # Also use the old system for comparison (optional)
                pass_counter.detect_goal(
                    ball_position, assigned_player, current_team, frame_num
                )

            # Detect tackles and interceptions
            tackle_counter.detect_tackles_and_interceptions(
                player_track, assigned_player, current_team, frame_num
            )
        else:
            # No player has the ball, pass -1 to indicate this
            # Use None for team to avoid KeyError
            pass_counter.count_passes(-1, None, frame_num)
            tackle_counter.detect_tackles_and_interceptions(
                player_track, -1, None, frame_num
            )

            # Still check for goals even if no player has the ball
            if ball_position:
                pass_counter.detect_goal(ball_position, -1, None, frame_num)

            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1 if no prior control

    team_ball_control = np.array(team_ball_control)

    # Get enhanced goal statistics first
    enhanced_goal_stats = goal_detector.get_goal_statistics()

    # Calculate final goal statistics using priority system
    final_team_goals, final_player_goals = calculate_final_goal_stats(
        pass_counter, enhanced_goal_stats, manual_goals
    )

    # Update goal detector with final counts for consistency
    goal_detector.set_final_goal_counts(final_team_goals, final_player_goals)

    # Export consolidated statistics to exactly two CSV files
    video_name = Path(input_video_path).stem
    team_csv_path, player_csv_path = export_consolidated_goal_statistics(
        video_name,
        pass_counter,
        enhanced_goal_stats,
        final_team_goals,
        final_player_goals,
        tackle_counter,
    )

    # Convert final player goals to simple count dictionary for compatibility
    final_player_goal_counts = {
        pid: data.get("goals", 0) for pid, data in final_player_goals.items()
    }

    # Apply manual goals if provided (for display purposes in video)
    if manual_goals:
        print(f"\n🎯 Applying {len(manual_goals)} manual goals for video display...")
        # Apply manual goals to pass counter for display in video
        for goal in manual_goals:
            pass_counter.add_manual_goal(
                team=goal["team"],
                player_id=goal["player_id"],
                frame_num=goal["frame_num"],
            )

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw Camera movement (only if enabled)
    if enable_camera_movement and camera_movement_estimator is not None:
        output_video_frames = camera_movement_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame
        )

    ## Draw Speed and Distance (only if enabled)
    if enable_speed_distance and speed_and_distance_estimator is not None:
        speed_and_distance_estimator.draw_speed_and_distance(
            output_video_frames, tracks
        )

    ## Draw Pass Counts
    output_video_frames = pass_counter.draw_pass_counts(output_video_frames)

    ## Draw Goal Counts
    output_video_frames = pass_counter.draw_goal_counts(output_video_frames)

    ## Draw Enhanced Goal Detection Information
    for frame_num in range(len(output_video_frames)):
        output_video_frames[frame_num] = goal_detector.draw_goal_info(
            output_video_frames[frame_num]
        )

    # Print final statistics for debugging
    print(
        f"Final pass counts: Team 1: {pass_counter.team_passes.get(1, 0)}, Team 2: {pass_counter.team_passes.get(2, 0)}"
    )
    print(
        f"Final goal counts: Team 1: {pass_counter.team_goals.get(1, 0)}, Team 2: {pass_counter.team_goals.get(2, 0)}"
    )

    # Print enhanced goal detection statistics
    print(
        f"Enhanced goal counts: Team 1: {enhanced_goal_stats['team_goals'].get(1, 0)}, Team 2: {enhanced_goal_stats['team_goals'].get(2, 0)}"
    )
    print(
        f"Total goals detected by enhanced system: {enhanced_goal_stats['total_goals']}"
    )

    if enhanced_goal_stats["goal_events"]:
        print("Goal events detected:")
        for i, event in enumerate(enhanced_goal_stats["goal_events"], 1):
            print(
                f"  Goal {i}: Team {event['team']} at frame {event['frame_num']} ({event['goal_side']} goal)"
            )
            if event["player_id"] != -1:
                print(f"    Scored by Player {event['player_id']}")
            print(f"    Ball position: {event['ball_position']}")

    # Print player statistics
    print("Player statistics:")
    for player_id in set(pass_counter.player_passes.keys()) | set(
        pass_counter.player_goals.keys()
    ):
        passes = pass_counter.player_passes.get(player_id, {}).get("passes", 0)
        goals = pass_counter.player_goals.get(player_id, {}).get("goals", 0)
        team = pass_counter.player_passes.get(player_id, {}).get(
            "team"
        ) or pass_counter.player_goals.get(player_id, {}).get("team", "Unknown")
        print(f"Player {player_id}: {passes} passes, {goals} goals (Team {team})")

    # Save video
    save_video(output_video_frames, output_video_path)
    print(f"\n🎬 Output video saved to: {output_video_path}")
    print(f"📊 Team statistics saved to: {team_csv_path}")
    print(f"📊 Player statistics saved to: {player_csv_path}")

    # Print final summary
    print(f"\n🏆 FINAL GOAL SUMMARY:")
    print(f"   Team 1: {final_team_goals[1]} goals")
    print(f"   Team 2: {final_team_goals[2]} goals")
    print(f"   Total players who scored: {len(final_player_goals)}")

    if enhanced_goal_stats.get("average_confidence", 0) > 0:
        print(
            f"   Average detection confidence: {enhanced_goal_stats['average_confidence']:.2f}"
        )
        print(f"   Detection accuracy: {enhanced_goal_stats['detection_accuracy']:.2f}")

    print(f"\n📈 ANALYSIS COMPLETE - Check CSV files for detailed statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football video analysis tool")
    parser.add_argument("--input", "-i", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the output video (optional)",
    )
    parser.add_argument(
        "--no-stubs",
        action="store_true",
        help="Don't use stub files for faster processing",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of stub files even if they exist",
    )
    parser.add_argument(
        "--goals-config",
        type=str,
        default=None,
        help="Path to a CSV file with manual goal information",
    )
    parser.add_argument(
        "--create-goals-template",
        type=str,
        default=None,
        help="Create a template goals configuration file at the specified path",
    )
    parser.add_argument(
        "--enable-camera-movement",
        action="store_true",
        help="Enable camera movement estimation (disabled by default for memory optimization)",
    )
    parser.add_argument(
        "--enable-speed-distance",
        action="store_true",
        help="Enable speed and distance estimation (disabled by default for memory optimization)",
    )

    args = parser.parse_args()

    # Create goals template if requested
    if args.create_goals_template:
        from utils.goal_utils import create_goals_template

        create_goals_template(args.create_goals_template)
        exit(0)

    # Ensure input is provided for normal operation
    if not args.input:
        parser.error(
            "the --input argument is required unless --create-goals-template is used"
        )

    main(
        input_video_path=args.input,
        output_video_path=args.output,
        use_stubs=not args.no_stubs,
        force_regenerate=args.force_regenerate,
        goals_config=args.goals_config,
        enable_camera_movement=args.enable_camera_movement,
        enable_speed_distance=args.enable_speed_distance,
    )
