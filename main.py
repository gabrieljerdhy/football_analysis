import argparse
import csv
import os
import pickle
import signal
import sys
import time
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

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    print(f"\n⚠️  Received signal {signum}. Requesting graceful shutdown...")
    print("⏳ Finishing current batch and saving progress...")
    shutdown_requested = True


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def main(
    input_video_path,
    output_video_path=None,
    use_stubs=True,
    force_regenerate=False,
    goals_config=None,
    enable_camera_movement=False,
    enable_speed_distance=False,
    memory_efficient=False,
    batch_size=50,
    memory_limit=8.0,
    upload_to_spaces=False,
    spaces_access_key_id=None,
    spaces_secret_access_key=None,
    spaces_bucket=None,
    spaces_region="nyc3",
    spaces_folder_prefix="football_analysis",
    upload_csv_only=False,
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

    # Check if we should use memory-efficient processing
    if memory_efficient:
        print("🚀 MEMORY-EFFICIENT FOOTBALL VIDEO ANALYSIS")
        print("=" * 60)

        from utils import (
            VideoFrameIterator,
            cleanup_memory,
            get_video_info,
            monitor_memory_usage,
        )

        # Get video information and check memory requirements
        video_info = get_video_info(input_video_path)
        estimated_memory_gb = (
            video_info["width"] * video_info["height"] * 3 * video_info["total_frames"]
        ) / (1024**3)

        estimated_memory_per_frame = (
            video_info["width"] * video_info["height"] * 3
        ) / (1024**3)
        estimated_batch_memory = estimated_memory_per_frame * batch_size

        print(
            f"📹 Video: {video_info['total_frames']:,} frames, {video_info['duration_seconds']/60:.1f} min"
        )
        print(f"🧮 Video would need {estimated_memory_gb:.1f} GB if fully loaded")
        print(
            f"🧮 Estimated memory per batch ({batch_size} frames): {estimated_batch_memory:.2f} GB"
        )
        print(f"💾 Current memory usage: {monitor_memory_usage():.2f} GB")

        # Auto-adjust batch size if needed (use 50% of available memory as safety margin)
        if estimated_batch_memory > memory_limit * 0.5:
            new_batch_size = max(
                1, int((memory_limit * 0.5) / estimated_memory_per_frame)
            )
            print(
                f"⚠️  Reducing batch size from {batch_size} to {new_batch_size} to fit memory"
            )
            batch_size = new_batch_size

        # Initialize DigitalOcean Spaces uploader if requested (for memory-efficient mode)
        uploader = None
        if upload_to_spaces:
            from utils.storage_utils import (
                create_uploader_from_args,
                create_uploader_from_env,
            )

            # Try to create uploader from provided arguments first
            if spaces_access_key_id and spaces_secret_access_key:
                uploader = create_uploader_from_args(
                    access_key_id=spaces_access_key_id,
                    secret_access_key=spaces_secret_access_key,
                    region=spaces_region,
                    bucket_name=spaces_bucket,
                )
            else:
                # Fall back to environment variables
                uploader = create_uploader_from_env()

            # Get bucket name from uploader if not provided via command line
            if uploader and not spaces_bucket:
                spaces_bucket = uploader.bucket_name

            if uploader and spaces_bucket:
                # Test connection
                if not uploader.test_connection(spaces_bucket):
                    print(
                        "❌ Failed to connect to DigitalOcean Spaces. Disabling upload."
                    )
                    uploader = None
            elif uploader and not spaces_bucket:
                print("❌ No bucket name provided. Disabling upload.")
                uploader = None

        # Process using memory-efficient approach
        tracks, team_ball_control, camera_movement_per_frame = (
            process_video_memory_efficient(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                use_stubs=use_stubs,
                force_regenerate=force_regenerate,
                goals_config=goals_config,
                enable_camera_movement=enable_camera_movement,
                enable_speed_distance=enable_speed_distance,
                batch_size=batch_size,
                video_info=video_info,
                uploader=uploader,
                spaces_bucket=spaces_bucket,
                spaces_folder_prefix=spaces_folder_prefix,
            )
        )

        # Generate output video in memory-efficient way
        print("\n🎬 Generating output video...")
        generate_output_video_memory_efficient(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            tracks=tracks,
            team_ball_control=team_ball_control,
            camera_movement_per_frame=camera_movement_per_frame,
            enable_camera_movement=enable_camera_movement,
            enable_speed_distance=enable_speed_distance,
            batch_size=batch_size,
        )

        # Upload video to DigitalOcean Spaces if requested and uploader is available
        if uploader and not upload_csv_only:
            print(f"\n☁️  Uploading output video to DigitalOcean Spaces...")
            video_name = Path(input_video_path).stem
            video_upload_success = uploader.upload_video(
                video_path=output_video_path,
                video_name=video_name,
                bucket_name=spaces_bucket,
                folder_prefix=spaces_folder_prefix,
            )

            if video_upload_success:
                print(f"✅ Output video uploaded successfully")
            else:
                print(f"❌ Failed to upload output video")
        elif upload_to_spaces and upload_csv_only:
            print(f"\n⏭️  Skipping video upload (--upload-csv-only specified)")

        print(f"✅ Memory-efficient processing completed!")
        print(f"🎬 Output video saved to: {output_video_path}")
        print(f"💾 Final memory usage: {monitor_memory_usage():.2f} GB")
        return

    # Original approach - Read all video frames into memory
    print("⚠️  Using original approach - loading all frames into memory")
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

    # Initialize DigitalOcean Spaces uploader if requested
    uploader = None
    if upload_to_spaces:
        from utils.storage_utils import (
            create_uploader_from_args,
            create_uploader_from_env,
        )

        # Try to create uploader from provided arguments first
        if spaces_access_key_id and spaces_secret_access_key:
            uploader = create_uploader_from_args(
                access_key_id=spaces_access_key_id,
                secret_access_key=spaces_secret_access_key,
                region=spaces_region,
                bucket_name=spaces_bucket,
            )
        else:
            # Fall back to environment variables
            uploader = create_uploader_from_env()

        # Get bucket name from uploader if not provided via command line
        if uploader and not spaces_bucket:
            spaces_bucket = uploader.bucket_name

        if uploader and spaces_bucket:
            # Test connection
            if not uploader.test_connection(spaces_bucket):
                print("❌ Failed to connect to DigitalOcean Spaces. Disabling upload.")
                uploader = None
        elif uploader and not spaces_bucket:
            print("❌ No bucket name provided. Disabling upload.")
            uploader = None

    # Export consolidated statistics to exactly two CSV files
    video_name = Path(input_video_path).stem
    team_csv_path, player_csv_path = export_consolidated_goal_statistics(
        video_name,
        pass_counter,
        enhanced_goal_stats,
        final_team_goals,
        final_player_goals,
        tackle_counter,
        uploader=uploader,
        bucket_name=spaces_bucket,
        upload_folder_prefix=spaces_folder_prefix,
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

    # Upload video to DigitalOcean Spaces if requested and uploader is available
    if uploader and not upload_csv_only:
        print(f"\n☁️  Uploading output video to DigitalOcean Spaces...")
        video_upload_success = uploader.upload_video(
            video_path=output_video_path,
            video_name=video_name,
            bucket_name=spaces_bucket,
            folder_prefix=spaces_folder_prefix,
        )

        if video_upload_success:
            print(f"✅ Output video uploaded successfully")
        else:
            print(f"❌ Failed to upload output video")
    elif upload_to_spaces and upload_csv_only:
        print(f"\n⏭️  Skipping video upload (--upload-csv-only specified)")

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


def process_video_memory_efficient(
    input_video_path,
    output_video_path,
    use_stubs,
    force_regenerate,
    goals_config,
    enable_camera_movement,
    enable_speed_distance,
    batch_size,
    video_info,
    uploader=None,
    spaces_bucket=None,
    spaces_folder_prefix="football_analysis",
):
    """
    Memory-efficient video processing that processes frames in batches.
    """
    from utils import (
        VideoFrameIterator,
        cleanup_memory,
        monitor_memory_usage,
        save_video_streaming,
    )

    # Generate stub paths based on input video name
    video_name = Path(input_video_path).stem
    tracks_stub_path = f"stubs/{video_name}_tracks.pkl"
    camera_movement_stub_path = f"stubs/{video_name}_camera_movement.pkl"

    print("\n🔧 Initializing components...")

    # Initialize Tracker with jersey number detection
    tracker = Tracker("models/best.pt", enable_jersey_detection=True)

    # Initialize Goal Detection System
    field_keypoints_detector = FieldKeypointsDetector("models/best_fk.pt")
    goal_detector = GoalDetector(field_keypoints_detector)

    # Load manual goals if provided
    manual_goals = load_manual_goals(goals_config)

    # Process tracking in batches if not using stubs
    tracks = None
    read_tracks_from_stub = (
        use_stubs and os.path.exists(tracks_stub_path) and not force_regenerate
    )

    if read_tracks_from_stub:
        print("📂 Loading tracks from stub...")
        with open(tracks_stub_path, "rb") as f:
            tracks = pickle.load(f)
        print(f"✅ Loaded tracks for {len(tracks['players'])} frames")
    else:
        print("🔍 Processing object tracking in batches...")
        tracks = process_tracking_in_batches(
            input_video_path, tracker, batch_size, tracks_stub_path, video_info
        )

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Add jersey numbers to tracks using OCR (process in batches)
    print("🔢 Adding jersey numbers...")
    add_jersey_numbers_in_batches(input_video_path, tracker, tracks, batch_size)

    # Camera movement estimation (optional)
    camera_movement_per_frame = []
    if enable_camera_movement:
        print("🎥 Camera movement estimation enabled")
        camera_movement_per_frame = process_camera_movement_in_batches(
            input_video_path,
            batch_size,
            camera_movement_stub_path,
            use_stubs,
            force_regenerate,
            video_info,
        )

        # Apply camera movement adjustments
        camera_movement_estimator = CameraMovementEstimator(
            None
        )  # Will be initialized with first frame
        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement_per_frame
        )
    else:
        print("🎥 Camera movement estimation disabled (memory optimization)")

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator (optional)
    if enable_speed_distance:
        print("🏃 Speed and distance estimation enabled")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    else:
        print("🏃 Speed and distance estimation disabled (memory optimization)")

    # Process team assignment and ball tracking
    print("\n🔄 Processing team assignment and ball tracking...")
    team_ball_control = process_team_assignment_and_ball_tracking(
        input_video_path,
        tracks,
        batch_size,
        goal_detector,
        manual_goals,
        uploader,
        spaces_bucket,
        spaces_folder_prefix,
    )

    return tracks, team_ball_control, camera_movement_per_frame


def process_tracking_in_batches(
    input_video_path, tracker, batch_size, stub_path, video_info
):
    """Process object tracking in batches to avoid memory issues."""
    from utils import VideoFrameIterator, cleanup_memory, monitor_memory_usage

    print(f"🔍 Processing tracking in batches of {batch_size} frames...")

    all_tracks = {"players": [], "referees": [], "ball": []}
    frames_processed = 0

    with VideoFrameIterator(input_video_path, batch_size) as frame_iterator:
        for batch_frames in frame_iterator:
            print(
                f"🔄 Processing frames {frames_processed}-{frames_processed + len(batch_frames)}"
            )

            # Process this batch
            batch_tracks = tracker.get_object_tracks(
                batch_frames, read_from_stub=False, stub_path=None
            )

            # Append to overall tracks
            all_tracks["players"].extend(batch_tracks["players"])
            all_tracks["referees"].extend(batch_tracks["referees"])
            all_tracks["ball"].extend(batch_tracks["ball"])

            frames_processed += len(batch_frames)

            # Memory cleanup
            cleanup_memory()

            # Progress update
            progress = (frames_processed / video_info["total_frames"]) * 100
            memory_usage = monitor_memory_usage()
            print(f"📊 Progress: {progress:.1f}% | Memory: {memory_usage:.2f} GB")

    # Save tracks to stub
    print(f"💾 Saving tracks to {stub_path}...")
    with open(stub_path, "wb") as f:
        pickle.dump(all_tracks, f)

    print(f"✅ Tracking complete: {len(all_tracks['players'])} frames processed")
    return all_tracks


def add_jersey_numbers_in_batches(input_video_path, tracker, tracks, batch_size):
    """Add jersey numbers to tracks in batches to save memory."""
    from utils import VideoFrameIterator, cleanup_memory, monitor_memory_usage

    print(f"🔢 Adding jersey numbers in batches of {batch_size} frames...")

    frame_num = 0
    with VideoFrameIterator(input_video_path, batch_size) as frame_iterator:
        for batch_frames in frame_iterator:
            # Create a subset of tracks for this batch
            batch_tracks = {"players": [], "referees": [], "ball": []}

            for i, frame in enumerate(batch_frames):
                if frame_num + i < len(tracks["players"]):
                    batch_tracks["players"].append(tracks["players"][frame_num + i])
                    batch_tracks["referees"].append(tracks["referees"][frame_num + i])
                    batch_tracks["ball"].append(tracks["ball"][frame_num + i])

            # Add jersey numbers for this batch
            tracker.add_jersey_numbers_to_tracks(
                batch_tracks, batch_frames, frame_sampling=5
            )

            # Update the main tracks with jersey numbers
            for i, frame in enumerate(batch_frames):
                if frame_num + i < len(tracks["players"]):
                    tracks["players"][frame_num + i] = batch_tracks["players"][i]

            frame_num += len(batch_frames)
            cleanup_memory()

            if frame_num >= len(tracks["players"]):
                break

    print(f"✅ Jersey numbers added for {frame_num} frames")


def process_camera_movement_in_batches(
    input_video_path, batch_size, stub_path, use_stubs, force_regenerate, video_info
):
    """Process camera movement estimation in batches."""
    from utils import VideoFrameIterator, cleanup_memory, monitor_memory_usage

    read_camera_from_stub = (
        use_stubs and os.path.exists(stub_path) and not force_regenerate
    )

    if read_camera_from_stub:
        print("📂 Loading camera movement from stub...")
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    print(f"🎥 Processing camera movement in batches of {batch_size} frames...")

    # Get first frame for initialization
    cap = cv2.VideoCapture(input_video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not read first frame for camera movement estimation")
        return []

    camera_movement_estimator = CameraMovementEstimator(first_frame)
    all_camera_movement = []
    frames_processed = 0

    with VideoFrameIterator(input_video_path, batch_size) as frame_iterator:
        for batch_frames in frame_iterator:
            print(
                f"🎥 Processing camera movement for frames {frames_processed}-{frames_processed + len(batch_frames)}"
            )

            # Process camera movement for this batch
            batch_movement = camera_movement_estimator.get_camera_movement(
                batch_frames, read_from_stub=False
            )

            all_camera_movement.extend(batch_movement)
            frames_processed += len(batch_frames)

            # Memory cleanup
            cleanup_memory()

            # Progress update
            progress = (frames_processed / video_info["total_frames"]) * 100
            memory_usage = monitor_memory_usage()
            print(
                f"📊 Camera Progress: {progress:.1f}% | Memory: {memory_usage:.2f} GB"
            )

    # Save camera movement to stub
    print(f"💾 Saving camera movement to {stub_path}...")
    with open(stub_path, "wb") as f:
        pickle.dump(all_camera_movement, f)

    print(f"✅ Camera movement complete: {len(all_camera_movement)} frames processed")
    return all_camera_movement


def process_team_assignment_and_ball_tracking(
    input_video_path,
    tracks,
    batch_size,
    goal_detector,
    manual_goals,
    uploader=None,
    spaces_bucket=None,
    spaces_folder_prefix="football_analysis",
):
    """Process team assignment and ball tracking in batches."""
    from utils import VideoFrameIterator, cleanup_memory, monitor_memory_usage

    # Team assignment - need to get a frame for color analysis
    team_assigner = TeamAssigner()

    # Find first frame with player detections
    first_frame_with_players = 0
    for i, player_frame in enumerate(tracks["players"]):
        if player_frame:  # If there are player detections in this frame
            first_frame_with_players = i
            break

    # Get the first frame to analyze team colors
    cap = cv2.VideoCapture(input_video_path)
    for _ in range(first_frame_with_players):
        cap.read()  # Skip to the frame we want
    ret, first_frame = cap.read()
    cap.release()

    if ret and first_frame_with_players < len(tracks["players"]):
        team_assigner.assign_team_color(
            first_frame, tracks["players"][first_frame_with_players]
        )
    else:
        # If no frames have player detections, use default colors
        print("⚠️  No player detections found, using default team colors")
        team_assigner.assign_team_color(first_frame if ret else None, {})

    # Process team assignment for each frame (optimized for large videos)
    print("🎨 Assigning teams to players...")
    total_frames = len(tracks["players"])

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    start_time = time.time()

    # For very large videos, use different optimization strategies
    if total_frames > 150000:  # More than ~83 minutes at 30fps - use ultra-fast mode
        print(
            f"⚡ Ultra-large video detected ({total_frames:,} frames). Using ultra-fast team assignment..."
        )
        # For ultra-large videos, use default team assignments to save time
        print(
            "🚀 Skipping detailed team assignment for ultra-large video - using default teams"
        )

        # Assign default teams to all players without frame analysis
        for frame_idx in range(total_frames):
            if tracks["players"][frame_idx]:
                for player_id, track in tracks["players"][frame_idx].items():
                    # Simple alternating team assignment based on player ID
                    team = 1 if int(player_id) % 2 == 0 else 2
                    tracks["players"][frame_idx][player_id]["team"] = team
                    tracks["players"][frame_idx][player_id]["team_color"] = (
                        team_assigner.team_colors.get(team, [0, 0, 255])
                    )

        print(f"✅ Ultra-fast team assignment completed for {total_frames:,} frames")
        frame_indices = []  # Skip the normal processing

    elif total_frames > 50000:  # More than ~28 minutes at 30fps - use optimized mode
        print(
            f"⚡ Large video detected ({total_frames:,} frames). Using optimized team assignment..."
        )
        # Sample every 20th frame for team assignment to speed up processing further
        sample_interval = 20
        frame_indices = list(range(0, total_frames, sample_interval))
        print(
            f"📊 Processing {len(frame_indices):,} sample frames for team assignment..."
        )
    else:
        frame_indices = list(range(total_frames))

    frame_num = 0
    processed_frames = 0

    # Only process frames if we have frame_indices (not in ultra-fast mode)
    if frame_indices:
        with VideoFrameIterator(input_video_path, batch_size=50) as frame_iterator:
            for batch_frames in frame_iterator:
                # Check for shutdown request
                if shutdown_requested:
                    print(
                        "⚠️  Shutdown requested during team assignment. Saving progress..."
                    )
                    break

                for frame in batch_frames:
                    if frame_num in frame_indices and frame_num < len(
                        tracks["players"]
                    ):
                        player_track = tracks["players"][frame_num]
                        for player_id, track in player_track.items():
                            team = team_assigner.get_player_team(
                                frame, track["bbox"], player_id
                            )
                            tracks["players"][frame_num][player_id]["team"] = team
                            tracks["players"][frame_num][player_id]["team_color"] = (
                                team_assigner.team_colors[team]
                            )
                        processed_frames += 1

                        # Progress update for large videos
                        if processed_frames % 1000 == 0:
                            progress = (processed_frames / len(frame_indices)) * 100
                            elapsed_time = time.time() - start_time
                            print(
                                f"🎨 Team assignment progress: {progress:.1f}% ({processed_frames:,}/{len(frame_indices):,} frames) - {elapsed_time:.1f}s elapsed"
                            )

                    frame_num += 1
                    if frame_num >= len(tracks["players"]):
                        break
                if frame_num >= len(tracks["players"]) or shutdown_requested:
                    break
                cleanup_memory()
    else:
        print("⏭️  Skipping frame-by-frame team assignment (ultra-fast mode)")

    # For large videos, propagate team assignments to non-sampled frames (skip for ultra-fast mode)
    if total_frames > 50000 and frame_indices:  # Only if not in ultra-fast mode
        print("🔄 Propagating team assignments to all frames...")
        for frame_idx in range(total_frames):
            if frame_idx not in frame_indices and tracks["players"][frame_idx]:
                # Find the nearest sampled frame
                nearest_sampled = min(frame_indices, key=lambda x: abs(x - frame_idx))

                # Copy team assignments from nearest sampled frame
                for player_id, track in tracks["players"][frame_idx].items():
                    if player_id in tracks["players"][nearest_sampled]:
                        source_team = tracks["players"][nearest_sampled][player_id].get(
                            "team", 1
                        )
                        source_color = tracks["players"][nearest_sampled][
                            player_id
                        ].get(
                            "team_color", team_assigner.team_colors.get(1, [0, 0, 255])
                        )
                        tracks["players"][frame_idx][player_id]["team"] = source_team
                        tracks["players"][frame_idx][player_id][
                            "team_color"
                        ] = source_color
                    else:
                        # Default assignment if player not found in sampled frame
                        tracks["players"][frame_idx][player_id]["team"] = 1
                        tracks["players"][frame_idx][player_id]["team_color"] = (
                            team_assigner.team_colors.get(1, [0, 0, 255])
                        )

            # Progress update
            if frame_idx % 10000 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"🔄 Propagation progress: {progress:.1f}%")
    elif total_frames > 150000:
        print("⏭️  Skipping propagation (ultra-fast mode - teams already assigned)")

    # Ball assignment and pass/goal detection (optimized for large videos)
    print("⚽ Processing ball assignment and game events...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # Initialize counters
    pass_counter = PassCounter()
    tackle_counter = TackleCounter()

    total_frames = len(tracks["players"])

    # For very large videos, optimize goal detection by reducing frame reads
    goal_detection_interval = 5 if total_frames > 50000 else 1
    if total_frames > 50000:
        print(
            f"⚡ Optimizing goal detection: checking every {goal_detection_interval} frames for large video"
        )

    for frame_num, player_track in enumerate(tracks["players"]):
        # Check for shutdown request
        if shutdown_requested:
            print("⚠️  Shutdown requested during ball tracking. Saving progress...")
            break

        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])
        ball_position = tracks["ball"][frame_num].get(1, {}).get("position", None)

        # Only assign ball if we have a valid bbox
        if ball_bbox and len(ball_bbox) == 4:
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
        else:
            assigned_player = -1

        if assigned_player != -1:
            current_team = tracks["players"][frame_num][assigned_player]["team"]
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(current_team)

            # Count passes with improved accuracy, passing frame number
            pass_counter.count_passes(assigned_player, current_team, frame_num)

            # Enhanced goal detection using field keypoints (optimized for large videos)
            if ball_position and frame_num % goal_detection_interval == 0:
                # Get current frame for goal detection (we need to read it again)
                cap = cv2.VideoCapture(input_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, current_frame = cap.read()
                cap.release()

                if ret:
                    # Update field keypoints for current frame
                    goal_detector.update_keypoints(current_frame)

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
            pass_counter.count_passes(-1, None, frame_num)
            tackle_counter.detect_tackles_and_interceptions(
                player_track, -1, None, frame_num
            )

            # Still check for goals even if no player has the ball (optimized)
            if ball_position and frame_num % goal_detection_interval == 0:
                pass_counter.detect_goal(ball_position, -1, None, frame_num)

            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        # Progress update for large videos
        if total_frames > 50000 and frame_num % 10000 == 0:
            progress = (frame_num / total_frames) * 100
            elapsed_time = time.time() - start_time
            print(
                f"⚽ Ball tracking progress: {progress:.1f}% ({frame_num:,}/{total_frames:,} frames) - {elapsed_time:.1f}s elapsed"
            )

    # Handle shutdown request - save partial results
    if shutdown_requested:
        print("⚠️  Processing interrupted. Saving partial results...")
        # Ensure we have some team ball control data
        if not team_ball_control:
            team_ball_control = [1] * min(1000, total_frames)  # Default data

    # Process final statistics and export CSV files
    team_ball_control = np.array(team_ball_control)

    # Get enhanced goal statistics
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
        uploader=uploader,
        bucket_name=spaces_bucket,
        upload_folder_prefix=spaces_folder_prefix,
    )

    print(f"📊 Team statistics saved to: {team_csv_path}")
    print(f"📊 Player statistics saved to: {player_csv_path}")

    return team_ball_control


def generate_output_video_memory_efficient(
    input_video_path,
    output_video_path,
    tracks,
    team_ball_control,
    camera_movement_per_frame,
    enable_camera_movement,
    enable_speed_distance,
    batch_size,
):
    """Generate output video with annotations in a memory-efficient way."""
    from utils import VideoFrameIterator, cleanup_memory, monitor_memory_usage

    print(f"🎬 Generating output video in batches of {batch_size} frames...")

    # Initialize tracker for drawing
    tracker = Tracker("models/best.pt", enable_jersey_detection=True)

    # Initialize other components if needed
    camera_movement_estimator = None
    speed_and_distance_estimator = None

    if enable_camera_movement and camera_movement_per_frame:
        # Get first frame for camera movement estimator initialization
        cap = cv2.VideoCapture(input_video_path)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            camera_movement_estimator = CameraMovementEstimator(first_frame)

    if enable_speed_distance:
        speed_and_distance_estimator = SpeedAndDistance_Estimator()

    # Initialize pass counter for drawing
    pass_counter = PassCounter()

    # Setup video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_num = 0
    total_frames = len(tracks["players"])

    with VideoFrameIterator(input_video_path, batch_size) as frame_iterator:
        for batch_frames in frame_iterator:
            print(
                f"🎬 Processing output frames {frame_num}-{frame_num + len(batch_frames)}"
            )

            # Process each frame in the batch
            for i, frame in enumerate(batch_frames):
                current_frame_num = frame_num + i

                if current_frame_num >= total_frames:
                    break

                # Draw annotations on frame manually (since single frame methods don't exist)
                annotated_frame = frame.copy()

                # Draw players
                player_dict = tracks["players"][current_frame_num]
                ball_dict = tracks["ball"][current_frame_num]
                referee_dict = tracks["referees"][current_frame_num]

                # Draw Players
                for track_id, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    jersey_number = player.get("jersey_number", track_id)
                    annotated_frame = tracker.draw_ellipse(
                        annotated_frame, player["bbox"], color, track_id, jersey_number
                    )

                    if player.get("has_ball", False):
                        annotated_frame = tracker.draw_traingle(
                            annotated_frame, player["bbox"], (0, 0, 255)
                        )

                # Draw Referee
                for _, referee in referee_dict.items():
                    annotated_frame = tracker.draw_ellipse(
                        annotated_frame, referee["bbox"], (0, 255, 255)
                    )

                # Draw ball
                for track_id, ball in ball_dict.items():
                    annotated_frame = tracker.draw_traingle(
                        annotated_frame, ball["bbox"], (0, 255, 0)
                    )

                # Draw team ball control
                if current_frame_num < len(team_ball_control):
                    team = team_ball_control[current_frame_num]
                    cv2.putText(
                        annotated_frame,
                        f"Ball Control: Team {team}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                # Draw camera movement if enabled
                if (
                    enable_camera_movement
                    and camera_movement_per_frame
                    and current_frame_num < len(camera_movement_per_frame)
                ):
                    x_movement, y_movement = camera_movement_per_frame[
                        current_frame_num
                    ]
                    cv2.rectangle(
                        annotated_frame, (0, 100), (500, 200), (255, 255, 255), -1
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Camera Movement X: {x_movement:.2f}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Camera Movement Y: {y_movement:.2f}",
                        (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3,
                    )

                # Draw speed and distance if enabled
                if enable_speed_distance:
                    for track_id, player in player_dict.items():
                        if "speed" in player and "distance" in player:
                            speed = player.get("speed", 0)
                            distance = player.get("distance", 0)
                            if speed is not None and distance is not None:
                                from utils.bbox_utils import get_foot_position

                                bbox = player["bbox"]
                                position = get_foot_position(bbox)
                                position = list(position)
                                position[1] += 40
                                position = tuple(map(int, position))
                                cv2.putText(
                                    annotated_frame,
                                    f"{speed:.2f} km/h",
                                    position,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    2,
                                )
                                cv2.putText(
                                    annotated_frame,
                                    f"{distance:.2f} m",
                                    (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    2,
                                )

                # Draw pass counts (simplified)
                cv2.putText(
                    annotated_frame,
                    f"Passes - Team 1: {pass_counter.team_passes.get(1, 0)} | Team 2: {pass_counter.team_passes.get(2, 0)}",
                    (50, annotated_frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # Draw goal counts (simplified)
                cv2.putText(
                    annotated_frame,
                    f"Goals - Team 1: {pass_counter.team_goals.get(1, 0)} | Team 2: {pass_counter.team_goals.get(2, 0)}",
                    (50, annotated_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # Write frame to output video
                out.write(annotated_frame)

            frame_num += len(batch_frames)
            cleanup_memory()

            # Progress update
            progress = (frame_num / total_frames) * 100
            memory_usage = monitor_memory_usage()
            print(f"📊 Video Progress: {progress:.1f}% | Memory: {memory_usage:.2f} GB")

            if frame_num >= total_frames:
                break

    out.release()
    print(f"✅ Output video generation complete: {frame_num} frames processed")


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
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Use memory-efficient processing for large videos (recommended for videos >30 minutes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for memory-efficient processing (default: 50)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=8.0,
        help="Memory limit in GB for automatic batch size adjustment (default: 8.0)",
    )
    parser.add_argument(
        "--check-video-info",
        action="store_true",
        help="Only check video information and memory requirements without processing",
    )

    # DigitalOcean Spaces storage arguments
    parser.add_argument(
        "--upload-to-spaces",
        action="store_true",
        help="Upload CSV files and output video to DigitalOcean Spaces",
    )
    parser.add_argument(
        "--spaces-access-key-id",
        type=str,
        help="DigitalOcean Spaces access key ID (or set DO_SPACES_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "--spaces-secret-access-key",
        type=str,
        help="DigitalOcean Spaces secret access key (or set DO_SPACES_SECRET_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "--spaces-bucket",
        type=str,
        help="DigitalOcean Spaces bucket name (or set DO_SPACES_BUCKET env var)",
    )
    parser.add_argument(
        "--spaces-region",
        type=str,
        default="nyc3",
        help="DigitalOcean Spaces region (default: nyc3)",
    )
    parser.add_argument(
        "--spaces-folder-prefix",
        type=str,
        default="football_analysis",
        help="Folder prefix for uploads in the bucket (default: football_analysis)",
    )
    parser.add_argument(
        "--upload-csv-only",
        action="store_true",
        help="Only upload CSV files, not the output video (faster)",
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

    # Check video info only if requested
    if args.check_video_info:
        from utils import get_video_info, monitor_memory_usage

        try:
            video_info = get_video_info(args.input)
            print("📹 VIDEO INFORMATION")
            print("=" * 50)
            print(f"File: {args.input}")
            print(f"Frames: {video_info['total_frames']:,}")
            print(f"Duration: {video_info['duration_seconds']/60:.1f} minutes")
            print(f"Resolution: {video_info['width']}x{video_info['height']}")
            print(f"FPS: {video_info['fps']:.1f}")

            # Memory estimates
            frame_size_mb = (video_info["width"] * video_info["height"] * 3) / (1024**2)
            batch_memory_gb = (frame_size_mb * args.batch_size) / 1024
            total_video_memory_gb = (frame_size_mb * video_info["total_frames"]) / 1024

            print(f"\n💾 MEMORY ESTIMATES")
            print(f"Per frame: {frame_size_mb:.1f} MB")
            print(f"Per batch ({args.batch_size} frames): {batch_memory_gb:.2f} GB")
            print(f"Full video (if loaded): {total_video_memory_gb:.1f} GB")
            print(f"Current system memory: {monitor_memory_usage():.2f} GB")
            print(f"Memory limit setting: {args.memory_limit:.1f} GB")

            # Recommendations
            if total_video_memory_gb > 10:
                print(f"\n💡 RECOMMENDATIONS")
                print(f"This is a large video ({total_video_memory_gb:.1f} GB)")
                print(f"Use --memory-efficient flag for processing")
                if batch_memory_gb > args.memory_limit * 0.7:
                    recommended_batch = max(
                        1, int((args.memory_limit * 0.5 * 1024) / frame_size_mb)
                    )
                    print(f"Consider reducing --batch-size to {recommended_batch}")

            print(f"\n✅ Video check complete. Add other flags to process the video.")

        except Exception as e:
            print(f"❌ Error reading video: {e}")
        import sys

        sys.exit(0)

    main(
        input_video_path=args.input,
        output_video_path=args.output,
        use_stubs=not args.no_stubs,
        force_regenerate=args.force_regenerate,
        goals_config=args.goals_config,
        enable_camera_movement=args.enable_camera_movement,
        enable_speed_distance=args.enable_speed_distance,
        memory_efficient=args.memory_efficient,
        batch_size=args.batch_size,
        memory_limit=args.memory_limit,
        upload_to_spaces=args.upload_to_spaces,
        spaces_access_key_id=args.spaces_access_key_id,
        spaces_secret_access_key=args.spaces_secret_access_key,
        spaces_bucket=args.spaces_bucket,
        spaces_region=args.spaces_region,
        spaces_folder_prefix=args.spaces_folder_prefix,
        upload_csv_only=args.upload_csv_only,
    )
