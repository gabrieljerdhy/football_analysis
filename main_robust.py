import argparse
import os

import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from pass_counter.pass_counter import PassCounter
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer


def process_video(
    input_path,
    output_path,
    model_path,
    use_stubs=True,
    confidence=0.1,
    resize_factor=1.0,
):
    print(f"Processing video: {input_path}")
    print(f"Using model: {model_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read Video
    video_frames = read_video(input_path)
    print(f"Loaded {len(video_frames)} frames")

    if len(video_frames) == 0:
        print("ERROR: No frames were loaded from the video!")
        return

    # Get original frame dimensions
    original_height, original_width = video_frames[0].shape[:2]
    print(f"Original frame dimensions: {original_width}x{original_height}")

    # Resize frames if needed to save memory
    if resize_factor != 1.0:
        print(f"Resizing frames by factor {resize_factor}")
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)
        resized_frames = []
        for frame in video_frames:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            resized_frames.append(resized_frame)
        video_frames = resized_frames
        print(f"Resized frame dimensions: {new_width}x{new_height}")

    # Initialize Tracker with custom confidence
    tracker = Tracker(model_path, confidence=confidence)
    print("Initialized tracker")

    # Create stubs directory if it doesn't exist
    os.makedirs("stubs", exist_ok=True)

    # Generate stub paths based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    track_stub_path = f"stubs/track_stubs_{base_name}.pkl"
    camera_stub_path = f"stubs/camera_movement_stub_{base_name}.pkl"

    # Get object tracks
    print("Getting object tracks...")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=use_stubs, stub_path=track_stub_path
    )

    # Check if we have any player tracks
    if len(tracks["players"]) == 0:
        print("ERROR: No players detected in the video!")
        return

    print(f"Got tracks for {len(tracks['players'])} frames")

    # Ensure we have track data for all frames
    max_frames = min(len(video_frames), len(tracks.get("players", [])))
    print(f"Will process {max_frames} frames")

    # Get object positions
    tracker.add_position_to_tracks(tracks)
    print("Added positions to tracks")

    # Camera movement estimator
    print("Estimating camera movement...")
    try:
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames, read_from_stub=use_stubs, stub_path=camera_stub_path
        )
        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement_per_frame
        )
    except Exception as e:
        print(f"WARNING: Camera movement estimation failed: {e}")
        print("Continuing without camera movement estimation")
        # Create a dummy camera movement estimator
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        # Create a dummy camera movement array
        camera_movement_per_frame = [[0, 0]] * len(video_frames)

    # View Transformer
    print("Transforming view...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    print("Interpolating ball positions...")
    if len(tracks["ball"]) > 0:
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    else:
        print("WARNING: No ball detected in the video!")

    # Speed and distance estimator
    print("Estimating speed and distance...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    if len(tracks["players"]) > 0:
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num in range(min(len(tracks["players"]), len(video_frames))):
            player_track = tracks["players"][frame_num]
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = (
                    team_assigner.team_colors[team]
                )

    # Initialize pass counter
    print("Counting passes...")
    pass_counter = PassCounter()
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # Process each frame - with error handling for index mismatches
    for frame_num in range(max_frames):
        player_track = tracks["players"][frame_num]

        # Check if we have ball data for this frame
        if (
            "ball" not in tracks
            or frame_num >= len(tracks["ball"])
            or 1 not in tracks["ball"][frame_num]
        ):
            # No ball data for this frame, just continue
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            continue

        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            current_team = tracks["players"][frame_num][assigned_player]["team"]
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(current_team)

            # Count passes with improved accuracy, passing frame number
            pass_counter.count_passes(assigned_player, current_team, frame_num)
        else:
            # No player has the ball, pass -1 to indicate this
            # Use None for team to avoid KeyError
            pass_counter.count_passes(-1, None, frame_num)

            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1 if no prior control

    # Ensure team_ball_control has data for all frames
    if len(team_ball_control) < max_frames:
        last_control = team_ball_control[-1] if team_ball_control else 1
        team_ball_control.extend([last_control] * (max_frames - len(team_ball_control)))

    team_ball_control = np.array(team_ball_control)

    # Draw output
    print("Drawing annotations...")
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(
        video_frames[:max_frames], tracks, team_ball_control
    )

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame[: len(output_video_frames)]
    )

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    ## Draw Pass Counts - this will now show incremental progress
    if (
        hasattr(pass_counter, "pass_counts_per_frame")
        and len(pass_counter.pass_counts_per_frame) > 0
    ):
        output_video_frames = pass_counter.draw_pass_counts(output_video_frames)

    # Print final pass counts for debugging
    print(
        f"Final pass counts: Team 1: {pass_counter.team_passes.get(1, 0)}, Team 2: {pass_counter.team_passes.get(2, 0)}"
    )

    # Save video
    print(f"Saving video to {output_path}")
    save_video(output_video_frames, output_path)
    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process football video with AI analysis"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input_videos/demo_vid_2.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_videos/output_video.avi",
        help="Output video path",
    )
    parser.add_argument(
        "--model", type=str, default="models/best.pt", help="YOLO model path"
    )
    parser.add_argument(
        "--no-stubs", action="store_true", help="Do not use or create stub files"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.1, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=0.5,
        help="Resize factor for input frames (0.5 = half size)",
    )

    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        args.model,
        use_stubs=not args.no_stubs,
        confidence=args.confidence,
        resize_factor=args.resize,
    )
