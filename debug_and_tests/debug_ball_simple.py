#!/usr/bin/env python3
"""
Simple debug to check ball position calculation step by step.
"""

import os
import sys

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers import Tracker
from utils import get_center_of_bbox, read_video


def debug_ball_step_by_step():
    """Debug ball tracking step by step."""
    print("🔍 Step-by-step ball tracking debug")

    # Read video
    video_frames = read_video("../input_videos/goal_test_1.mp4")
    print(f"📹 Loaded {len(video_frames)} frames")

    # Initialize tracker
    tracker = Tracker("../models/best.pt")

    # Get tracks for first 20 frames only
    debug_frames = video_frames[:20]
    tracks = tracker.get_object_tracks(
        debug_frames, read_from_stub=False, stub_path=None
    )

    print(f"\n📊 Raw ball tracks (before position calculation):")
    for frame_num, ball_track in enumerate(tracks["ball"][:10]):
        print(f"Frame {frame_num}: {ball_track}")

    print(f"\n🔧 Adding positions to tracks...")
    tracker.add_position_to_tracks(tracks)

    print(f"\n📊 Ball tracks after position calculation:")
    for frame_num, ball_track in enumerate(tracks["ball"][:10]):
        if 1 in ball_track and ball_track[1]:
            bbox = ball_track[1].get("bbox", None)
            position = ball_track[1].get("position", None)
            print(f"Frame {frame_num}: bbox={bbox}, position={position}")

            # Manual position calculation for verification
            if bbox:
                manual_position = get_center_of_bbox(bbox)
                print(f"  Manual calculation: {manual_position}")
        else:
            print(f"Frame {frame_num}: No ball detected")

    print(f"\n🔄 Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print(f"\n📊 Ball tracks after interpolation:")
    for frame_num, ball_track in enumerate(tracks["ball"][:10]):
        if 1 in ball_track and ball_track[1]:
            bbox = ball_track[1].get("bbox", None)
            position = ball_track[1].get("position", None)
            print(f"Frame {frame_num}: bbox={bbox}, position={position}")
        else:
            print(f"Frame {frame_num}: No ball data")


if __name__ == "__main__":
    debug_ball_step_by_step()
