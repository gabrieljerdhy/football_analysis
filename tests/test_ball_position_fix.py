#!/usr/bin/env python3
"""
Test script to fix ball position calculation issues.
"""

import os
import sys
import pickle
import numpy as np

# Add src to path
sys.path.append('src')

from trackers import Tracker
from utils import get_center_of_bbox
from utils.goal_utils import load_manual_goals

def test_ball_position_fix():
    """Test and fix ball position calculation."""
    print("🔧 TESTING BALL POSITION CALCULATION")
    print("=" * 50)
    
    # Load tracks
    tracks_path = "data/stubs/videoplayback_process_tracks.pkl"
    if not os.path.exists(tracks_path):
        print(f"❌ Tracks file not found: {tracks_path}")
        return
    
    print(f"📂 Loading tracks from {tracks_path}...")
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    
    # Load manual goals
    manual_goals = load_manual_goals("data/manual_goals_videoplayback_process.csv")
    
    print(f"\n🔍 Analyzing ball tracks before position calculation...")
    
    # Check ball tracks at goal frames before position calculation
    for goal in manual_goals[:2]:  # Check first 2 goals
        frame_num = goal['frame_num']
        if frame_num < len(tracks['ball']):
            ball_track = tracks['ball'][frame_num]
            if ball_track and 1 in ball_track:
                ball_info = ball_track[1]
                bbox = ball_info.get('bbox')
                position = ball_info.get('position')
                print(f"Frame {frame_num} BEFORE: bbox={bbox}, position={position}")
                
                # Manual position calculation
                if bbox:
                    manual_pos = get_center_of_bbox(bbox)
                    print(f"  Manual calculation: {manual_pos}")
    
    # Initialize tracker to use its methods
    tracker = Tracker(
        "data/models/best_player_detect.pt",
        enable_jersey_detection=False,
        ball_model_path="data/models/best_ball_latest.pt",
        enable_enhanced_ball_detection=True,
    )
    
    print(f"\n🔧 Adding positions to tracks...")
    tracker.add_position_to_tracks(tracks)
    
    print(f"\n🔍 Analyzing ball tracks after position calculation...")
    
    # Check ball tracks at goal frames after position calculation
    for goal in manual_goals[:2]:  # Check first 2 goals
        frame_num = goal['frame_num']
        if frame_num < len(tracks['ball']):
            ball_track = tracks['ball'][frame_num]
            if ball_track and 1 in ball_track:
                ball_info = ball_track[1]
                bbox = ball_info.get('bbox')
                position = ball_info.get('position')
                print(f"Frame {frame_num} AFTER: bbox={bbox}, position={position}")
    
    print(f"\n⚽ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    print(f"\n🔍 Analyzing ball tracks after interpolation...")
    
    # Check ball tracks at goal frames after interpolation
    for goal in manual_goals[:2]:  # Check first 2 goals
        frame_num = goal['frame_num']
        if frame_num < len(tracks['ball']):
            ball_track = tracks['ball'][frame_num]
            if ball_track and 1 in ball_track:
                ball_info = ball_track[1]
                bbox = ball_info.get('bbox')
                position = ball_info.get('position')
                confidence = ball_info.get('confidence', 0.0)
                source = ball_info.get('source', 'unknown')
                print(f"Frame {frame_num} FINAL: bbox={bbox}, position={position}")
                print(f"  Confidence: {confidence}, Source: {source}")
    
    # Test goal detection with fixed positions
    print(f"\n🥅 Testing goal detection with fixed positions...")
    
    from goal_detection import FieldKeypointsDetector, GoalDetector
    from goal_detection.enhanced_goal_detector import EnhancedGoalDetector
    
    try:
        field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
        goal_detector = GoalDetector(field_keypoints_detector)
        enhanced_goal_detector = EnhancedGoalDetector(field_keypoints_detector)
        
        # Create a test frame for keypoints detection
        import cv2
        video_path = "data/input_videos/videoplayback_process.mp4"
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, test_frame = cap.read()
            cap.release()
            
            if ret:
                # Update keypoints
                goal_detector.update_keypoints(test_frame)
                enhanced_goal_detector.update_keypoints(test_frame)
                
                # Test goal detection at manual goal frames
                for goal in manual_goals[:2]:
                    frame_num = goal['frame_num']
                    if frame_num < len(tracks['ball']):
                        ball_track = tracks['ball'][frame_num]
                        if ball_track and 1 in ball_track:
                            ball_info = ball_track[1]
                            ball_position = ball_info.get('position')
                            
                            if ball_position:
                                print(f"\n🎯 Testing goal detection at frame {frame_num}:")
                                print(f"   Ball position: {ball_position}")
                                
                                # Test original goal detector
                                goal_side_original = goal_detector._check_ball_in_goal(ball_position)
                                print(f"   Original detector: {goal_side_original}")
                                
                                # Test enhanced goal detector
                                goal_side_enhanced = enhanced_goal_detector._detect_goal_fusion(ball_position)
                                print(f"   Enhanced detector: {goal_side_enhanced}")
                                
                                # Test keypoints directly
                                keypoint_result = field_keypoints_detector.is_ball_in_goal_area(ball_position)
                                print(f"   Keypoints detector: {keypoint_result}")
                                
                                # Get goal areas
                                goal_areas = field_keypoints_detector.get_goal_areas()
                                if goal_areas:
                                    print(f"   Available goal areas: {list(goal_areas.keys())}")
                                    for side, area in goal_areas.items():
                                        print(f"     {side}: {area}")
                                else:
                                    print(f"   ❌ No goal areas detected")
                            else:
                                print(f"❌ No ball position at frame {frame_num}")
    
    except Exception as e:
        print(f"❌ Error testing goal detection: {e}")
    
    print(f"\n✅ Ball position test complete!")

if __name__ == "__main__":
    test_ball_position_fix()
