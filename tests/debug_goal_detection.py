#!/usr/bin/env python3
"""
Debug script to analyze goal detection issues.
This script will help identify why the system is detecting 1-1 instead of 4-0.
"""

import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

# Add src to path
sys.path.append('src')

from goal_detection import FieldKeypointsDetector, GoalDetector
from goal_detection.enhanced_goal_detector import EnhancedGoalDetector
from utils.goal_utils import load_manual_goals
from utils import read_video

def debug_goal_detection():
    """Debug the goal detection system step by step."""
    print("🔍 DEBUGGING GOAL DETECTION SYSTEM")
    print("=" * 50)
    
    # Load manual goals to understand expected results
    manual_goals = load_manual_goals("data/manual_goals_videoplayback_process.csv")
    print(f"\n📋 Manual Goals Expected:")
    for goal in manual_goals:
        print(f"   Frame {goal['frame_num']}: Team {goal['team']}, Player {goal['player_id']}")
    
    # Initialize components
    print(f"\n🔧 Initializing Goal Detection Components...")
    try:
        field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
        goal_detector = GoalDetector(field_keypoints_detector)
        enhanced_goal_detector = EnhancedGoalDetector(field_keypoints_detector)
        print("   ✅ Components initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize components: {e}")
        return
    
    # Check if tracks exist
    tracks_path = "data/stubs/videoplayback_process_tracks.pkl"
    if not os.path.exists(tracks_path):
        print(f"   ❌ Tracks file not found: {tracks_path}")
        print("   💡 Run main.py first to generate tracks")
        return
    
    # Load tracks
    print(f"\n📂 Loading tracks from {tracks_path}...")
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    
    total_frames = len(tracks['players'])
    ball_frames = len(tracks['ball'])
    print(f"   ✅ Loaded tracks: {total_frames} player frames, {ball_frames} ball frames")
    
    # Analyze ball detection around manual goal frames
    print(f"\n⚽ Analyzing Ball Detection at Goal Frames...")
    
    for goal in manual_goals:
        frame_num = goal['frame_num']
        team = goal['team']
        
        print(f"\n🎯 Goal at Frame {frame_num} (Team {team}):")
        
        # Check if frame exists in tracks
        if frame_num >= len(tracks['ball']):
            print(f"   ❌ Frame {frame_num} not in tracks (max: {len(tracks['ball'])})")
            continue
            
        # Get ball information
        ball_track = tracks['ball'][frame_num]
        if not ball_track or 1 not in ball_track:
            print(f"   ❌ No ball detected at frame {frame_num}")
            continue
            
        ball_info = ball_track[1]
        ball_bbox = ball_info.get('bbox')
        ball_position = ball_info.get('position')
        ball_confidence = ball_info.get('confidence', 0.0)
        
        print(f"   📍 Ball Position: {ball_position}")
        print(f"   📦 Ball BBox: {ball_bbox}")
        print(f"   🎯 Ball Confidence: {ball_confidence}")
        
        if ball_position:
            # Test field keypoints detection
            print(f"   🔍 Testing Field Keypoints Detection...")
            
            # Create a dummy frame for testing (we'll improve this)
            test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Update keypoints
            goal_detector.update_keypoints(test_frame)
            enhanced_goal_detector.update_keypoints(test_frame)
            
            # Test goal area detection
            goal_side_original = goal_detector._check_ball_in_goal(ball_position)
            goal_side_enhanced = enhanced_goal_detector._detect_goal_fusion(ball_position)
            
            print(f"   🥅 Original Goal Detection: {goal_side_original}")
            print(f"   🥅 Enhanced Goal Detection: {goal_side_enhanced}")
            
            # Test keypoints directly
            keypoint_result = field_keypoints_detector.is_ball_in_goal_area(ball_position)
            print(f"   🔑 Keypoints Goal Detection: {keypoint_result}")
            
            # Check goal areas
            goal_areas = field_keypoints_detector.get_goal_areas()
            print(f"   📐 Goal Areas Available: {list(goal_areas.keys()) if goal_areas else 'None'}")
            
            if goal_areas:
                for side, area in goal_areas.items():
                    print(f"      {side}: {area}")
    
    # Analyze overall ball detection quality
    print(f"\n📊 Overall Ball Detection Analysis...")
    
    ball_detected_frames = 0
    high_confidence_frames = 0
    
    for frame_num in range(min(1000, len(tracks['ball']))):  # Sample first 1000 frames
        ball_track = tracks['ball'][frame_num]
        if ball_track and 1 in ball_track:
            ball_detected_frames += 1
            confidence = ball_track[1].get('confidence', 0.0)
            if confidence > 0.7:
                high_confidence_frames += 1
    
    detection_rate = ball_detected_frames / min(1000, len(tracks['ball'])) * 100
    high_conf_rate = high_confidence_frames / max(1, ball_detected_frames) * 100
    
    print(f"   📈 Ball Detection Rate: {detection_rate:.1f}%")
    print(f"   🎯 High Confidence Rate: {high_conf_rate:.1f}%")
    
    # Test field keypoints detection quality
    print(f"\n🔑 Testing Field Keypoints Detection Quality...")
    
    # Load a sample frame to test keypoints
    try:
        video_path = "data/input_videos/videoplayback_process.mp4"
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, sample_frame = cap.read()
            cap.release()
            
            if ret:
                keypoints = field_keypoints_detector.detect_keypoints(sample_frame)
                print(f"   🔍 Keypoints Detected: {len(keypoints)} types")
                for kp_type, positions in keypoints.items():
                    print(f"      {kp_type}: {len(positions)} points")
                
                # Test goal area calculation
                goal_areas = field_keypoints_detector.get_goal_areas()
                if goal_areas:
                    print(f"   🥅 Goal Areas Calculated: {list(goal_areas.keys())}")
                else:
                    print(f"   ❌ No goal areas calculated")
            else:
                print(f"   ❌ Could not read sample frame")
        else:
            print(f"   ❌ Video file not found: {video_path}")
    except Exception as e:
        print(f"   ❌ Error testing keypoints: {e}")
    
    print(f"\n🏁 Debug Analysis Complete!")
    print(f"💡 Next steps:")
    print(f"   1. Check ball detection accuracy at goal frames")
    print(f"   2. Verify field keypoints detection")
    print(f"   3. Test goal area calculations")
    print(f"   4. Improve goal detection logic")

if __name__ == "__main__":
    debug_goal_detection()
