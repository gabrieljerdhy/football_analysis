#!/usr/bin/env python3
"""
Test script for the improved goal detection system.
"""

import os
import sys
import pickle
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

from goal_detection import FieldKeypointsDetector
from goal_detection.improved_goal_system import ImprovedGoalDetectionSystem
from utils.goal_utils import load_manual_goals
from trackers import Tracker

def test_improved_goal_system():
    """Test the improved goal detection system."""
    print("🚀 TESTING IMPROVED GOAL DETECTION SYSTEM")
    print("=" * 60)
    
    # Load manual goals for reference
    manual_goals = load_manual_goals("data/manual_goals_videoplayback_process.csv")
    print(f"\n📋 Expected goals from manual data: {len(manual_goals)}")
    for goal in manual_goals:
        print(f"   Frame {goal['frame_num']}: Team {goal['team']}")
    
    # Initialize components
    print(f"\n🔧 Initializing improved goal detection system...")
    try:
        field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
        improved_goal_system = ImprovedGoalDetectionSystem(field_keypoints_detector)
        print("✅ Improved goal detection system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Load tracks
    tracks_path = "data/stubs/videoplayback_process_tracks.pkl"
    if not os.path.exists(tracks_path):
        print(f"❌ Tracks file not found: {tracks_path}")
        return
    
    print(f"\n📂 Loading tracks...")
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    
    # Initialize tracker to add positions
    tracker = Tracker(
        "data/models/best_player_detect.pt",
        enable_jersey_detection=False,
        ball_model_path="data/models/best_ball_latest.pt",
        enable_enhanced_ball_detection=True,
    )
    
    print(f"🔧 Adding positions to tracks...")
    tracker.add_position_to_tracks(tracks)
    
    print(f"⚽ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Load a test frame for keypoints initialization
    video_path = "data/input_videos/videoplayback_process.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, test_frame = cap.read()
        cap.release()
        
        if ret:
            print(f"🔑 Updating keypoints with test frame...")
            improved_goal_system.update_keypoints(test_frame, force_detection=True)
    
    # Test goal detection at manual goal frames
    print(f"\n🎯 Testing goal detection at manual goal frames...")
    
    detected_goals = 0
    
    for i, goal in enumerate(manual_goals):
        frame_num = goal['frame_num']
        expected_team = goal['team']
        
        print(f"\n🔍 Testing goal {i+1} at frame {frame_num} (expected: Team {expected_team}):")
        
        if frame_num >= len(tracks['ball']):
            print(f"   ❌ Frame {frame_num} not in tracks (max: {len(tracks['ball'])})")
            continue
        
        # Get ball information
        ball_track = tracks['ball'][frame_num]
        if not ball_track or 1 not in ball_track:
            print(f"   ❌ No ball detected at frame {frame_num}")
            continue
        
        ball_info = ball_track[1]
        ball_position = ball_info.get('position')
        ball_confidence = ball_info.get('confidence', 0.0)
        ball_source = ball_info.get('source', 'unknown')
        
        if not ball_position:
            print(f"   ❌ No ball position at frame {frame_num}")
            continue
        
        print(f"   📍 Ball position: {ball_position}")
        print(f"   🎯 Ball confidence: {ball_confidence}")
        print(f"   📊 Ball source: {ball_source}")
        
        # Test goal detection
        goal_event = improved_goal_system.detect_goal(
            ball_position=ball_position,
            player_id=-1,  # Unknown player
            team=expected_team,
            frame_num=frame_num,
            ball_confidence=ball_confidence,
            ball_source=ball_source,
        )
        
        if goal_event:
            detected_goals += 1
            print(f"   ✅ GOAL DETECTED!")
            print(f"      Side: {goal_event.goal_side}")
            print(f"      Team: {goal_event.team}")
            print(f"      Confidence: {goal_event.confidence:.3f}")
            print(f"      Method: {goal_event.detection_method}")
        else:
            print(f"   ❌ No goal detected")
    
    # Get final statistics
    print(f"\n📊 FINAL RESULTS:")
    stats = improved_goal_system.get_goal_statistics()
    
    print(f"   Goals detected: {detected_goals}/{len(manual_goals)}")
    print(f"   Detection rate: {detected_goals/len(manual_goals)*100:.1f}%")
    print(f"   System total goals: {stats['total_goals']}")
    print(f"   Team 1 goals: {stats['team_goals'].get(1, 0)}")
    print(f"   Team 2 goals: {stats['team_goals'].get(2, 0)}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    
    if stats['goal_events']:
        print(f"\n🎯 Goal events detected by system:")
        for i, event in enumerate(stats['goal_events'], 1):
            print(f"   Goal {i}: Frame {event['frame_num']}, Team {event['team']}, Side: {event['goal_side']}")
            print(f"      Confidence: {event['confidence']:.3f}, Method: {event['detection_method']}")
    
    # Test with different ball positions to verify goal areas
    print(f"\n🧪 Testing goal area boundaries:")
    
    test_positions = [
        (50, 400, "Left goal area"),
        (100, 500, "Left goal area (center)"),
        (200, 600, "Left goal area (bottom)"),
        (1800, 400, "Right goal area"),
        (1850, 500, "Right goal area (center)"),
        (1750, 600, "Right goal area (bottom)"),
        (960, 540, "Center field"),
        (500, 300, "Left field"),
        (1400, 700, "Right field"),
    ]
    
    for x, y, description in test_positions:
        # Test detection without actual goal event
        detection_result = improved_goal_system._detect_goal_multi_method((x, y))
        if detection_result:
            side, confidence, method = detection_result
            print(f"   {description} ({x}, {y}): {side} goal (conf: {confidence:.3f}, method: {method})")
        else:
            print(f"   {description} ({x}, {y}): No goal detected")
    
    print(f"\n✅ Improved goal detection system test complete!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if detected_goals == len(manual_goals):
        print(f"   🎉 Perfect detection! System is working correctly.")
    elif detected_goals >= len(manual_goals) * 0.75:
        print(f"   ✅ Good detection rate. Minor tuning may improve results.")
    else:
        print(f"   ⚠️  Low detection rate. Consider:")
        print(f"      - Adjusting goal area boundaries")
        print(f"      - Lowering confidence thresholds")
        print(f"      - Improving ball position accuracy")
        print(f"      - Adding more detection methods")

if __name__ == "__main__":
    test_improved_goal_system()
