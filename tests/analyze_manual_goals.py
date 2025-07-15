#!/usr/bin/env python3
"""
Analyze manual goals to understand the expected goal positions and improve detection.
"""

import os
import sys
import pickle
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

from utils.goal_utils import load_manual_goals
from trackers import Tracker

def analyze_manual_goals():
    """Analyze manual goals to understand ball positions and improve detection."""
    print("🔍 ANALYZING MANUAL GOALS FOR IMPROVED DETECTION")
    print("=" * 60)
    
    # Load manual goals
    manual_goals = load_manual_goals("data/manual_goals_videoplayback_process.csv")
    print(f"\n📋 Manual goals to analyze: {len(manual_goals)}")
    
    # Load tracks
    tracks_path = "data/stubs/videoplayback_process_tracks.pkl"
    if not os.path.exists(tracks_path):
        print(f"❌ Tracks file not found: {tracks_path}")
        return
    
    print(f"📂 Loading tracks...")
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
    
    # Load video for frame analysis
    video_path = "data/input_videos/videoplayback_process.mp4"
    cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
    
    # Analyze each manual goal
    print(f"\n🎯 DETAILED ANALYSIS OF MANUAL GOALS:")
    print("=" * 60)
    
    valid_goals = []
    ball_positions = []
    
    for i, goal in enumerate(manual_goals):
        frame_num = goal['frame_num']
        team = goal['team']
        
        print(f"\n📊 GOAL {i+1}: Frame {frame_num}, Team {team}")
        print("-" * 40)
        
        # Check if frame exists
        if frame_num >= len(tracks['ball']):
            print(f"❌ Frame {frame_num} exceeds video length ({len(tracks['ball'])} frames)")
            print(f"   Video is approximately {len(tracks['ball'])/24:.1f} seconds long")
            continue
        
        # Get ball information
        ball_track = tracks['ball'][frame_num]
        if not ball_track or 1 not in ball_track:
            print(f"❌ No ball detected at frame {frame_num}")
            continue
        
        ball_info = ball_track[1]
        ball_position = ball_info.get('position')
        ball_bbox = ball_info.get('bbox')
        ball_confidence = ball_info.get('confidence', 0.0)
        ball_source = ball_info.get('source', 'unknown')
        
        print(f"📍 Ball position: {ball_position}")
        print(f"📦 Ball bbox: {ball_bbox}")
        print(f"🎯 Ball confidence: {ball_confidence:.3f}")
        print(f"📊 Ball source: {ball_source}")
        
        if ball_position:
            ball_positions.append(ball_position)
            x, y = ball_position
            
            # Analyze position relative to field
            video_width = 1920
            video_height = 1080
            
            # Calculate relative position
            x_ratio = x / video_width
            y_ratio = y / video_height
            
            print(f"📐 Relative position: ({x_ratio:.3f}, {y_ratio:.3f})")
            
            # Determine field area
            if x_ratio < 0.2:
                area = "Left side"
            elif x_ratio > 0.8:
                area = "Right side"
            else:
                area = "Center field"
            
            if y_ratio < 0.3:
                area += " (upper)"
            elif y_ratio > 0.7:
                area += " (lower)"
            else:
                area += " (middle)"
            
            print(f"🏟️  Field area: {area}")
            
            # Check if it's in current goal areas
            # Left goal areas
            left_primary = (0 <= x <= 192 and 324 <= y <= 756)
            left_extended = (0 <= x <= 288 and 270 <= y <= 810)
            left_fallback = (0 <= x <= 250 and 267 <= y <= 813)
            
            # Right goal areas  
            right_primary = (1728 <= x <= 1920 and 324 <= y <= 756)
            right_extended = (1632 <= x <= 1920 and 270 <= y <= 810)
            right_fallback = (1670 <= x <= 1920 and 267 <= y <= 813)
            
            goal_detection = []
            if left_primary:
                goal_detection.append("Left Primary")
            if left_extended:
                goal_detection.append("Left Extended")
            if left_fallback:
                goal_detection.append("Left Fallback")
            if right_primary:
                goal_detection.append("Right Primary")
            if right_extended:
                goal_detection.append("Right Extended")
            if right_fallback:
                goal_detection.append("Right Fallback")
            
            if goal_detection:
                print(f"✅ In goal areas: {', '.join(goal_detection)}")
                valid_goals.append(goal)
            else:
                print(f"❌ NOT in any goal area")
                
                # Calculate distance to nearest goal
                left_goal_center = (96, 540)  # Center of left goal
                right_goal_center = (1824, 540)  # Center of right goal
                
                dist_left = ((x - left_goal_center[0])**2 + (y - left_goal_center[1])**2)**0.5
                dist_right = ((x - right_goal_center[0])**2 + (y - right_goal_center[1])**2)**0.5
                
                nearest_goal = "left" if dist_left < dist_right else "right"
                nearest_dist = min(dist_left, dist_right)
                
                print(f"📏 Nearest goal: {nearest_goal} (distance: {nearest_dist:.1f} pixels)")
                
                # Suggest expanded goal area
                if nearest_goal == "left" and x < 400:  # Reasonable left side
                    print(f"💡 Could be detected with expanded left goal area")
                elif nearest_goal == "right" and x > 1520:  # Reasonable right side
                    print(f"💡 Could be detected with expanded right goal area")
                else:
                    print(f"⚠️  Position seems too far from goals - might not be a valid goal")
        
        # Load and analyze the actual frame if possible
        if cap and ball_position:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Save frame for manual inspection
                frame_filename = f"debug_frame_{frame_num}_team_{team}.jpg"
                
                # Draw ball position on frame
                x, y = ball_position
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 3)
                cv2.putText(frame, f"Ball: ({x}, {y})", (int(x)+15, int(y)-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw goal areas
                # Left goal
                cv2.rectangle(frame, (0, 324), (192, 756), (255, 0, 0), 2)
                cv2.putText(frame, "Left Goal", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Right goal
                cv2.rectangle(frame, (1728, 324), (1920, 756), (255, 0, 0), 2)
                cv2.putText(frame, "Right Goal", (1730, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imwrite(frame_filename, frame)
                print(f"💾 Saved frame analysis: {frame_filename}")
    
    if cap:
        cap.release()
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS:")
    print("=" * 40)
    print(f"Total manual goals: {len(manual_goals)}")
    print(f"Valid goals in current areas: {len(valid_goals)}")
    print(f"Detection rate with current areas: {len(valid_goals)/len(manual_goals)*100:.1f}%")
    
    if ball_positions:
        # Analyze ball position distribution
        x_positions = [pos[0] for pos in ball_positions]
        y_positions = [pos[1] for pos in ball_positions]
        
        print(f"\n📐 Ball position statistics:")
        print(f"X range: {min(x_positions):.0f} - {max(x_positions):.0f}")
        print(f"Y range: {min(y_positions):.0f} - {max(y_positions):.0f}")
        print(f"X mean: {np.mean(x_positions):.0f}")
        print(f"Y mean: {np.mean(y_positions):.0f}")
        
        # Suggest improved goal areas
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVED GOAL AREAS:")
        
        # Find leftmost and rightmost positions
        leftmost_x = min(x_positions)
        rightmost_x = max(x_positions)
        
        if leftmost_x < 400:  # Left side goal
            suggested_left_width = int(leftmost_x * 1.2)  # 20% margin
            print(f"   Left goal width: Expand to {suggested_left_width} pixels (current: 250)")
        
        if rightmost_x > 1520:  # Right side goal
            suggested_right_start = int(rightmost_x * 0.9)  # 10% margin
            print(f"   Right goal start: Move to {suggested_right_start} pixels (current: 1670)")
        
        # Y-axis analysis
        y_min = min(y_positions)
        y_max = max(y_positions)
        suggested_y_margin = int((y_max - y_min) * 0.1)  # 10% margin
        
        print(f"   Y range: {y_min-suggested_y_margin} - {y_max+suggested_y_margin} (current: 267-813)")
    
    print(f"\n✅ Manual goals analysis complete!")
    print(f"📁 Check saved frame images for visual verification")

if __name__ == "__main__":
    analyze_manual_goals()
