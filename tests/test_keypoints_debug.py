#!/usr/bin/env python3
"""
Debug script to test field keypoints detection.
"""

import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

from goal_detection import FieldKeypointsDetector

def test_keypoints_detection():
    """Test field keypoints detection in detail."""
    print("🔑 DEBUGGING FIELD KEYPOINTS DETECTION")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt", confidence_threshold=0.3)
        print("✅ Field keypoints detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Load test frame
    video_path = "data/input_videos/videoplayback_process.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Loading test frame from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    # Try multiple frames to find one with good keypoints
    test_frames = [0, 100, 500, 1000, 1500, 2000]
    
    for frame_num in test_frames:
        print(f"\n🔍 Testing frame {frame_num}...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Could not read frame {frame_num}")
            continue
        
        print(f"   Frame shape: {frame.shape}")
        
        # Test keypoint detection with force detection
        keypoints = detector.detect_keypoints(frame, force_detection=True)
        
        print(f"   🔑 Keypoints detected: {len(keypoints)}")
        
        if keypoints:
            print("   📍 Detected keypoints:")
            for kp_name, kp_data in keypoints.items():
                position = kp_data['position']
                confidence = kp_data['confidence']
                print(f"      {kp_name}: {position} (conf: {confidence:.3f})")
            
            # Test goal area calculation
            detector._update_goal_areas()
            goal_areas = detector.get_goal_areas(use_cache=False)
            
            print(f"   🥅 Goal areas calculated:")
            for side, area in goal_areas.items():
                if area:
                    print(f"      {side}: x={area['x_min']}-{area['x_max']}, y={area['y_min']}-{area['y_max']}")
                    print(f"         keypoints: {area.get('keypoints_used', [])}")
                    print(f"         confidence: {area.get('confidence', 0.0):.3f}")
                else:
                    print(f"      {side}: None")
            
            # Test ball position in goal areas
            test_positions = [
                (50, 400),    # Left side
                (1850, 400),  # Right side
                (960, 400),   # Center
                (182, 478),   # From manual goal frame 2400
                (1599, 867),  # From manual goal frame 1200
            ]
            
            print(f"   🎯 Testing ball positions:")
            for pos in test_positions:
                result = detector.is_ball_in_goal_area(pos)
                print(f"      {pos}: {result}")
            
            # If we found keypoints, we can stop testing
            if len(keypoints) > 0:
                break
        else:
            print("   ❌ No keypoints detected")
    
    cap.release()
    
    # Test with lower confidence threshold
    print(f"\n🔄 Testing with very low confidence threshold (0.1)...")
    detector_low_conf = FieldKeypointsDetector("data/models/best_field_keypoint.pt", confidence_threshold=0.1)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)  # Test middle frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        keypoints_low = detector_low_conf.detect_keypoints(frame, force_detection=True)
        print(f"   🔑 Keypoints with low confidence: {len(keypoints_low)}")
        
        if keypoints_low:
            for kp_name, kp_data in keypoints_low.items():
                position = kp_data['position']
                confidence = kp_data['confidence']
                print(f"      {kp_name}: {position} (conf: {confidence:.3f})")
    
    # Test fallback goal areas
    print(f"\n🔧 Testing fallback goal area calculation...")
    
    # Create fallback goal areas based on video dimensions
    video_width = 1920  # Assuming HD video
    video_height = 1080
    
    # Standard football field proportions for goal areas
    goal_width_ratio = 0.08  # Goals are about 8% of field width
    goal_height_ratio = 0.35  # Goals are about 35% of field height
    
    goal_width = int(video_width * goal_width_ratio)
    goal_height = int(video_height * goal_height_ratio)
    goal_y_center = video_height // 2
    
    fallback_left_goal = {
        'x_min': 0,
        'x_max': goal_width,
        'y_min': goal_y_center - goal_height // 2,
        'y_max': goal_y_center + goal_height // 2,
    }
    
    fallback_right_goal = {
        'x_min': video_width - goal_width,
        'x_max': video_width,
        'y_min': goal_y_center - goal_height // 2,
        'y_max': goal_y_center + goal_height // 2,
    }
    
    print(f"   📐 Fallback goal areas for {video_width}x{video_height}:")
    print(f"      Left: x={fallback_left_goal['x_min']}-{fallback_left_goal['x_max']}, y={fallback_left_goal['y_min']}-{fallback_left_goal['y_max']}")
    print(f"      Right: x={fallback_right_goal['x_min']}-{fallback_right_goal['x_max']}, y={fallback_right_goal['y_min']}-{fallback_right_goal['y_max']}")
    
    # Test manual goal positions with fallback areas
    manual_positions = [
        (182, 478),   # Frame 2400
        (1599, 867),  # Frame 1200
    ]
    
    print(f"   🎯 Testing manual goal positions with fallback areas:")
    for pos in manual_positions:
        x, y = pos
        
        # Test left goal
        if (fallback_left_goal['x_min'] <= x <= fallback_left_goal['x_max'] and 
            fallback_left_goal['y_min'] <= y <= fallback_left_goal['y_max']):
            result = "left"
        # Test right goal
        elif (fallback_right_goal['x_min'] <= x <= fallback_right_goal['x_max'] and 
              fallback_right_goal['y_min'] <= y <= fallback_right_goal['y_max']):
            result = "right"
        else:
            result = None
        
        print(f"      {pos}: {result}")
    
    print(f"\n✅ Keypoints debug complete!")
    print(f"💡 Recommendations:")
    print(f"   1. Field keypoints model may need retraining or different confidence threshold")
    print(f"   2. Implement fallback goal areas based on video dimensions")
    print(f"   3. Use enhanced goal detection with multiple methods")

if __name__ == "__main__":
    test_keypoints_detection()
