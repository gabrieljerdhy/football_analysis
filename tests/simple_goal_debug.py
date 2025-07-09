#!/usr/bin/env python3
"""
Simple Goal Detection Debug

Quick test to identify the main issues with goal detection.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_model_loading():
    """Test if models can be loaded properly."""
    print("🔍 Testing model loading...")
    
    try:
        from goal_detection.field_keypoints_detector import FieldKeypointsDetector
        
        model_path = "../data/models/best_field_keypoint.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        print(f"✅ Model file exists: {model_path}")
        
        # Try to load the detector
        detector = FieldKeypointsDetector(model_path, device="cpu")  # Use CPU for safety
        print("✅ FieldKeypointsDetector loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading FieldKeypointsDetector: {e}")
        return False

def test_video_access():
    """Test if video can be accessed."""
    print("\n🎥 Testing video access...")
    
    video_path = "../data/input_videos/videoplayback_process.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video opened successfully")
    print(f"   Frames: {total_frames}, FPS: {fps:.2f}")
    print(f"   Resolution: {width}x{height}")
    
    # Test reading a frame
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame read successfully: {frame.shape}")
    else:
        print("❌ Could not read frame")
        
    cap.release()
    return ret

def test_scoreboard_detection():
    """Test scoreboard detection on a few frames."""
    print("\n📊 Testing scoreboard detection...")
    
    try:
        from scoreboard_detection.scoreboard_analyzer import ScoreboardAnalyzer
        
        analyzer = ScoreboardAnalyzer(
            detection_interval=1,  # Test every frame
            min_detection_confidence=0.5,  # Lower threshold for testing
            min_extraction_confidence=0.5,
        )
        
        video_path = "../data/input_videos/videoplayback_process.mp4"
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Could not open video for scoreboard test")
            return False
            
        # Test on a few frames
        test_frames = [0, 100, 500, 1000, 2000]
        detections = []
        
        for frame_num in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            result = analyzer.analyze_frame(frame, frame_num)
            if result:
                detections.append({
                    'frame': frame_num,
                    'score': f"{result.get('team1_score', '?')}-{result.get('team2_score', '?')}",
                    'confidence': result.get('confidence', 0)
                })
                print(f"   Frame {frame_num}: {result.get('team1_score', '?')}-{result.get('team2_score', '?')} (conf: {result.get('confidence', 0):.3f})")
        
        cap.release()
        
        if detections:
            print(f"✅ Scoreboard detection working: {len(detections)} detections")
            # Check if all detections show the same score
            scores = [d['score'] for d in detections]
            unique_scores = set(scores)
            print(f"   Unique scores detected: {unique_scores}")
            
            if len(unique_scores) == 1 and '2-2' in unique_scores:
                print("⚠️  All detections show 2-2 - this might be the issue!")
                print("💡 The scoreboard might be showing incorrect data or from a different match")
        else:
            print("❌ No scoreboard detections found")
            
        return len(detections) > 0
        
    except Exception as e:
        print(f"❌ Error in scoreboard detection: {e}")
        return False

def analyze_priority_system():
    """Analyze the goal detection priority system."""
    print("\n🏆 Analyzing goal detection priority system...")
    
    print("Priority order (highest to lowest):")
    print("1. Manual goals (from CSV config)")
    print("2. Scoreboard-extracted scores")
    print("3. Enhanced goal detection")
    print("4. Regular goal detection")
    
    print("\nCurrent situation based on CSV output:")
    print("- Manual goals: Not provided")
    print("- Scoreboard scores: 2-2 (confidence: 0.945) ← WINNING")
    print("- Enhanced goals: 0-0")
    print("- Regular goals: 0-0")
    
    print("\n💡 Root cause analysis:")
    print("1. Scoreboard detection is overriding everything with 2-2")
    print("2. Enhanced goal detection is not working (0 keypoints detected)")
    print("3. Regular goal detection is also not working")
    
    print("\n🔧 Potential solutions:")
    print("1. Fix field keypoint detection (model loading issue?)")
    print("2. Improve scoreboard detection accuracy")
    print("3. Add manual goal override for this specific video")
    print("4. Adjust scoreboard confidence thresholds")

def main():
    """Run all tests."""
    print("🚀 Starting Goal Detection Debug Analysis\n")
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test video access
    video_ok = test_video_access()
    
    # Test scoreboard detection
    scoreboard_ok = test_scoreboard_detection()
    
    # Analyze priority system
    analyze_priority_system()
    
    print(f"\n📋 Summary:")
    print(f"   Model loading: {'✅' if model_ok else '❌'}")
    print(f"   Video access: {'✅' if video_ok else '❌'}")
    print(f"   Scoreboard detection: {'✅' if scoreboard_ok else '❌'}")
    
    if not model_ok:
        print("\n🔥 CRITICAL: Field keypoint model loading failed!")
        print("   This explains why enhanced goal detection shows 0 goals")
    
    if scoreboard_ok:
        print("\n⚠️  Scoreboard detection is working but may be reading wrong data")
        print("   Consider lowering scoreboard confidence or adding manual override")

if __name__ == "__main__":
    main()
