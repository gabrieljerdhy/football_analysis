#!/usr/bin/env python3
"""
Debug Goal Detection Issues

This script investigates why the goal detection system is not working properly
for the videoplayback_process.mp4 video that should have 4-0 goals but shows 2-2.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from goal_detection.field_keypoints_detector import FieldKeypointsDetector
from goal_detection.goal_detector import GoalDetector
from scoreboard_detection.scoreboard_analyzer import ScoreboardAnalyzer


def debug_goal_detection():
    """Debug the goal detection system with the problematic video."""

    video_path = "../data/input_videos/videoplayback_process.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    print(f"🔍 Debugging goal detection for: {video_path}")

    # Initialize components
    print("🏗️ Initializing detection systems...")

    # Field keypoints detector
    field_detector = FieldKeypointsDetector(
        "../data/models/best_field_keypoint.pt", device="cuda"
    )

    # Goal detector
    goal_detector = GoalDetector(field_detector)
    goal_detector.set_keypoint_optimization(
        detection_interval=5, stability_threshold=10
    )

    # Scoreboard analyzer
    scoreboard_analyzer = ScoreboardAnalyzer(
        detection_interval=30,
        min_detection_confidence=0.6,
        min_extraction_confidence=0.6,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS")

    # Test keypoint detection on first few frames
    print("\n🎯 Testing field keypoint detection...")

    keypoint_results = []
    for frame_num in range(0, min(100, total_frames), 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Test keypoint detection
        keypoints = field_detector.detect_keypoints(frame, force_detection=True)
        keypoint_results.append(
            {
                "frame": frame_num,
                "keypoints_detected": len(keypoints),
                "keypoints": list(keypoints.keys()),
            }
        )

        if frame_num == 0:
            print(f"   Frame {frame_num}: {len(keypoints)} keypoints detected")
            if keypoints:
                print(f"   Keypoint types: {list(keypoints.keys())[:5]}...")

    # Test scoreboard detection on sample frames
    print("\n📊 Testing scoreboard detection...")

    scoreboard_results = []
    for frame_num in range(0, min(1000, total_frames), 100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Test scoreboard detection
        result = scoreboard_analyzer.analyze_frame(frame, frame_num)
        if result:
            scoreboard_results.append(
                {
                    "frame": frame_num,
                    "team1_score": result.get("team1_score"),
                    "team2_score": result.get("team2_score"),
                    "confidence": result.get("confidence"),
                    "method": result.get("detection_method"),
                }
            )
            print(
                f"   Frame {frame_num}: {result.get('team1_score')}-{result.get('team2_score')} (conf: {result.get('confidence'):.3f})"
            )

    # Test goal area detection with sample ball positions
    print("\n⚽ Testing goal area detection...")

    # Get video dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"   Video dimensions: {width}x{height}")

        # Update goal detector with frame
        goal_detector.update_keypoints(frame)

        # Test various ball positions
        test_positions = [
            (50, height // 2),  # Far left (potential left goal)
            (width - 50, height // 2),  # Far right (potential right goal)
            (width // 2, height // 2),  # Center
            (100, height // 2),  # Left side
            (width - 100, height // 2),  # Right side
        ]

        for i, pos in enumerate(test_positions):
            goal_side = goal_detector._check_ball_in_goal(pos)
            print(
                f"   Position {pos}: {'Goal detected: ' + goal_side if goal_side else 'No goal'}"
            )

    cap.release()

    # Summary
    print(f"\n📋 Summary:")
    print(
        f"   Keypoint detection: {len([r for r in keypoint_results if r['keypoints_detected'] > 0])}/{len(keypoint_results)} frames successful"
    )
    print(f"   Scoreboard detection: {len(scoreboard_results)} detections found")

    if scoreboard_results:
        # Check if scoreboard consistently shows 2-2
        scores = [(r["team1_score"], r["team2_score"]) for r in scoreboard_results]
        unique_scores = set(scores)
        print(f"   Unique scores detected: {unique_scores}")

        if (2, 2) in unique_scores:
            print(
                "   ⚠️  Scoreboard consistently detects 2-2 - this might be incorrect!"
            )
            print(
                "   💡 Suggestion: The scoreboard might be showing a different match or wrong data"
            )

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("   1. Check if scoreboard detection is reading the correct scoreboard")
    print("   2. Verify field keypoint detection is working properly")
    print("   3. Test goal detection with known ball positions near goals")
    print("   4. Consider adjusting scoreboard confidence thresholds")
    print("   5. Add manual goal override if scoreboard is incorrect")


if __name__ == "__main__":
    debug_goal_detection()
