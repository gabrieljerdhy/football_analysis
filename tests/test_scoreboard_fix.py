#!/usr/bin/env python3
"""
Test script to verify scoreboard detection fix
"""

import os

import cv2

from src.scoreboard_detection import ScoreboardAnalyzer


def test_scoreboard_detection():
    """Test scoreboard detection on a video file."""

    # Test video path
    video_path = "data/input_videos/videoplayback_process.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False

    print(f"🎯 Testing scoreboard detection on: {video_path}")

    # Initialize scoreboard analyzer
    analyzer = ScoreboardAnalyzer(
        detection_interval=30,
        min_detection_confidence=0.6,
        min_extraction_confidence=0.6,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video has {total_frames} frames")

    # Test on a few frames
    test_frames = [0, 100, 500, 1000, 2000, 3000]
    detections = []

    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        result = analyzer.analyze_frame(frame, frame_num)
        if result:
            detections.append(
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

    cap.release()

    # Get final statistics
    stats = analyzer.get_statistics()
    print(f"\n📊 Scoreboard Detection Statistics:")
    print(f"   Frames processed: {stats['frames_processed']}")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Successful extractions: {stats['successful_extractions']}")
    print(f"   Scoreboard detected: {stats['scoreboard_detected']}")
    print(f"   Detection rate: {stats.get('detection_rate', 0):.3f}")

    if stats["scoreboard_detected"]:
        print("✅ Scoreboard detection is working!")
        final_score = analyzer.get_final_score()
        if final_score:
            print(
                f"🏆 Final score: {final_score['team1_score']}-{final_score['team2_score']} (confidence: {final_score['confidence']:.3f})"
            )
        else:
            print("⚠️  Scoreboard detected but no final score extracted")
    else:
        print("❌ No scoreboard detected")

    return stats["scoreboard_detected"]


if __name__ == "__main__":
    test_scoreboard_detection()
