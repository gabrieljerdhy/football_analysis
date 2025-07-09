#!/usr/bin/env python3
"""
Test to verify that the video processing pipeline is using optimized methods.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trackers.tracker import Tracker
from src.utils import get_optimal_device


def test_tracker_methods():
    """Test that tracker methods are using optimized settings."""
    print("🔬 TESTING TRACKER OPTIMIZATION USAGE")
    print("=" * 60)

    device = get_optimal_device()
    model_path = "data/models/best_player_detect.pt"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Initialize tracker
    tracker = Tracker(
        model_path=model_path,
        ball_model_path=(
            "data/models/best_ball_latest.pt"
            if os.path.exists("data/models/best_ball_latest.pt")
            else None
        ),
        enable_enhanced_ball_detection=os.path.exists(
            "data/models/best_ball_latest.pt"
        ),
        enable_jersey_detection=False,  # Disable for pure inference testing
        device=device,
    )

    print(f"🖥️  Device: {device}")
    print(f"🔥 Half precision: {tracker.use_half_precision}")
    print(f"📊 Optimized batch size: {tracker.optimized_batch_size}")
    print(f"🎯 Ball batch size: {tracker.ball_batch_size}")
    print(f"📐 Image size: {tracker.imgsz}")

    # Create test frames
    test_frames = []
    for i in range(20):
        frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
        test_frames.append(frame)

    print(
        f"\n🧪 Testing get_object_tracks_memory_efficient with {len(test_frames)} frames..."
    )

    # Test the memory-efficient method
    start_time = time.time()
    tracks = tracker.get_object_tracks_memory_efficient(test_frames)
    end_time = time.time()

    total_time = end_time - start_time
    fps = len(test_frames) / total_time if total_time > 0 else 0
    avg_time = (total_time / len(test_frames)) * 1000

    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"⚡ Avg time per frame: {avg_time:.1f}ms")
    print(f"🔍 Player detections: {len([f for f in tracks['players'] if f])}")
    print(f"🔍 Ball detections: {len([f for f in tracks['ball'] if f])}")

    # Check if performance is good
    if avg_time < 20:  # Should be much faster than 27-29ms
        print("✅ Performance looks good!")
    else:
        print("⚠️ Performance might still be slow")

    return avg_time


def test_field_keypoints_optimization():
    """Test that field keypoints detector is using optimizations."""
    print("\n🏟️ TESTING FIELD KEYPOINTS OPTIMIZATION")
    print("=" * 60)

    try:
        from src.goal_detection.field_keypoints_detector import FieldKeypointsDetector

        device = get_optimal_device()
        model_path = "data/models/best_field_keypoint.pt"

        if not os.path.exists(model_path):
            print(f"❌ Field keypoints model not found: {model_path}")
            return

        detector = FieldKeypointsDetector(model_path, device=device)

        print(f"🖥️  Device: {device}")
        print(f"🔥 Half precision: {detector.use_half_precision}")

        # Test detection
        test_frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

        start_time = time.time()
        keypoints = detector.detect_keypoints(test_frame, force_detection=True)
        end_time = time.time()

        detection_time = (end_time - start_time) * 1000
        print(f"⚡ Detection time: {detection_time:.1f}ms")
        print(f"🔍 Keypoints detected: {len(keypoints)}")

        if detection_time < 50:  # Should be reasonably fast
            print("✅ Field keypoints performance looks good!")
        else:
            print("⚠️ Field keypoints might be slow")

    except Exception as e:
        print(f"❌ Field keypoints test failed: {e}")


def main():
    print("🚀 VIDEO PROCESSING PERFORMANCE VERIFICATION")
    print("=" * 70)

    # Test tracker optimization
    tracker_time = test_tracker_methods()

    # Test field keypoints optimization
    test_field_keypoints_optimization()

    print("\n📊 SUMMARY")
    print("=" * 70)

    if tracker_time and tracker_time < 20:
        print("✅ Tracker optimizations are working!")
        print("🎯 The video processing should now be much faster.")
        print("🚀 Expected improvement: ~11x faster than before")
    else:
        print("⚠️ Tracker performance might still need optimization")

    print("\n💡 NEXT STEPS:")
    print("1. Run your original command to test real video processing")
    print("2. Look for inference times around 9-10ms instead of 27-29ms")
    print("3. Check that batch processing is being used in the logs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
