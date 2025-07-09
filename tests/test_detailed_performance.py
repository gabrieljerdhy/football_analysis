#!/usr/bin/env python3
"""
Detailed performance test to identify bottlenecks in the video processing pipeline.
"""

import os
import sys
import time

import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trackers.tracker import Tracker
from src.utils import get_optimal_device


def create_test_frames(num_frames=20):
    """Create test frames."""
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def test_individual_components():
    """Test individual components to identify bottlenecks."""
    print("🔬 DETAILED PERFORMANCE ANALYSIS")
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

    test_frames = create_test_frames(100)  # Use more frames to test batch processing
    print(f"🧪 Testing with {len(test_frames)} frames on {device}")

    # Test 1: Pure YOLO inference (main model)
    print("\n1️⃣ Testing detect_frames_memory_efficient...")
    start_time = time.time()
    detections = tracker.detect_frames_memory_efficient(test_frames)
    end_time = time.time()

    main_model_time = end_time - start_time
    main_fps = len(test_frames) / main_model_time if main_model_time > 0 else 0
    main_avg = (main_model_time / len(test_frames)) * 1000

    print(f"⏱️  Main model time: {main_model_time:.2f}s")
    print(f"🎬 Main model FPS: {main_fps:.1f}")
    print(f"⚡ Main model avg: {main_avg:.1f}ms/frame")

    # Test 2: Ball detection
    if tracker.enable_enhanced_ball_detection:
        print("\n2️⃣ Testing detect_ball_enhanced...")
        start_time = time.time()
        ball_detections = tracker.detect_ball_enhanced(test_frames)
        end_time = time.time()

        ball_time = end_time - start_time
        ball_fps = len(test_frames) / ball_time if ball_time > 0 else 0
        ball_avg = (ball_time / len(test_frames)) * 1000

        print(f"⏱️  Ball model time: {ball_time:.2f}s")
        print(f"🎬 Ball model FPS: {ball_fps:.1f}")
        print(f"⚡ Ball model avg: {ball_avg:.1f}ms/frame")
    else:
        ball_time = 0
        print("\n2️⃣ Ball detection disabled")

    # Test 3: Object tracking (supervision)
    print("\n3️⃣ Testing object tracking...")
    start_time = time.time()

    tracks = {"players": [], "referees": [], "ball": []}

    for frame_num, detection in enumerate(detections):
        import supervision as sv

        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # Convert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        # Convert GoalKeeper to player object
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = cls_names_inv["player"]

        # Track Objects
        detection_with_tracks = tracker.tracker.update_with_detections(
            detection_supervision
        )

        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})

        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv["player"]:
                tracks["players"][frame_num][track_id] = {"bbox": bbox}

            if cls_id == cls_names_inv["referee"]:
                tracks["referees"][frame_num][track_id] = {"bbox": bbox}

    end_time = time.time()

    tracking_time = end_time - start_time
    tracking_fps = len(test_frames) / tracking_time if tracking_time > 0 else 0
    tracking_avg = (tracking_time / len(test_frames)) * 1000

    print(f"⏱️  Tracking time: {tracking_time:.2f}s")
    print(f"🎬 Tracking FPS: {tracking_fps:.1f}")
    print(f"⚡ Tracking avg: {tracking_avg:.1f}ms/frame")

    # Test 4: Full pipeline
    print("\n4️⃣ Testing full get_object_tracks_memory_efficient...")
    start_time = time.time()
    full_tracks = tracker.get_object_tracks_memory_efficient(test_frames)
    end_time = time.time()

    full_time = end_time - start_time
    full_fps = len(test_frames) / full_time if full_time > 0 else 0
    full_avg = (full_time / len(test_frames)) * 1000

    print(f"⏱️  Full pipeline time: {full_time:.2f}s")
    print(f"🎬 Full pipeline FPS: {full_fps:.1f}")
    print(f"⚡ Full pipeline avg: {full_avg:.1f}ms/frame")

    # Analysis
    print("\n📊 PERFORMANCE BREAKDOWN")
    print("=" * 60)
    print(f"{'Component':<25} {'Time (s)':<10} {'FPS':<8} {'ms/frame':<10}")
    print("-" * 60)
    print(
        f"{'Main Model':<25} {main_model_time:<10.2f} {main_fps:<8.1f} {main_avg:<10.1f}"
    )
    print(f"{'Ball Model':<25} {ball_time:<10.2f} {ball_fps:<8.1f} {ball_avg:<10.1f}")
    print(
        f"{'Object Tracking':<25} {tracking_time:<10.2f} {tracking_fps:<8.1f} {tracking_avg:<10.1f}"
    )
    print(
        f"{'Full Pipeline':<25} {full_time:<10.2f} {full_fps:<8.1f} {full_avg:<10.1f}"
    )

    # Identify bottleneck
    times = [
        ("Main Model", main_model_time),
        ("Ball Model", ball_time),
        ("Object Tracking", tracking_time),
    ]

    bottleneck = max(times, key=lambda x: x[1])
    print(f"\n🎯 Bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f}s)")

    if full_avg < 15:
        print("✅ Overall performance looks good!")
    elif full_avg < 25:
        print("⚠️ Performance is acceptable but could be better")
    else:
        print("❌ Performance needs improvement")

    return full_avg


def main():
    try:
        avg_time = test_individual_components()

        print("\n💡 RECOMMENDATIONS")
        print("=" * 60)

        if avg_time and avg_time < 15:
            print("✅ The optimizations are working well!")
            print("🚀 Your video processing should be much faster now.")
        else:
            print("⚠️ There may still be bottlenecks to address.")
            print("🔍 Check the breakdown above to see which component is slowest.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
