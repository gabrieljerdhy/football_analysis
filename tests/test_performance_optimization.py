#!/usr/bin/env python3
"""
Performance test script to validate YOLO inference optimizations.
Tests different batch sizes and optimization settings to measure performance improvements.
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trackers.tracker import Tracker
from utils import get_optimal_device


def create_dummy_frames(num_frames=100, height=720, width=1280):
    """Create dummy video frames for testing."""
    frames = []
    for i in range(num_frames):
        # Create realistic-looking frame with some variation
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Add some structure to make it more realistic
        frame[height//4:3*height//4, width//4:3*width//4] = np.random.randint(50, 200, (height//2, width//2, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def benchmark_tracker_performance(tracker, frames, test_name=""):
    """Benchmark tracker performance with given frames."""
    print(f"\n🔬 Testing {test_name}")
    print(f"📊 Frames: {len(frames)}, Device: {tracker.device}")
    
    # Warm up
    if len(frames) > 0:
        _ = tracker.detect_frames([frames[0]])
    
    # Benchmark detection
    start_time = time.time()
    detections = tracker.detect_frames(frames)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = len(frames) / total_time if total_time > 0 else 0
    avg_time_per_frame = (total_time / len(frames)) * 1000 if len(frames) > 0 else 0
    
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"⚡ Avg time per frame: {avg_time_per_frame:.1f}ms")
    print(f"🔍 Detections: {len(detections)}")
    
    return {
        'total_time': total_time,
        'fps': fps,
        'avg_time_per_frame': avg_time_per_frame,
        'num_detections': len(detections)
    }


def test_batch_size_performance():
    """Test performance with different batch sizes."""
    print("🚀 YOLO PERFORMANCE OPTIMIZATION TEST")
    print("=" * 60)
    
    device = get_optimal_device()
    print(f"🖥️  Using device: {device}")
    
    # Check if models exist
    model_path = "data/models/best_player_detect.pt"
    ball_model_path = "data/models/best_ball_latest.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the model files are in the correct location.")
        return
    
    # Create test frames
    test_frames_small = create_dummy_frames(20, 384, 640)  # Small batch
    test_frames_medium = create_dummy_frames(50, 384, 640)  # Medium batch
    test_frames_large = create_dummy_frames(100, 384, 640)  # Large batch
    
    print(f"📹 Created test frames: 20, 50, 100 frames")
    
    # Test optimized tracker
    print("\n🔥 Testing OPTIMIZED tracker...")
    tracker_optimized = Tracker(
        model_path=model_path,
        ball_model_path=ball_model_path if os.path.exists(ball_model_path) else None,
        enable_enhanced_ball_detection=os.path.exists(ball_model_path),
        enable_jersey_detection=False,  # Disable for pure inference testing
        device=device
    )
    
    results = {}
    
    # Test different frame counts
    for frame_count, frames in [("20 frames", test_frames_small), 
                               ("50 frames", test_frames_medium), 
                               ("100 frames", test_frames_large)]:
        results[frame_count] = benchmark_tracker_performance(
            tracker_optimized, frames, f"Optimized - {frame_count}"
        )
    
    # Print summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        print(f"{test_name:15} | {result['fps']:6.1f} FPS | {result['avg_time_per_frame']:6.1f}ms/frame")
    
    # Memory usage
    if device.type == "cuda":
        try:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"\n💾 GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        except Exception as e:
            print(f"⚠️ Could not get GPU memory info: {e}")


def test_optimization_features():
    """Test specific optimization features."""
    print("\n🔧 TESTING OPTIMIZATION FEATURES")
    print("=" * 60)
    
    device = get_optimal_device()
    model_path = "data/models/best_player_detect.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create test frames
    test_frames = create_dummy_frames(30, 384, 640)
    
    # Test with different optimization settings
    print("\n🧪 Testing Half Precision...")
    if device.type == "cuda":
        tracker = Tracker(model_path=model_path, device=device, enable_jersey_detection=False)
        print(f"Half precision enabled: {tracker.use_half_precision}")
        print(f"Optimized batch size: {tracker.optimized_batch_size}")
        print(f"Image size: {tracker.imgsz}")
        
        # Quick performance test
        start_time = time.time()
        _ = tracker.detect_frames(test_frames)
        end_time = time.time()
        
        fps = len(test_frames) / (end_time - start_time)
        print(f"Performance: {fps:.1f} FPS")
    else:
        print("⚠️ Half precision testing requires CUDA")


if __name__ == "__main__":
    try:
        test_batch_size_performance()
        test_optimization_features()
        print("\n✅ Performance testing completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
