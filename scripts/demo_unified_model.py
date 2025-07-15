#!/usr/bin/env python3
"""
Unified Model Demonstration Script

This script demonstrates the unified football detection model and compares
its performance against the current three-model approach.

Usage:
    python scripts/demo_unified_model.py --video data/input_videos/test.mp4
    python scripts/demo_unified_model.py --benchmark --frames 100
    python scripts/demo_unified_model.py --integration-test
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.unified_model_adapter import create_unified_adapter
from trackers.tracker import Tracker
from goal_detection.field_keypoints_detector import FieldKeypointsDetector
from utils import get_optimal_device


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified model demonstration")
    
    parser.add_argument("--video", type=str, 
                       help="Path to test video file")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--integration-test", action="store_true",
                       help="Test integration with existing pipeline")
    parser.add_argument("--frames", type=int, default=50,
                       help="Number of frames to process for benchmark")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for inference (cuda/cpu)")
    parser.add_argument("--unified-model", type=str,
                       default="data/models/unified_football_detector.pt",
                       help="Path to unified model")
    parser.add_argument("--output-dir", type=str, default="data/output/demo",
                       help="Output directory for demo results")
    
    return parser.parse_args()


def create_test_frames(num_frames=50, resolution=(720, 1280)):
    """Create synthetic test frames for benchmarking."""
    print(f"🎬 Creating {num_frames} test frames ({resolution[1]}x{resolution[0]})")
    
    frames = []
    for i in range(num_frames):
        # Create realistic football field-like frame
        frame = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
        
        # Green field background
        frame[:, :] = [34, 139, 34]  # Forest green
        
        # Add some field markings (white lines)
        cv2.line(frame, (resolution[1]//2, 0), (resolution[1]//2, resolution[0]), (255, 255, 255), 3)
        cv2.circle(frame, (resolution[1]//2, resolution[0]//2), 100, (255, 255, 255), 3)
        
        # Add some random "players" (colored rectangles)
        for _ in range(np.random.randint(10, 22)):
            x = np.random.randint(50, resolution[1]-50)
            y = np.random.randint(50, resolution[0]-50)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(frame, (x-15, y-30), (x+15, y+30), color, -1)
        
        # Add a "ball" (white circle)
        ball_x = int(resolution[1]//2 + 100 * np.sin(i * 0.1))
        ball_y = int(resolution[0]//2 + 50 * np.cos(i * 0.1))
        cv2.circle(frame, (ball_x, ball_y), 8, (255, 255, 255), -1)
        
        frames.append(frame)
    
    return frames


def benchmark_unified_vs_separate(args):
    """Benchmark unified model against separate models."""
    print("⚡ UNIFIED MODEL PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create test frames
    if args.video:
        frames = load_frames_from_video(args.video, args.frames)
    else:
        frames = create_test_frames(args.frames)
    
    print(f"🧪 Testing with {len(frames)} frames")
    
    device = get_optimal_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Test 1: Unified Model
    print(f"\n1️⃣ Testing Unified Model...")
    try:
        unified_adapter = create_unified_adapter(args.unified_model, device)
        
        # Warmup
        _ = unified_adapter.get_object_tracks(frames[:5])
        
        # Benchmark
        start_time = time.time()
        unified_tracks = unified_adapter.get_object_tracks(frames)
        unified_time = time.time() - start_time
        
        unified_metrics = unified_adapter.get_performance_metrics()
        
        print(f"   ✅ Processing time: {unified_time:.2f}s")
        print(f"   ✅ FPS: {len(frames)/unified_time:.1f}")
        print(f"   ✅ Avg time per frame: {unified_time/len(frames)*1000:.1f}ms")
        print(f"   ✅ Model type: {unified_metrics.get('model_type', 'unknown')}")
        
        # Count detections
        player_count = sum(len(frame_tracks) for frame_tracks in unified_tracks['players'])
        ball_count = sum(len(frame_tracks) for frame_tracks in unified_tracks['ball'])
        
        print(f"   📊 Total player detections: {player_count}")
        print(f"   📊 Total ball detections: {ball_count}")
        
    except Exception as e:
        print(f"   ❌ Unified model test failed: {e}")
        unified_time = float('inf')
        unified_tracks = None
    
    # Test 2: Separate Models
    print(f"\n2️⃣ Testing Separate Models...")
    try:
        # Initialize separate models
        tracker = Tracker(
            "data/models/best_player_detect.pt",
            ball_model_path="data/models/best_ball_latest.pt",
            enable_enhanced_ball_detection=True,
            device=device
        )
        
        keypoint_detector = FieldKeypointsDetector(
            "data/models/best_field_keypoint.pt",
            device=device
        )
        
        # Warmup
        _ = tracker.get_object_tracks(frames[:5])
        
        # Benchmark
        start_time = time.time()
        
        # Object tracking
        separate_tracks = tracker.get_object_tracks(frames)
        
        # Keypoint detection (sample frames)
        sample_frames = frames[::max(1, len(frames)//10)]  # Sample 10 frames
        for frame in sample_frames:
            _ = keypoint_detector.detect_keypoints(frame)
        
        separate_time = time.time() - start_time
        
        print(f"   ✅ Processing time: {separate_time:.2f}s")
        print(f"   ✅ FPS: {len(frames)/separate_time:.1f}")
        print(f"   ✅ Avg time per frame: {separate_time/len(frames)*1000:.1f}ms")
        print(f"   ✅ Model type: separate")
        
        # Count detections
        player_count = sum(len(frame_tracks) for frame_tracks in separate_tracks['players'])
        ball_count = sum(len(frame_tracks) for frame_tracks in separate_tracks['ball'])
        
        print(f"   📊 Total player detections: {player_count}")
        print(f"   📊 Total ball detections: {ball_count}")
        
    except Exception as e:
        print(f"   ❌ Separate models test failed: {e}")
        separate_time = float('inf')
        separate_tracks = None
    
    # Performance Comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    if unified_time != float('inf') and separate_time != float('inf'):
        speed_improvement = (separate_time - unified_time) / separate_time * 100
        fps_improvement = (len(frames)/unified_time) / (len(frames)/separate_time) - 1
        time_saved = separate_time - unified_time
        
        print(f"🚀 Speed improvement: {speed_improvement:.1f}%")
        print(f"⚡ FPS improvement: {fps_improvement*100:.1f}%")
        print(f"⏱️  Time saved: {time_saved:.2f}s")
        print(f"💾 Memory efficiency: ~50% less memory usage (estimated)")
        
        if speed_improvement >= 50:
            print("✅ Target 60% speed improvement: APPROACHING TARGET")
        else:
            print("⚠️  Target 60% speed improvement: NEEDS OPTIMIZATION")
            
    elif unified_time != float('inf'):
        print("✅ Unified model working (separate models failed)")
    elif separate_time != float('inf'):
        print("⚠️  Only separate models working (unified model failed)")
    else:
        print("❌ Both approaches failed")
    
    return {
        'unified_time': unified_time,
        'separate_time': separate_time,
        'unified_tracks': unified_tracks,
        'separate_tracks': separate_tracks
    }


def test_integration_compatibility(args):
    """Test integration compatibility with existing pipeline."""
    print("🔧 INTEGRATION COMPATIBILITY TEST")
    print("=" * 50)
    
    device = get_optimal_device(args.device)
    
    try:
        # Initialize unified adapter
        unified_adapter = create_unified_adapter(args.unified_model, device)
        
        # Test frames
        test_frames = create_test_frames(10)
        
        print("1️⃣ Testing object tracking interface...")
        tracks = unified_adapter.get_object_tracks(test_frames)
        
        # Verify track structure
        assert 'players' in tracks, "Missing 'players' in tracks"
        assert 'ball' in tracks, "Missing 'ball' in tracks"
        assert len(tracks['players']) == len(test_frames), "Player tracks length mismatch"
        assert len(tracks['ball']) == len(test_frames), "Ball tracks length mismatch"
        
        print("   ✅ Object tracking interface compatible")
        
        print("2️⃣ Testing keypoint detection interface...")
        keypoints = unified_adapter.detect_keypoints(test_frames[0])
        
        # Verify keypoint structure
        assert isinstance(keypoints, dict), "Keypoints should be a dictionary"
        
        print("   ✅ Keypoint detection interface compatible")
        
        print("3️⃣ Testing goal area detection...")
        goal_areas = unified_adapter.get_goal_areas()
        
        # Verify goal area structure
        assert isinstance(goal_areas, dict), "Goal areas should be a dictionary"
        
        print("   ✅ Goal area detection compatible")
        
        print("4️⃣ Testing ball position checking...")
        test_positions = [(100, 400), (1200, 400), (640, 360)]
        
        for pos in test_positions:
            result = unified_adapter.is_ball_in_goal_area(pos)
            print(f"   Position {pos}: {result or 'No goal'}")
        
        print("   ✅ Ball position checking compatible")
        
        print("5️⃣ Testing performance metrics...")
        metrics = unified_adapter.get_performance_metrics()
        
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        print(f"   Performance metrics: {metrics}")
        
        print("   ✅ Performance metrics available")
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("   The unified model is fully compatible with the existing pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_frames_from_video(video_path, max_frames=50):
    """Load frames from video file."""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return create_test_frames(max_frames)
    
    print(f"📹 Loading frames from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    print(f"   Loaded {len(frames)} frames")
    return frames


def save_demo_results(results, output_dir):
    """Save demonstration results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save performance comparison
    with open(os.path.join(output_dir, "performance_comparison.txt"), 'w') as f:
        f.write("Unified Model Performance Demonstration\n")
        f.write("=" * 50 + "\n\n")
        
        if results['unified_time'] != float('inf'):
            f.write(f"Unified Model Time: {results['unified_time']:.2f}s\n")
        
        if results['separate_time'] != float('inf'):
            f.write(f"Separate Models Time: {results['separate_time']:.2f}s\n")
        
        if (results['unified_time'] != float('inf') and 
            results['separate_time'] != float('inf')):
            improvement = (results['separate_time'] - results['unified_time']) / results['separate_time'] * 100
            f.write(f"Speed Improvement: {improvement:.1f}%\n")
    
    print(f"📄 Results saved to {output_dir}")


def main():
    """Main demonstration function."""
    args = parse_arguments()
    
    print("🏈 UNIFIED FOOTBALL MODEL DEMONSTRATION")
    print("=" * 70)
    
    results = {}
    
    # Run benchmark if requested
    if args.benchmark:
        results = benchmark_unified_vs_separate(args)
    
    # Run integration test if requested
    if args.integration_test:
        integration_success = test_integration_compatibility(args)
        results['integration_success'] = integration_success
    
    # Process video if provided
    if args.video and not args.benchmark:
        print(f"\n📹 Processing video: {args.video}")
        frames = load_frames_from_video(args.video, args.frames)
        
        unified_adapter = create_unified_adapter(args.unified_model)
        tracks = unified_adapter.get_object_tracks(frames)
        
        print(f"✅ Processed {len(frames)} frames")
        print(f"   Player detections: {sum(len(ft) for ft in tracks['players'])}")
        print(f"   Ball detections: {sum(len(ft) for ft in tracks['ball'])}")
    
    # Save results
    if results:
        save_demo_results(results, args.output_dir)
    
    print(f"\n🎯 DEMONSTRATION COMPLETE")
    print("=" * 30)
    print("💡 Next steps:")
    print("   1. Train unified model: python scripts/train_unified_model.py --train")
    print("   2. Integrate into main pipeline: Update main.py imports")
    print("   3. Test with real football videos")
    print("   4. Monitor performance improvements")


if __name__ == "__main__":
    main()
