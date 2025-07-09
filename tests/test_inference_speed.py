#!/usr/bin/env python3
"""
Quick inference speed test to compare optimized vs unoptimized YOLO inference.
"""

import os
import sys
import time
import numpy as np
import torch
from ultralytics import YOLO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import get_optimal_device


def create_test_frames(num_frames=50):
    """Create test frames similar to video resolution."""
    frames = []
    for i in range(num_frames):
        # Create frame similar to your video resolution
        frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def test_unoptimized_inference():
    """Test inference with small batches (original approach)."""
    print("🐌 Testing UNOPTIMIZED inference (small batches)...")
    
    device = get_optimal_device()
    model_path = "data/models/best_player_detect.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = YOLO(model_path)
    frames = create_test_frames(50)
    
    # Simulate original small batch processing
    batch_size = 10  # Original small batch size
    total_time = 0
    detections = []
    
    print(f"Processing {len(frames)} frames in batches of {batch_size}...")
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        start_time = time.time()
        batch_detections = model.predict(batch_frames, conf=0.1, device=device, verbose=False)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        detections.extend(batch_detections)
    
    fps = len(frames) / total_time if total_time > 0 else 0
    avg_time = (total_time / len(frames)) * 1000
    
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"⚡ Avg time per frame: {avg_time:.1f}ms")
    
    return {'fps': fps, 'avg_time': avg_time, 'total_time': total_time}


def test_optimized_inference():
    """Test inference with optimized settings."""
    print("\n🚀 Testing OPTIMIZED inference (large batches + optimizations)...")
    
    device = get_optimal_device()
    model_path = "data/models/best_player_detect.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = YOLO(model_path)
    frames = create_test_frames(50)
    
    # Optimized settings
    batch_size = 64 if device.type == "cuda" else 16
    use_half = device.type == "cuda"
    imgsz = 640
    
    # Enable half precision if possible
    if use_half and device.type == "cuda":
        try:
            model.model.half()
            print("✅ Half precision enabled")
        except Exception as e:
            print(f"⚠️ Half precision failed: {e}")
            use_half = False
    
    # Warmup
    print("🔥 Warming up model...")
    dummy_frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
    _ = model.predict(dummy_frame, verbose=False, device=device, imgsz=imgsz, half=use_half)
    
    print(f"Processing {len(frames)} frames in batches of {batch_size}...")
    
    total_time = 0
    detections = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        start_time = time.time()
        batch_detections = model.predict(
            batch_frames, 
            conf=0.1, 
            device=device, 
            imgsz=imgsz,
            half=use_half,
            verbose=False
        )
        end_time = time.time()
        
        total_time += (end_time - start_time)
        detections.extend(batch_detections)
    
    fps = len(frames) / total_time if total_time > 0 else 0
    avg_time = (total_time / len(frames)) * 1000
    
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"⚡ Avg time per frame: {avg_time:.1f}ms")
    
    return {'fps': fps, 'avg_time': avg_time, 'total_time': total_time}


def main():
    print("🔬 YOLO INFERENCE SPEED COMPARISON")
    print("=" * 50)
    
    device = get_optimal_device()
    print(f"🖥️  Device: {device}")
    
    if device.type == "cuda":
        print(f"🚀 GPU: {torch.cuda.get_device_name(device)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")
    
    # Test both approaches
    unoptimized_results = test_unoptimized_inference()
    optimized_results = test_optimized_inference()
    
    # Compare results
    if unoptimized_results and optimized_results:
        print("\n📊 COMPARISON RESULTS")
        print("=" * 50)
        print(f"{'Metric':<20} {'Unoptimized':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 65)
        
        fps_improvement = (optimized_results['fps'] / unoptimized_results['fps']) if unoptimized_results['fps'] > 0 else 0
        time_improvement = (unoptimized_results['avg_time'] / optimized_results['avg_time']) if optimized_results['avg_time'] > 0 else 0
        
        print(f"{'FPS':<20} {unoptimized_results['fps']:<15.1f} {optimized_results['fps']:<15.1f} {fps_improvement:<15.1f}x")
        print(f"{'Avg Time (ms)':<20} {unoptimized_results['avg_time']:<15.1f} {optimized_results['avg_time']:<15.1f} {time_improvement:<15.1f}x")
        
        print(f"\n🎯 Performance improvement: {fps_improvement:.1f}x faster!")
        
        if fps_improvement > 2:
            print("🔥 Excellent optimization!")
        elif fps_improvement > 1.5:
            print("✅ Good optimization!")
        else:
            print("⚠️ Modest improvement - check GPU utilization")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
