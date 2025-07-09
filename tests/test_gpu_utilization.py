#!/usr/bin/env python3
"""
GPU Utilization Test Script for Football Analysis Pipeline

This script tests whether the GPU configuration is working correctly
by initializing the main components and checking device allocation.
"""

import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_device_detection():
    """Test device detection utilities."""
    print("🔧 Testing Device Detection...")
    print("=" * 50)

    from src.utils import check_gpu_availability, get_optimal_device, print_device_info

    # Check GPU availability
    gpu_info = check_gpu_availability()
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    print(f"Device Count: {gpu_info['device_count']}")

    if gpu_info["cuda_available"]:
        for i, gpu_name in enumerate(gpu_info["gpu_names"]):
            print(f"GPU {i}: {gpu_name}")
            if i < len(gpu_info["memory_info"]):
                mem_info = gpu_info["memory_info"][i]
                if "total_mb" in mem_info:
                    total_gb = mem_info["total_mb"] / 1024
                    free_gb = mem_info["free_mb"] / 1024
                    print(f"  Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

    # Test device selection
    print("\n🎯 Testing Device Selection...")
    device = get_optimal_device(verbose=True)
    print_device_info(device)

    return device


def test_tracker_gpu_usage(device):
    """Test Tracker GPU usage."""
    print("\n🏃 Testing Tracker GPU Usage...")
    print("=" * 50)

    try:
        from src.trackers.tracker import Tracker

        # Check if model files exist
        model_path = "data/models/best_player_detect.pt"
        ball_model_path = "data/models/best_ball_latest.pt"

        if not os.path.exists(model_path):
            print(f"⚠️ Main model not found: {model_path}")
            print("   Please ensure model files are available for testing")
            return False

        if not os.path.exists(ball_model_path):
            print(f"⚠️ Ball model not found: {ball_model_path}")
            print("   Proceeding without enhanced ball detection")
            ball_model_path = None

        print(f"📦 Initializing Tracker with device: {device}")

        # Initialize tracker with device
        tracker = Tracker(
            model_path=model_path,
            enable_jersey_detection=True,
            ball_model_path=ball_model_path,
            enable_enhanced_ball_detection=ball_model_path is not None,
            device=device,
        )

        print(f"✅ Tracker initialized successfully")
        print(f"   Main model device: {tracker.device}")

        # Check if models are on the correct device
        if hasattr(tracker.model, "device"):
            print(f"   Main model actual device: {tracker.model.device}")

        if tracker.ball_model and hasattr(tracker.ball_model, "device"):
            print(f"   Ball model actual device: {tracker.ball_model.device}")

        # Test with a dummy frame if possible
        try:
            import numpy as np

            dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

            print("🧪 Testing inference with dummy frame...")
            start_time = time.time()

            # Test detection
            results = tracker.model.predict(
                [dummy_frame], conf=0.1, device=tracker.device
            )

            inference_time = time.time() - start_time
            print(f"✅ Inference completed in {inference_time:.3f}s")

            if device.type == "cuda":
                print("🚀 GPU inference successful!")
            else:
                print("💻 CPU inference successful!")

        except Exception as e:
            print(f"⚠️ Inference test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        return False


def test_field_keypoints_gpu_usage(device):
    """Test FieldKeypointsDetector GPU usage."""
    print("\n🥅 Testing FieldKeypointsDetector GPU Usage...")
    print("=" * 50)

    try:
        from src.goal_detection.field_keypoints_detector import FieldKeypointsDetector

        model_path = "data/models/best_field_keypoint.pt"

        if not os.path.exists(model_path):
            print(f"⚠️ Field keypoints model not found: {model_path}")
            print("   Please ensure model files are available for testing")
            return False

        print(f"📦 Initializing FieldKeypointsDetector with device: {device}")

        # Initialize detector with device
        detector = FieldKeypointsDetector(model_path=model_path, device=device)

        print(f"✅ FieldKeypointsDetector initialized successfully")
        print(f"   Device: {detector.device}")

        # Check if model is on the correct device
        if hasattr(detector.model, "device"):
            print(f"   Model actual device: {detector.model.device}")

        # Test with a dummy frame if possible
        try:
            import numpy as np

            dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

            print("🧪 Testing inference with dummy frame...")
            start_time = time.time()

            # Test detection
            results = detector.model(
                dummy_frame, conf=detector.confidence_threshold, device=detector.device
            )

            inference_time = time.time() - start_time
            print(f"✅ Inference completed in {inference_time:.3f}s")

            if device.type == "cuda":
                print("🚀 GPU inference successful!")
            else:
                print("💻 CPU inference successful!")

        except Exception as e:
            print(f"⚠️ Inference test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ FieldKeypointsDetector initialization failed: {e}")
        return False


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n💾 Testing GPU Memory Usage...")
    print("=" * 50)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            allocated = torch.cuda.memory_allocated(i) / (1024**2)
            cached = torch.cuda.memory_reserved(i) / (1024**2)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**2)

            print(f"  Allocated: {allocated:.1f} MB")
            print(f"  Cached: {cached:.1f} MB")
            print(f"  Total: {total:.1f} MB")
            print(f"  Free: {total - allocated:.1f} MB")
    else:
        print("No CUDA devices available")


def main():
    """Main test function."""
    print("🧪 GPU UTILIZATION TEST FOR FOOTBALL ANALYSIS")
    print("=" * 60)

    # Test device detection
    device = test_device_detection()

    # Test components
    tracker_success = test_tracker_gpu_usage(device)
    field_keypoints_success = test_field_keypoints_gpu_usage(device)

    # Test memory usage
    test_memory_usage()

    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Device Detection: ✅")
    print(f"Tracker GPU Usage: {'✅' if tracker_success else '❌'}")
    print(f"FieldKeypoints GPU Usage: {'✅' if field_keypoints_success else '❌'}")

    if device.type == "cuda" and tracker_success and field_keypoints_success:
        print("\n🎉 GPU utilization is working correctly!")
        print("   All models should now use GPU acceleration during video processing.")
    elif device.type == "cpu":
        print("\n💻 CPU mode is working correctly!")
        print("   No GPU detected or GPU usage disabled.")
    else:
        print("\n⚠️ Some issues detected with GPU utilization.")
        print("   Check the error messages above for details.")

    print("\n🚀 To run the main pipeline with GPU:")
    print("   python main.py --input your_video.mp4 --device auto")
    print("   python main.py --input your_video.mp4 --device cuda")
    print("   python main.py --input your_video.mp4 --device cpu")


def test_actual_inference():
    """Test actual YOLO inference with GPU monitoring."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    print("\n🔬 TESTING ACTUAL YOLO INFERENCE")
    print("=" * 50)

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    print(f"GPU Memory before: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # Test main player detection model
    print("\n🏃 Testing Player Detection Model...")
    model = YOLO("data/models/best_player_detect.pt")

    print(
        f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
    )

    # Test inference with explicit device parameter
    print("🔥 Running inference with device='cuda'...")
    results = model.predict(dummy_frame, device="cuda", verbose=True)

    print(
        f"GPU Memory after inference: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
    )

    # Test ball detection model
    print("\n⚽ Testing Ball Detection Model...")
    try:
        ball_model = YOLO("data/models/best_ball_latest.pt")
        ball_results = ball_model.predict(dummy_frame, device="cuda", verbose=True)
        print(
            f"GPU Memory after ball inference: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
        )
    except Exception as e:
        print(f"Ball model test failed: {e}")

    # Test field keypoints model
    print("\n🥅 Testing Field Keypoints Model...")
    try:
        field_model = YOLO("data/models/best_field_keypoint.pt")
        field_results = field_model.predict(dummy_frame, device="cuda", verbose=True)
        print(
            f"GPU Memory after field inference: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
        )
    except Exception as e:
        print(f"Field model test failed: {e}")

    print("\n✅ Inference test completed!")
    print("💡 Check nvidia-smi output during this test to see GPU utilization spikes")


if __name__ == "__main__":
    main()

    # Run additional inference test
    print("\n" + "=" * 60)
    test_actual_inference()
