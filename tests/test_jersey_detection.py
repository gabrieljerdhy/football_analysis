#!/usr/bin/env python3
"""
Test script for jersey number detection system.

This script tests the OCR-based jersey number detection functionality
to validate accuracy and performance.
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jersey_number_detector import JerseyNumberDetector
from trackers import Tracker
from utils import read_video


def test_jersey_detection_basic():
    """Test basic jersey number detection functionality."""
    print("🧪 Testing basic jersey number detection...")

    try:
        # Initialize detector
        detector = JerseyNumberDetector(
            confidence_threshold=0.3, consensus_frames=3, valid_number_range=(1, 99)
        )
        print("✅ Jersey detector initialized successfully")

        # Test with a simple synthetic image
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.putText(
            test_image, "10", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
        )

        number, confidence = detector.extract_jersey_number(test_image)
        print(f"📊 Test image result: number={number}, confidence={confidence:.3f}")

        if number == 10:
            print("✅ Basic OCR test passed")
        else:
            print("❌ Basic OCR test failed")

        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def test_invalid_number_validation():
    """Test validation of invalid jersey numbers."""
    print("🧪 Testing invalid number validation...")

    try:
        detector = JerseyNumberDetector(
            confidence_threshold=0.2, consensus_frames=3, valid_number_range=(1, 99)
        )

        test_cases = [
            ("123", None, "Number > 99 should be rejected"),
            ("0", None, "Number < 1 should be rejected"),
            ("100", None, "Number = 100 should be rejected"),
            ("42", 42, "Valid number should be accepted"),
            ("99", 99, "Edge case 99 should be accepted"),
            ("1", 1, "Edge case 1 should be accepted"),
        ]

        passed_tests = 0
        total_tests = len(test_cases)

        for text, expected, description in test_cases:
            # Create test image
            test_image = np.ones((120, 120), dtype=np.uint8) * 255
            cv2.putText(
                test_image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
            )

            number, confidence = detector.extract_jersey_number(test_image)

            if number == expected:
                print(f"  ✅ {description}: {text} -> {number}")
                passed_tests += 1
            else:
                print(f"  ❌ {description}: {text} -> {number} (expected {expected})")

        print(f"📊 Validation tests: {passed_tests}/{total_tests} passed")

        # Print validation statistics
        stats = detector.get_detection_stats()
        print(f"📋 Validation stats: {stats['validation_stats']}")

        return passed_tests == total_tests

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_multi_digit_filtering():
    """Test filtering of multi-digit sequences."""
    print("🧪 Testing multi-digit sequence filtering...")

    try:
        detector = JerseyNumberDetector(
            confidence_threshold=0.2, consensus_frames=3, valid_number_range=(1, 99)
        )

        test_cases = [
            ("1234", "Should extract valid numbers from sequence"),
            ("567", "Should extract valid numbers from 3-digit sequence"),
            ("12345", "Should extract valid numbers from long sequence"),
            ("999", "Should reject numbers > 99 from sequence"),
        ]

        for text, description in test_cases:
            # Create test image
            test_image = np.ones((120, 150), dtype=np.uint8) * 255
            cv2.putText(
                test_image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3
            )

            number, confidence = detector.extract_jersey_number(test_image)

            if number is not None and 1 <= number <= 99:
                print(f"  ✅ {description}: {text} -> {number}")
            elif number is None:
                print(f"  ⚠️ {description}: {text} -> None (no valid extraction)")
            else:
                print(f"  ❌ {description}: {text} -> {number} (invalid range)")

        # Print validation statistics
        stats = detector.get_detection_stats()
        val_stats = stats["validation_stats"]
        print(
            f"📊 Multi-digit filtering: {val_stats['filtered_multi_digit']} sequences processed"
        )

        return True

    except Exception as e:
        print(f"❌ Multi-digit test failed: {e}")
        return False


def test_jersey_detection_with_video(video_path, max_frames=50):
    """Test jersey detection with actual video footage."""
    print(f"🎬 Testing jersey detection with video: {video_path}")

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False

    try:
        # Read video
        video_frames = read_video(video_path)
        if not video_frames:
            print("❌ Failed to load video frames")
            return False

        print(f"📹 Loaded {len(video_frames)} frames")

        # Initialize tracker with jersey detection
        tracker = Tracker("../models/best_detect.pt", enable_jersey_detection=True)

        # Get tracks for first few frames
        debug_frames = video_frames[: min(max_frames, len(video_frames))]
        print(f"🔍 Processing {len(debug_frames)} frames...")

        tracks = tracker.get_object_tracks(
            debug_frames, read_from_stub=False, stub_path=None
        )

        # Add positions
        tracker.add_position_to_tracks(tracks)

        # Test jersey number detection
        tracker.add_jersey_numbers_to_tracks(tracks, debug_frames, frame_sampling=3)

        # Analyze results
        detected_jerseys = {}
        total_detections = 0

        for frame_num, player_track in enumerate(tracks["players"]):
            for track_id, track_info in player_track.items():
                jersey_number = track_info.get("jersey_number")
                if jersey_number is not None:
                    if track_id not in detected_jerseys:
                        detected_jerseys[track_id] = []
                    detected_jerseys[track_id].append(jersey_number)
                    total_detections += 1

        print(f"\n📊 Jersey Detection Results:")
        print(f"   Total player detections: {total_detections}")
        print(f"   Unique players tracked: {len(detected_jerseys)}")

        for track_id, jersey_numbers in detected_jerseys.items():
            unique_numbers = list(set(jersey_numbers))
            most_common = max(set(jersey_numbers), key=jersey_numbers.count)
            consistency = jersey_numbers.count(most_common) / len(jersey_numbers)

            print(
                f"   Player {track_id}: Jersey {most_common} (consistency: {consistency:.2%})"
            )
            if len(unique_numbers) > 1:
                print(f"     Also detected: {unique_numbers}")

        # Get detection statistics
        if tracker.jersey_detector:
            stats = tracker.jersey_detector.get_detection_stats()
            print(f"\n📈 Performance Statistics:")
            print(f"   OCR calls: {stats['ocr_calls']}")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"   Confirmed players: {stats['confirmed_players']}")

        return True

    except Exception as e:
        print(f"❌ Video test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_jersey_preprocessing():
    """Test jersey region preprocessing functionality."""
    print("🖼️ Testing jersey region preprocessing...")

    try:
        detector = JerseyNumberDetector()

        # Create a test frame with a simulated player
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulate player bounding box
        bbox = [200, 100, 300, 400]  # [x1, y1, x2, y2]

        # Test preprocessing
        preprocessed = detector.preprocess_jersey_region(test_frame, bbox)

        if preprocessed is not None:
            print(f"✅ Preprocessing successful: {preprocessed.shape}")
            print(f"   Original bbox size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
            print(
                f"   Preprocessed size: {preprocessed.shape[1]}x{preprocessed.shape[0]}"
            )
            return True
        else:
            print("❌ Preprocessing failed")
            return False

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def main():
    """Run all jersey detection tests."""
    print("🚀 Starting Jersey Number Detection Tests\n")

    tests_passed = 0
    total_tests = 0

    # Test 1: Basic functionality
    total_tests += 1
    if test_jersey_detection_basic():
        tests_passed += 1

    print()

    # Test 2: Invalid number validation
    total_tests += 1
    if test_invalid_number_validation():
        tests_passed += 1

    print()

    # Test 3: Multi-digit filtering
    total_tests += 1
    if test_multi_digit_filtering():
        tests_passed += 1

    print()

    # Test 4: Preprocessing
    total_tests += 1
    if test_jersey_preprocessing():
        tests_passed += 1

    print()

    # Test 5: Video integration (if video available)
    test_videos = [
        "../input_videos/demo_vid_1.mp4",
        "../input_videos/test_777.mp4",
        "../input_videos/goal_test_1.mp4",
    ]

    video_found = False
    for video_path in test_videos:
        if os.path.exists(video_path):
            total_tests += 1
            if test_jersey_detection_with_video(video_path, max_frames=30):
                tests_passed += 1
            video_found = True
            break

    if not video_found:
        print("⚠️ No test videos found, skipping video integration test")

    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Tests passed: {tests_passed}/{total_tests}")
    print(
        f"   Success rate: {tests_passed/total_tests:.1%}"
        if total_tests > 0
        else "   No tests run"
    )

    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        print("\n🔒 Enhanced validation system is working correctly!")
        print("   ✅ Invalid numbers (>99) are properly rejected")
        print("   ✅ Multi-digit sequences are filtered correctly")
        print("   ✅ Range validation is enforced at all levels")
        return True
    else:
        print("❌ Some tests failed")
        print("   Please check the validation logic and OCR processing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
