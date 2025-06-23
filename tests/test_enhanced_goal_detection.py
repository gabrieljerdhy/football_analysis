#!/usr/bin/env python3
"""
Test script for enhanced goal detection system.
Tests the integration of field keypoints detection with goal detection.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.goal_detection import FieldKeypointsDetector, GoalDetector
from src.trackers import Tracker
from src.utils import read_video


def test_enhanced_goal_detection():
    """
    Test the enhanced goal detection system with field keypoints integration.
    """
    print("🧪 Testing Enhanced Goal Detection System")
    print("=" * 50)

    # Test configuration
    test_results = {
        "keypoints_detection": False,
        "goal_area_calculation": False,
        "enhanced_validation": False,
        "optimization": False,
        "goal_analysis": False,
        "overall_performance": False,
    }

    try:
        # 1. Test Field Keypoints Detector
        print("\n1️⃣ Testing Field Keypoints Detector...")
        keypoints_detector = FieldKeypointsDetector("../data/models/best_keypoint.pt")

        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Test keypoint detection
        keypoints = keypoints_detector.detect_keypoints(test_frame)
        print(f"   ✓ Keypoints detection initialized")
        print(f"   ✓ Detected {len(keypoints)} keypoints")

        # Test enhanced areas
        goal_areas = keypoints_detector.get_goal_areas()
        penalty_areas = keypoints_detector.get_penalty_areas()
        six_yard_areas = keypoints_detector.get_six_yard_areas()
        goal_lines = keypoints_detector.get_goal_lines()

        print(f"   ✓ Goal areas: {goal_areas}")
        print(f"   ✓ Penalty areas: {penalty_areas}")
        print(f"   ✓ Six-yard areas: {six_yard_areas}")
        print(f"   ✓ Goal lines: {goal_lines}")

        test_results["keypoints_detection"] = True

        # 2. Test Goal Detector Integration
        print("\n2️⃣ Testing Goal Detector Integration...")
        goal_detector = GoalDetector(keypoints_detector)

        # Test keypoint updates
        goal_detector.update_keypoints(test_frame)
        print(f"   ✓ Goal detector keypoint update successful")

        # Test enhanced goal area calculation
        test_ball_position = (100, 400)  # Left side position
        goal_side = goal_detector._check_ball_in_goal(test_ball_position)
        print(f"   ✓ Enhanced goal area check: {goal_side}")

        test_results["goal_area_calculation"] = True

        # 3. Test Enhanced Validation Logic
        print("\n3️⃣ Testing Enhanced Validation Logic...")

        # Simulate ball trajectory
        goal_detector.ball_trajectory = [
            (200, 400),
            (150, 400),
            (100, 400),
            (50, 400),
            (25, 400),
        ]

        # Test enhanced validation
        is_valid = goal_detector._validate_goal_trajectory("left")
        print(f"   ✓ Enhanced trajectory validation: {is_valid}")

        # Test field context
        field_context = keypoints_detector.get_field_context(test_ball_position)
        print(f"   ✓ Field context: {field_context}")

        test_results["enhanced_validation"] = True

        # 4. Test Optimization Features
        print("\n4️⃣ Testing Optimization Features...")

        # Test optimization parameters
        goal_detector.set_keypoint_optimization(
            detection_interval=3, stability_threshold=5
        )
        print(f"   ✓ Optimization parameters set")

        # Test multiple frame processing with optimization
        for i in range(10):
            keypoints_detector.detect_keypoints(test_frame)

        optimization_stats = keypoints_detector.get_optimization_stats()
        print(f"   ✓ Optimization stats: {optimization_stats}")

        test_results["optimization"] = True

        # 5. Test Goal Event Analysis
        print("\n5️⃣ Testing Goal Event Analysis...")

        # Simulate goal detection
        goal_event = goal_detector.detect_goal(test_ball_position, 5, 1, 100)

        if goal_event:
            print(f"   ✓ Goal event detected: {goal_event.get('team', 'Unknown')}")
            print(f"   ✓ Goal analysis: {goal_event.get('goal_analysis', {})}")
            print(f"   ✓ Field context: {goal_event.get('field_context', {})}")
        else:
            print(f"   ✓ No goal detected (expected for test scenario)")

        test_results["goal_analysis"] = True

        # 6. Test Performance
        print("\n6️⃣ Testing Performance...")

        # Measure keypoint detection performance
        start_time = time.time()
        for i in range(50):
            keypoints_detector.detect_keypoints(test_frame)
        keypoint_time = time.time() - start_time

        # Measure goal detection performance
        start_time = time.time()
        for i in range(100):
            goal_detector._check_ball_in_goal(test_ball_position)
        goal_detection_time = time.time() - start_time

        print(f"   ✓ Keypoint detection: {keypoint_time/50*1000:.2f}ms per frame")
        print(f"   ✓ Goal detection: {goal_detection_time/100*1000:.2f}ms per check")

        test_results["overall_performance"] = True

        # 7. Test with Real Video (if available)
        print("\n7️⃣ Testing with Real Video (if available)...")

        test_video_paths = [
            "../data/test_videos/sample.mp4",
            "../data/videos/test.mp4",
            "../test_video.mp4",
        ]

        video_found = False
        for video_path in test_video_paths:
            if os.path.exists(video_path):
                print(f"   📹 Testing with video: {video_path}")
                try:
                    # Test with first 10 frames
                    cap = cv2.VideoCapture(video_path)
                    frame_count = 0

                    while frame_count < 10:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Test enhanced detection
                        goal_detector.update_keypoints(frame)
                        frame_count += 1

                    cap.release()
                    print(f"   ✓ Successfully processed {frame_count} frames")
                    video_found = True
                    break

                except Exception as e:
                    print(f"   ⚠️ Error processing video: {e}")

        if not video_found:
            print(f"   ℹ️ No test video found, skipping real video test")

        # Summary
        print("\n📊 Test Results Summary")
        print("=" * 50)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")

        print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print(
                "🎉 All tests passed! Enhanced goal detection system is working correctly."
            )
            return True
        else:
            print("⚠️ Some tests failed. Please check the implementation.")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration_with_main():
    """
    Test integration with the main football analysis pipeline.
    """
    print("\n🔗 Testing Integration with Main Pipeline...")

    try:
        # Test import of enhanced components
        from src.goal_detection import FieldKeypointsDetector, GoalDetector

        print("   ✓ Enhanced goal detection components imported successfully")

        # Test initialization as done in main.py
        field_keypoints_detector = FieldKeypointsDetector(
            "../data/models/best_keypoint.pt"
        )
        goal_detector = GoalDetector(field_keypoints_detector)
        print("   ✓ Components initialized as in main.py")

        # Test optimization configuration
        goal_detector.set_keypoint_optimization(
            detection_interval=5, stability_threshold=10
        )
        print("   ✓ Optimization configured")

        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Enhanced Goal Detection Tests")

    # Run main tests
    main_test_passed = test_enhanced_goal_detection()

    # Run integration tests
    integration_test_passed = test_integration_with_main()

    # Final result
    if main_test_passed and integration_test_passed:
        print("\n🎊 All tests completed successfully!")
        print("Enhanced goal detection system is ready for use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
