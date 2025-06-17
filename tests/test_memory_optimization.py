#!/usr/bin/env python3
"""
Test script to verify memory optimization changes work correctly.
This script tests that the main function can be called with camera movement
and speed/distance estimation disabled.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_function_import():
    """Test that main function can be imported with new parameters."""
    try:
        from main import main

        print("✓ Successfully imported main function")
        return True
    except Exception as e:
        print(f"✗ Failed to import main function: {e}")
        return False


def test_main_function_signature():
    """Test that main function accepts the new parameters."""
    try:
        import inspect

        from main import main

        # Get function signature
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())

        # Check if new parameters are present
        expected_params = [
            "input_video_path",
            "output_video_path",
            "use_stubs",
            "force_regenerate",
            "goals_config",
            "enable_camera_movement",
            "enable_speed_distance",
        ]

        for param in expected_params:
            if param not in params:
                print(f"✗ Missing parameter: {param}")
                return False

        print("✓ Main function has all expected parameters")
        print(f"  Parameters: {params}")
        return True

    except Exception as e:
        print(f"✗ Failed to check function signature: {e}")
        return False


def test_memory_optimization_flags():
    """Test that memory optimization flags work correctly."""
    try:
        from main import main

        # Mock all the heavy dependencies to avoid actual processing
        with patch("main.read_video") as mock_read_video, patch(
            "main.Tracker"
        ) as mock_tracker, patch(
            "main.FieldKeypointsDetector"
        ) as mock_field_detector, patch(
            "main.GoalDetector"
        ) as mock_goal_detector, patch(
            "main.CameraMovementEstimator"
        ) as mock_camera, patch(
            "main.SpeedAndDistance_Estimator"
        ) as mock_speed, patch(
            "main.save_video"
        ) as mock_save_video, patch(
            "main.os.makedirs"
        ), patch(
            "main.load_manual_goals"
        ) as mock_load_goals, patch(
            "builtins.open", create=True
        ) as mock_open, patch(
            "main.csv.DictWriter"
        ) as mock_csv_writer:

            # Setup mocks
            mock_read_video.return_value = [
                MagicMock() for _ in range(10)
            ]  # 10 dummy frames
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance
            mock_tracker_instance.get_object_tracks.return_value = {
                "players": [{} for _ in range(10)],
                "ball": [{1: {"bbox": [0, 0, 10, 10]}} for _ in range(10)],
                "referees": [{} for _ in range(10)],
            }
            mock_tracker_instance.interpolate_ball_positions.return_value = [
                {1: {"bbox": [0, 0, 10, 10], "position": (5, 5)}} for _ in range(10)
            ]
            mock_tracker_instance.draw_annotations.return_value = [
                MagicMock() for _ in range(10)
            ]

            mock_field_detector_instance = MagicMock()
            mock_field_detector.return_value = mock_field_detector_instance

            mock_goal_detector_instance = MagicMock()
            mock_goal_detector.return_value = mock_goal_detector_instance
            mock_goal_detector_instance.get_goal_statistics.return_value = {
                "team_goals": {1: 0, 2: 0},
                "player_goals": {},
                "total_goals": 0,
                "goal_events": [],
            }
            mock_goal_detector_instance.draw_goal_info.return_value = MagicMock()

            mock_load_goals.return_value = []

            # Mock CSV writer
            mock_csv_writer_instance = MagicMock()
            mock_csv_writer.return_value = mock_csv_writer_instance

            # Test with both flags disabled (default)
            print("\n🧪 Testing with memory optimization enabled (default)...")
            try:
                main(
                    input_video_path="dummy_video.mp4",
                    output_video_path="dummy_output.avi",
                    enable_camera_movement=False,
                    enable_speed_distance=False,
                )
                print("✓ Main function executed successfully with memory optimization")

                # Verify camera movement and speed estimators were not called
                mock_camera.assert_not_called()
                mock_speed.assert_not_called()
                print("✓ Camera movement and speed estimators were not initialized")

            except Exception as e:
                print(f"✗ Failed with memory optimization: {e}")
                return False

            # Reset mocks
            mock_camera.reset_mock()
            mock_speed.reset_mock()

            # Test with both flags enabled
            print("\n🧪 Testing with camera movement and speed estimation enabled...")
            try:
                main(
                    input_video_path="dummy_video.mp4",
                    output_video_path="dummy_output.avi",
                    enable_camera_movement=True,
                    enable_speed_distance=True,
                )
                print("✓ Main function executed successfully with features enabled")

                # Verify camera movement and speed estimators were called
                mock_camera.assert_called_once()
                mock_speed.assert_called_once()
                print(
                    "✓ Camera movement and speed estimators were initialized when enabled"
                )

            except Exception as e:
                print(f"✗ Failed with features enabled: {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Failed to test memory optimization flags: {e}")
        return False


def main_test():
    """Run all tests."""
    print("🧪 Testing Memory Optimization Changes")
    print("=" * 50)

    tests = [
        test_main_function_import,
        test_main_function_signature,
        test_memory_optimization_flags,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"❌ {test.__name__} failed")

    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Memory optimization is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main_test()
    sys.exit(0 if success else 1)
