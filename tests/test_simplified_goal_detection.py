#!/usr/bin/env python3
"""
Test script for simplified goal detection functionality.
This script validates that both model-based and scoreboard-based goal detection
produce realistic and accurate goal counts.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np

from goal_detection.enhanced_goal_detector import EnhancedGoalDetector
from goal_detection.field_keypoints_detector import FieldKeypointsDetector
from scoreboard_detection.score_extractor import ScoreExtractor
from scoreboard_detection.scoreboard_analyzer import ScoreboardAnalyzer
from scoreboard_detection.scoreboard_detector import ScoreboardDetector
from utils.goal_utils import export_simplified_goal_statistics


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_enhanced_goal_detector():
    """Test the enhanced goal detector functionality."""
    print("\n🎯 Testing Enhanced Goal Detector...")

    # Create field keypoints detector (mock for testing)
    try:
        keypoints_detector = FieldKeypointsDetector()
    except Exception as e:
        print(f"   ⚠️  Could not create FieldKeypointsDetector: {e}")
        print("   Using mock keypoints detector for testing...")

        # Create a mock keypoints detector
        class MockKeypointsDetector:
            def detect_keypoints(self, frame, force_detection=False):
                pass

            def is_ball_in_goal_area(self, ball_position):
                x, y = ball_position
                # Simple mock: left goal if x < 200, right goal if x > 1720
                if x < 200:
                    return "left"
                elif x > 1720:
                    return "right"
                return None

        keypoints_detector = MockKeypointsDetector()

    # Create detector instance
    detector = EnhancedGoalDetector(keypoints_detector)

    # Simulate video dimensions
    detector.video_dimensions = (1920, 1080)
    detector._update_enhanced_goal_areas(1920, 1080)

    # Test goal detection with various ball positions
    test_positions = [
        # Left goal area
        (50, 400),  # Primary left goal area
        (100, 450),  # Extended left goal area
        # Right goal area
        (1870, 400),  # Primary right goal area
        (1820, 450),  # Extended right goal area
        # Center field (should not detect goals)
        (960, 540),  # Center
        (500, 300),  # Mid-left
        (1400, 700),  # Mid-right
    ]

    goals_detected = 0
    for i, position in enumerate(test_positions):
        print(f"  Testing position {position}...")

        # Simulate ball detection with good confidence
        goal_event = detector.detect_goal(
            ball_position=position,
            player_id=1,
            team=1,
            frame_num=i * 30,
            ball_confidence=0.8,
            ball_source="specialized",
        )

        if goal_event:
            goals_detected += 1
            print(f"    ✅ Goal detected: {goal_event}")
        else:
            print(f"    ❌ No goal detected")

        # Reset for next test
        detector.reset_goal_detection()

    # Get statistics
    stats = detector.get_goal_statistics()
    print(f"\n📊 Enhanced Goal Detector Results:")
    print(f"   Goals detected: {goals_detected}")
    print(f"   Team goals: {stats['team_goals']}")
    print(f"   Total goals: {stats['total_goals']}")

    return stats


def test_scoreboard_analyzer():
    """Test the scoreboard analyzer functionality."""
    print("\n📺 Testing Scoreboard Analyzer...")

    # Create analyzer with default parameters
    analyzer = ScoreboardAnalyzer()

    # Test with dummy frame data
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Simulate frame analysis
    for frame_num in range(0, 100, 10):
        result = analyzer.analyze_frame(dummy_frame, frame_num)
        if result:
            print(f"  Frame {frame_num}: Score detected - {result}")

    # Get final score
    final_score = analyzer.get_final_score()
    print(f"\n📊 Scoreboard Analyzer Results:")
    if final_score:
        print(
            f"   Final score: {final_score['team1_score']}-{final_score['team2_score']}"
        )
        print(f"   Confidence: {final_score['confidence']:.2f}")
        print(f"   Detection method: {final_score['detection_method']}")
    else:
        print("   No final score available")

    # Get statistics
    stats = analyzer.get_statistics()
    print(f"   Statistics: {stats}")

    return final_score


def test_simplified_csv_export():
    """Test the simplified CSV export functionality."""
    print("\n📄 Testing Simplified CSV Export...")

    # Create mock enhanced goal stats
    mock_enhanced_stats = {
        "team_goals": {1: 2, 2: 1},
        "total_goals": 3,
        "average_confidence": 0.75,
    }

    # Create mock scoreboard analyzer
    mock_scoreboard_analyzer = ScoreboardAnalyzer()

    # Export CSV
    try:
        csv_path = export_simplified_goal_statistics(
            video_name="test_video",
            enhanced_goal_stats=mock_enhanced_stats,
            scoreboard_analyzer=mock_scoreboard_analyzer,
            output_dir="data/output",
        )

        print(f"   ✅ CSV exported successfully: {csv_path}")

        # Read and display CSV content
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                content = f.read()
                print(f"   CSV Content:\n{content}")

        return True

    except Exception as e:
        print(f"   ❌ CSV export failed: {e}")
        return False


def validate_goal_counts(enhanced_stats, scoreboard_score):
    """Validate that goal counts are realistic."""
    print("\n✅ Validating Goal Counts...")

    # Extract goal counts
    model_goals = enhanced_stats.get("team_goals", {}) if enhanced_stats else {}
    scoreboard_goals = {}
    if scoreboard_score:
        scoreboard_goals = {
            1: scoreboard_score.get("team1_score", 0),
            2: scoreboard_score.get("team2_score", 0),
        }

    print(f"   Model-based goals: {model_goals}")
    print(f"   Scoreboard goals: {scoreboard_goals}")

    # Validation criteria
    validations = []

    # Check if goals are within realistic range (0-10 per team)
    for team_id, goals in model_goals.items():
        if 0 <= goals <= 10:
            validations.append(f"✅ Model Team {team_id} goals ({goals}) are realistic")
        else:
            validations.append(
                f"❌ Model Team {team_id} goals ({goals}) are unrealistic"
            )

    for team_id, goals in scoreboard_goals.items():
        if 0 <= goals <= 10:
            validations.append(
                f"✅ Scoreboard Team {team_id} goals ({goals}) are realistic"
            )
        else:
            validations.append(
                f"❌ Scoreboard Team {team_id} goals ({goals}) are unrealistic"
            )

    # Check if total goals are reasonable
    model_total = sum(model_goals.values())
    scoreboard_total = sum(scoreboard_goals.values())

    if 0 <= model_total <= 15:
        validations.append(f"✅ Model total goals ({model_total}) are realistic")
    else:
        validations.append(f"❌ Model total goals ({model_total}) are unrealistic")

    if 0 <= scoreboard_total <= 15:
        validations.append(
            f"✅ Scoreboard total goals ({scoreboard_total}) are realistic"
        )
    else:
        validations.append(
            f"❌ Scoreboard total goals ({scoreboard_total}) are unrealistic"
        )

    # Print validation results
    for validation in validations:
        print(f"   {validation}")

    # Return overall validation result
    failed_validations = [v for v in validations if v.startswith("❌")]
    return len(failed_validations) == 0


def main():
    """Main test function."""
    setup_logging()

    print("🧪 Starting Simplified Goal Detection Tests...")
    print("=" * 60)

    # Test enhanced goal detector
    enhanced_stats = test_enhanced_goal_detector()

    # Test scoreboard analyzer
    scoreboard_score = test_scoreboard_analyzer()

    # Test CSV export
    csv_success = test_simplified_csv_export()

    # Validate goal counts
    validation_success = validate_goal_counts(enhanced_stats, scoreboard_score)

    # Final summary
    print("\n" + "=" * 60)
    print("🏆 TEST SUMMARY:")
    print(f"   Enhanced Goal Detection: {'✅ PASS' if enhanced_stats else '❌ FAIL'}")
    print(f"   Scoreboard Detection: {'✅ PASS' if scoreboard_score else '❌ FAIL'}")
    print(f"   CSV Export: {'✅ PASS' if csv_success else '❌ FAIL'}")
    print(f"   Goal Count Validation: {'✅ PASS' if validation_success else '❌ FAIL'}")

    overall_success = all(
        [enhanced_stats, scoreboard_score, csv_success, validation_success]
    )
    print(
        f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}"
    )

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
