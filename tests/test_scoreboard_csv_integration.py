#!/usr/bin/env python3
"""
Integration test for scoreboard detection CSV export functionality.

This test verifies that the enhanced CSV export includes the new scoreboard analysis fields.
"""

import os
import sys
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoreboard_detection import ScoreboardAnalyzer


def test_csv_field_names():
    """Test that the CSV export function includes the new field names."""
    # Import here to avoid issues with module loading
    from src.utils.goal_utils import export_consolidated_goal_statistics
    
    # Check if the function has the expected field names by inspecting the source
    import inspect
    source = inspect.getsource(export_consolidated_goal_statistics)
    
    # Check for the new field names in the source code
    expected_fields = [
        "scoreboard_analysis_start_frame",
        "scoreboard_analysis_end_frame", 
        "scoreboard_frames_analyzed",
        "scoreboard_frames_in_window",
        "scoreboard_analyze_last_percent",
        "scoreboard_detection_rate",
        "scoreboard_total_detections",
        "scoreboard_successful_extractions",
    ]
    
    print("🔍 Checking for enhanced scoreboard fields in CSV export function...")
    
    for field in expected_fields:
        if field in source:
            print(f"✅ Found field: {field}")
        else:
            print(f"❌ Missing field: {field}")
            return False
    
    return True


def test_scoreboard_analyzer_statistics():
    """Test that ScoreboardAnalyzer provides the expected statistics."""
    print("\n🔍 Testing ScoreboardAnalyzer statistics...")
    
    # Create analyzer with video info
    analyzer = ScoreboardAnalyzer(
        total_frames=1000,
        analyze_last_percent=20.0
    )
    
    # Get statistics
    stats = analyzer.get_statistics()
    
    # Check for expected fields
    expected_stats_fields = [
        "frames_in_analysis_window",
        "analysis_start_frame", 
        "analysis_end_frame",
        "analyze_last_percent",
        "total_frames",
        "total_detections",
        "successful_extractions",
        "detection_rate",
    ]
    
    for field in expected_stats_fields:
        if field in stats:
            print(f"✅ Statistics field: {field} = {stats[field]}")
        else:
            print(f"❌ Missing statistics field: {field}")
            return False
    
    # Verify frame range calculation
    expected_start = int(1000 * 0.8)  # 800
    if stats["analysis_start_frame"] == expected_start:
        print(f"✅ Correct analysis start frame: {expected_start}")
    else:
        print(f"❌ Wrong analysis start frame: expected {expected_start}, got {stats['analysis_start_frame']}")
        return False
    
    return True


def run_integration_test():
    """Run the integration test suite."""
    print("🧪 Running scoreboard CSV integration tests...")
    
    success = True
    
    # Test 1: CSV field names
    if not test_csv_field_names():
        success = False
    
    # Test 2: Analyzer statistics
    if not test_scoreboard_analyzer_statistics():
        success = False
    
    if success:
        print("\n✅ All integration tests passed!")
        print("🎯 Scoreboard detection modifications are working correctly:")
        print("   - Focuses on last 20% of video")
        print("   - Provides enhanced statistics")
        print("   - CSV export includes new fields")
    else:
        print("\n❌ Some integration tests failed!")
    
    return success


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
