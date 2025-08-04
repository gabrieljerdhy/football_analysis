#!/usr/bin/env python3
"""
Test script to verify logging integration with the main analysis pipeline.

This script tests that the logging system properly integrates with the main
football analysis pipeline and captures all expected information.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_utils import create_analysis_logger, get_analysis_logger


def test_logging_system():
    """Test the logging system functionality."""
    
    print("🧪 Testing Football Analysis Logging System...")
    
    # Test 1: Logger Creation
    print("\n1️⃣ Testing logger creation...")
    
    test_video_path = "data/input_videos/test_match.mp4"
    logger = create_analysis_logger(
        input_video_path=test_video_path,
        output_dir="data/output",
        match_name="Test Match",
        enable_console=False
    )
    
    assert logger is not None, "Logger should be created successfully"
    assert os.path.exists(logger.log_file_path), "Log file should be created"
    print(f"✅ Logger created: {logger.log_file_path}")
    
    # Test 2: Configuration Logging
    print("\n2️⃣ Testing configuration logging...")
    
    config = {
        'input_video_path': test_video_path,
        'memory_efficient': True,
        'batch_size': 50,
        'enable_scoreboard_detection': True
    }
    logger.set_configuration(config)
    
    assert logger.analysis_context['configuration'] == config, "Configuration should be stored"
    print("✅ Configuration logged successfully")
    
    # Test 3: Video Metadata Logging
    print("\n3️⃣ Testing video metadata logging...")
    
    metadata = {
        'width': 1280,
        'height': 720,
        'total_frames': 3600,
        'duration_seconds': 150.0,
        'fps': 24.0
    }
    logger.set_input_video(test_video_path, metadata)
    
    assert logger.analysis_context['video_metadata'] == metadata, "Metadata should be stored"
    print("✅ Video metadata logged successfully")
    
    # Test 4: Stage Logging
    print("\n4️⃣ Testing stage logging...")
    
    logger.start_stage("test_stage", "Testing stage functionality")
    logger.end_stage("test_stage", {"test_result": "success"})
    
    assert len(logger.performance_metrics['stages_completed']) > 0, "Stage should be recorded"
    print("✅ Stage logging works correctly")
    
    # Test 5: Detection Event Logging
    print("\n5️⃣ Testing detection event logging...")
    
    logger.log_detection_result("goal", 1500, 0.92, {"team": 1, "player_id": 10})
    logger.log_detection_result("pass", 2000, 0.78, {"team": 2, "player_id": 7})
    
    print("✅ Detection events logged successfully")
    
    # Test 6: Performance Metrics
    print("\n6️⃣ Testing performance metrics...")
    
    logger.log_performance_metric("test_metric", 42.5)
    logger.log_memory_usage(3.2, "test_stage")
    
    assert logger.performance_metrics['test_metric'] == 42.5, "Performance metric should be stored"
    print("✅ Performance metrics logged successfully")
    
    # Test 7: Team Statistics
    print("\n7️⃣ Testing team statistics logging...")
    
    team_stats = {
        "team_1": {"goals": 2, "passes": 120},
        "team_2": {"goals": 1, "passes": 95}
    }
    logger.log_team_statistics(team_stats)
    
    assert logger.analysis_context['team_statistics'] == team_stats, "Team stats should be stored"
    print("✅ Team statistics logged successfully")
    
    # Test 8: Output Files Logging
    print("\n8️⃣ Testing output files logging...")
    
    output_files = {
        "output_video": "test_output.avi",
        "team_csv": "test_team_stats.csv",
        "player_csv": "test_player_stats.csv"
    }
    logger.log_output_files(output_files)
    
    assert logger.analysis_context['output_files'] == output_files, "Output files should be stored"
    print("✅ Output files logged successfully")
    
    # Test 9: Error and Warning Logging
    print("\n9️⃣ Testing error and warning logging...")
    
    initial_errors = logger.performance_metrics['errors_encountered']
    initial_warnings = logger.performance_metrics['warnings_encountered']
    
    logger.log_error("Test error message")
    logger.log_warning("Test warning message")
    
    assert logger.performance_metrics['errors_encountered'] == initial_errors + 1, "Error count should increase"
    assert logger.performance_metrics['warnings_encountered'] == initial_warnings + 1, "Warning count should increase"
    print("✅ Error and warning logging works correctly")
    
    # Test 10: Analysis Finalization
    print("\n🔟 Testing analysis finalization...")
    
    final_results = {
        "total_frames": 3600,
        "goals_detected": 3,
        "processing_time": 125.5
    }
    
    logger.finalize_analysis(final_results)
    
    # Check that summary file was created
    summary_path = logger.log_file_path.replace('.log', '_summary.json')
    assert os.path.exists(summary_path), "Summary JSON file should be created"
    
    # Verify summary content
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    assert 'analysis_info' in summary_data, "Summary should contain analysis info"
    assert 'performance_metrics' in summary_data, "Summary should contain performance metrics"
    assert 'analysis_context' in summary_data, "Summary should contain analysis context"
    assert summary_data['analysis_context']['final_results'] == final_results, "Final results should be stored"
    
    print("✅ Analysis finalization works correctly")
    print(f"📋 Summary file created: {summary_path}")
    
    # Test 11: File Content Verification
    print("\n1️⃣1️⃣ Testing log file content...")
    
    # Read the log file and verify it contains expected content
    with open(logger.log_file_path, 'r') as f:
        log_content = f.read()
    
    expected_content = [
        "FOOTBALL ANALYSIS STARTED",
        "Test Match",
        "Analysis Configuration",
        "Video Metadata",
        "Starting Stage: test_stage",
        "Completed Stage: test_stage",
        "goal detected at frame 1500",
        "pass detected at frame 2000",
        "Team Statistics",
        "Output Files Generated",
        "ERROR: Test error message",
        "WARNING: Test warning message",
        "FOOTBALL ANALYSIS COMPLETED"
    ]
    
    for content in expected_content:
        assert content in log_content, f"Log should contain: {content}"
    
    print("✅ Log file contains all expected content")
    
    print(f"\n🎉 All tests passed! Logging system is working correctly.")
    print(f"📝 Test log file: {logger.log_file_path}")
    print(f"📋 Test summary file: {summary_path}")
    
    return True


def test_convenience_functions():
    """Test the convenience logging functions."""
    
    print("\n🧪 Testing convenience logging functions...")
    
    from utils.logging_utils import (
        set_global_logger,
        log_detection_event,
        log_performance_metric,
        log_memory_usage,
        log_scoreboard_event
    )
    
    # Create a logger and set it as global
    logger = create_analysis_logger(
        input_video_path="data/input_videos/convenience_test.mp4",
        output_dir="data/output",
        match_name="Convenience Test",
        enable_console=False
    )
    set_global_logger(logger)
    
    # Test convenience functions
    log_detection_event("goal", 1000, 0.95, {"team": 1})
    log_performance_metric("test_fps", 30.0)
    log_memory_usage(2.5, "test")
    log_scoreboard_event(2000, "1-0", 0.88, True)
    
    logger.finalize_analysis({"test": "convenience_functions"})
    
    print("✅ Convenience functions work correctly")
    print(f"📝 Convenience test log: {logger.log_file_path}")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)
    
    try:
        # Run main tests
        test_logging_system()
        
        # Run convenience function tests
        test_convenience_functions()
        
        print("\n🏆 All logging system tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
