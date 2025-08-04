#!/usr/bin/env python3
"""
Demo script showing the comprehensive logging system for football analysis.

This script demonstrates how to use the logging system to track analysis runs,
performance metrics, and debugging information.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_utils import (
    create_analysis_logger,
    set_global_logger,
    log_detection_event,
    log_performance_metric,
    log_memory_usage,
    log_scoreboard_event
)


def demo_logging_system():
    """Demonstrate the comprehensive logging system."""
    
    # Create a demo video path
    demo_video_path = "data/input_videos/demo_match.mp4"
    
    # Initialize the logging system
    print("🚀 Initializing Football Analysis Logging Demo...")
    
    # Create analysis logger
    logger = create_analysis_logger(
        input_video_path=demo_video_path,
        output_dir="data/output",
        match_name="Demo Match",
        enable_console=True  # Enable console output for demo
    )
    
    # Set as global logger
    set_global_logger(logger)
    
    # Log configuration
    config = {
        'input_video_path': demo_video_path,
        'output_video_path': 'data/output_videos/demo_match_output.avi',
        'use_stubs': True,
        'enable_camera_movement': False,
        'enable_speed_distance': False,
        'enable_scoreboard_detection': True,
        'memory_efficient': True,
        'batch_size': 100,
        'device': 'cuda'
    }
    logger.set_configuration(config)
    
    # Simulate video metadata
    video_metadata = {
        'width': 1920,
        'height': 1080,
        'total_frames': 7200,
        'duration_seconds': 300.0,
        'fps': 24.0
    }
    logger.set_input_video(demo_video_path, video_metadata)
    
    # Simulate analysis stages
    print("\n📊 Simulating analysis stages...")
    
    # Stage 1: Video Loading
    logger.start_stage("video_loading", "Loading video frames")
    time.sleep(0.5)  # Simulate processing time
    log_memory_usage(2.5, "video_loading")
    logger.end_stage("video_loading", {"frames_loaded": 7200})
    
    # Stage 2: Object Tracking
    logger.start_stage("object_tracking", "Tracking players and ball")
    time.sleep(1.0)  # Simulate processing time
    log_memory_usage(4.2, "object_tracking")
    
    # Simulate some detection events
    log_detection_event("pass", 1250, 0.85, {"team": 1, "player_id": 7})
    log_detection_event("tackle", 1890, 0.72, {"team": 2, "player_id": 23})
    log_detection_event("goal", 3456, 0.91, {"team": 1, "player_id": 10, "goal_type": "header"})
    
    logger.end_stage("object_tracking", {"total_detections": 1247})
    
    # Stage 3: Goal Detection
    logger.start_stage("goal_detection", "Analyzing goal events")
    time.sleep(0.3)
    
    # More goal events
    log_detection_event("goal", 5234, 0.88, {"team": 2, "player_id": 9, "goal_type": "penalty"})
    log_detection_event("goal", 6789, 0.93, {"team": 1, "player_id": 11, "goal_type": "free_kick"})
    
    logger.end_stage("goal_detection", {"goals_detected": 3})
    
    # Stage 4: Scoreboard Detection
    logger.start_stage("scoreboard_detection", "Detecting scoreboard information")
    time.sleep(0.4)
    
    # Simulate scoreboard detections
    log_scoreboard_event(3500, "1-0", 0.87, True)
    log_scoreboard_event(5300, "1-1", 0.82, True)
    log_scoreboard_event(6800, "2-1", 0.91, True)
    
    logger.end_stage("scoreboard_detection", {"scoreboard_detections": 15})
    
    # Stage 5: Statistics Generation
    logger.start_stage("statistics_generation", "Generating team and player statistics")
    time.sleep(0.6)
    
    # Log performance metrics
    log_performance_metric("processing_speed_fps", 24.3)
    log_performance_metric("total_passes_detected", 234)
    log_performance_metric("total_tackles_detected", 45)
    log_performance_metric("detection_accuracy", 0.87)
    
    logger.end_stage("statistics_generation")
    
    # Log team statistics
    team_stats = {
        "team_1": {
            "goals": 2,
            "passes": 145,
            "tackles": 23,
            "possession_percentage": 58.3
        },
        "team_2": {
            "goals": 1,
            "passes": 89,
            "tackles": 22,
            "possession_percentage": 41.7
        }
    }
    logger.log_team_statistics(team_stats)
    
    # Log output files
    output_files = {
        "output_video": "data/output_videos/demo_match_output.avi",
        "team_statistics_csv": "data/output/demo_match_team_stats.csv",
        "player_statistics_csv": "data/output/demo_match_player_stats.csv",
        "goal_detection_csv": "data/output/demo_match_goal_detected.csv"
    }
    logger.log_output_files(output_files)
    
    # Final memory usage
    log_memory_usage(3.8, "completion")
    
    # Finalize analysis
    final_results = {
        "processing_mode": "demo",
        "total_frames_processed": 7200,
        "team_1_goals": 2,
        "team_2_goals": 1,
        "total_players_scored": 3,
        "average_goal_confidence": 0.91,
        "detection_accuracy": 0.87,
        "processing_time_seconds": 2.8
    }
    
    logger.finalize_analysis(final_results)
    
    print(f"\n✅ Demo completed! Check the log file: {logger.log_file_path}")
    print(f"📋 Summary file: {logger.log_file_path.replace('.log', '_summary.json')}")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)
    
    # Run the demo
    demo_logging_system()
