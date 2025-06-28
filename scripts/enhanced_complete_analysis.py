#!/usr/bin/env python3
"""
Enhanced complete analysis pipeline using the new unified statistics manager
and improved ball tracking capabilities.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from goal_detection import FieldKeypointsDetector, GoalDetector
from pass_counter.unified_statistics_manager import UnifiedStatisticsManager, AnalysisConfiguration
from pass_counter.pass_counter import PassCounter  # Fallback for compatibility
from pass_counter.tackle_counter import TackleCounter  # Fallback for compatibility
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import VideoFrameIterator, get_video_info, monitor_memory_usage, save_video
from utils.goal_utils import (
    calculate_final_goal_stats,
    export_consolidated_goal_statistics,
    load_manual_goals,
)
from view_transformer import ViewTransformer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Football Analysis Pipeline")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output-dir", default="data/output", 
                       help="Output directory for results")
    parser.add_argument("--use-enhanced-stats", action="store_true", default=True,
                       help="Use enhanced statistics manager (default: True)")
    parser.add_argument("--enable-trajectory-analysis", action="store_true", default=True,
                       help="Enable ball trajectory analysis")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                       help="Minimum confidence threshold for events")
    parser.add_argument("--frame-rate", type=float, default=24.0,
                       help="Video frame rate")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for processing")
    parser.add_argument("--export-video", action="store_true", default=False,
                       help="Export annotated video")
    parser.add_argument("--stub-path", help="Path to tracking stub file")
    
    return parser.parse_args()


def setup_enhanced_analysis(args) -> UnifiedStatisticsManager:
    """Setup the enhanced analysis configuration."""
    config = AnalysisConfiguration(
        frame_rate=args.frame_rate,
        enable_enhanced_ball_tracking=True,
        enable_trajectory_analysis=args.enable_trajectory_analysis,
        enable_confidence_validation=True,
        min_pass_confidence=args.min_confidence,
        min_tackle_confidence=args.min_confidence,
        min_possession_frames=5,
        export_detailed_events=True,
        export_player_stats=True,
        export_team_stats=True,
        export_quality_metrics=True
    )
    
    return UnifiedStatisticsManager(config)


def load_or_generate_tracks(input_video: str, stub_path: Optional[str], 
                          batch_size: int) -> Dict:
    """Load existing tracks or generate new ones."""
    if stub_path and os.path.exists(stub_path):
        print(f"📂 Loading tracks from stub: {stub_path}")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        return tracks
    
    print("🔍 Generating new tracking data...")
    
    # Initialize tracker with enhanced ball detection
    tracker = Tracker(
        model_path="data/models/best_detect.pt",
        ball_model_path="data/models/best_ball.pt",
        enable_enhanced_ball_detection=True,
        enable_jersey_detection=True
    )
    
    # Get video frames
    video_info = get_video_info(input_video)
    frame_iterator = VideoFrameIterator(input_video)
    frames = list(frame_iterator)
    
    print(f"📹 Processing {len(frames)} frames...")
    
    # Generate tracks with enhanced ball detection
    tracks = tracker.get_object_tracks_memory_efficient(
        frames, 
        read_from_stub=False,
        stub_path=stub_path
    )
    
    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)
    
    # Interpolate ball positions with enhanced method
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    return tracks


def assign_teams_and_process_tracking(tracks: Dict, input_video: str) -> Dict:
    """Assign teams and process additional tracking data."""
    print("👥 Assigning teams...")
    
    # Initialize team assigner
    team_assigner = TeamAssigner()
    
    # Get first frame for team assignment
    frame_iterator = VideoFrameIterator(input_video)
    first_frame = next(frame_iterator)
    
    # Assign teams
    team_assigner.assign_team_color(first_frame, tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                first_frame, track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )
    
    return tracks


def process_camera_movement_and_transformation(tracks: Dict, input_video: str) -> Dict:
    """Process camera movement and apply view transformation."""
    print("📷 Processing camera movement...")
    
    # This is a simplified version - in the full implementation,
    # you would use the CameraMovementEstimator and ViewTransformer
    # For now, we'll add placeholder transformed positions
    
    for object_name in ["players", "ball"]:
        for frame_num, frame_tracks in enumerate(tracks[object_name]):
            for track_id, track_data in frame_tracks.items():
                if "position" in track_data:
                    # Placeholder transformation - in reality, this would use
                    # the actual camera movement and view transformation
                    track_data["position_transformed"] = track_data["position"]
    
    return tracks


def calculate_speed_and_distance(tracks: Dict) -> Dict:
    """Calculate speed and distance for players."""
    print("🏃 Calculating speed and distance...")
    
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    
    return tracks


def run_enhanced_analysis(args):
    """Run the enhanced analysis pipeline."""
    print("🚀 Starting Enhanced Football Analysis Pipeline")
    print("="*60)
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize enhanced statistics manager
    if args.use_enhanced_stats:
        print("✨ Using Enhanced Statistics Manager")
        stats_manager = setup_enhanced_analysis(args)
    else:
        print("📊 Using Legacy Statistics")
        stats_manager = None
    
    try:
        # Step 1: Load or generate tracking data
        print("\n🔍 STEP 1: Object Tracking")
        tracks = load_or_generate_tracks(
            args.input_video, args.stub_path, args.batch_size
        )
        
        # Step 2: Team assignment and basic processing
        print("\n👥 STEP 2: Team Assignment")
        tracks = assign_teams_and_process_tracking(tracks, args.input_video)
        
        # Step 3: Camera movement and view transformation
        print("\n📷 STEP 3: Camera Movement & Transformation")
        tracks = process_camera_movement_and_transformation(tracks, args.input_video)
        
        # Step 4: Speed and distance calculation
        print("\n🏃 STEP 4: Speed & Distance Calculation")
        tracks = calculate_speed_and_distance(tracks)
        
        # Step 5: Enhanced statistics analysis
        if args.use_enhanced_stats and stats_manager:
            print("\n📊 STEP 5: Enhanced Statistics Analysis")
            analysis_results = stats_manager.analyze_match(
                tracks, args.output_dir, save_results=True
            )
            
            # Export summary report
            summary_path = os.path.join(args.output_dir, "enhanced_analysis_summary.txt")
            stats_manager.export_summary_report(summary_path)
            
        else:
            print("\n📊 STEP 5: Legacy Statistics Analysis")
            run_legacy_statistics_analysis(tracks, args.output_dir)
        
        # Step 6: Goal detection (if needed)
        print("\n⚽ STEP 6: Goal Detection")
        process_goal_detection(tracks, args.input_video, args.output_dir)
        
        # Step 7: Export video (if requested)
        if args.export_video:
            print("\n🎬 STEP 7: Video Export")
            export_annotated_video(tracks, args.input_video, args.output_dir)
        
        print("\n✅ Enhanced Analysis Pipeline Complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print final summary
        if args.use_enhanced_stats and stats_manager:
            summary = stats_manager.get_analysis_summary()
            if summary:
                print(f"\n📈 Final Summary:")
                print(f"   • Total Events: {summary['total_events']}")
                print(f"   • Quality Score: {summary['quality_score']:.2f}")
                print(f"   • Processing Time: {summary['performance_metrics']['processing_time']:.1f}s")
        
    except Exception as e:
        print(f"\n❌ Error in analysis pipeline: {str(e)}")
        raise


def run_legacy_statistics_analysis(tracks: Dict, output_dir: str):
    """Run legacy statistics analysis for compatibility."""
    # Initialize legacy counters
    pass_counter = PassCounter()
    tackle_counter = TackleCounter()
    player_assigner = PlayerBallAssigner()
    
    team_ball_control = []
    
    # Process each frame
    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num >= len(tracks["ball"]):
            continue
            
        ball_data = tracks["ball"][frame_num].get(1, {})
        ball_bbox = ball_data.get("bbox", [])
        
        if ball_bbox and len(ball_bbox) == 4:
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
        else:
            assigned_player = -1
        
        if assigned_player != -1:
            current_team = tracks["players"][frame_num][assigned_player].get("team", 1)
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(current_team)
            
            # Count passes
            pass_counter.count_passes(assigned_player, current_team, frame_num)
            
            # Detect tackles
            tackle_counter.detect_tackles_and_interceptions(
                player_track, assigned_player, current_team, frame_num
            )
        else:
            # No player has ball
            pass_counter.count_passes(-1, None, frame_num)
            tackle_counter.detect_tackles_and_interceptions(
                player_track, -1, None, frame_num
            )
    
    # Export legacy statistics
    pass_counter.export_team_stats_to_csv(os.path.join(output_dir, "team_stats.csv"))
    pass_counter.export_player_stats_to_csv(os.path.join(output_dir, "player_stats.csv"))
    tackle_counter.export_team_stats_to_csv(os.path.join(output_dir, "tackle_team_stats.csv"))
    tackle_counter.export_player_stats_to_csv(os.path.join(output_dir, "tackle_player_stats.csv"))


def process_goal_detection(tracks: Dict, input_video: str, output_dir: str):
    """Process goal detection if needed."""
    # This is a placeholder for goal detection
    # In the full implementation, this would use the GoalDetector
    print("⚽ Goal detection processing (placeholder)")


def export_annotated_video(tracks: Dict, input_video: str, output_dir: str):
    """Export annotated video with enhanced visualizations."""
    print("🎬 Exporting annotated video...")
    
    # This is a placeholder for video export
    # In the full implementation, this would create an annotated video
    # with enhanced visualizations showing the improved tracking
    output_video_path = os.path.join(output_dir, "enhanced_analysis_output.mp4")
    print(f"📹 Video would be exported to: {output_video_path}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input
    if not os.path.exists(args.input_video):
        print(f"❌ Input video not found: {args.input_video}")
        return 1
    
    try:
        run_enhanced_analysis(args)
        return 0
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
