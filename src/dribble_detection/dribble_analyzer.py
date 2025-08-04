#!/usr/bin/env python3
"""
Dribble Analyzer

This module provides comprehensive analysis of dribble events and statistics
for football video analysis. It integrates with the main processing pipeline
and provides CSV export functionality.
"""

import csv
import os
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .dribble_detector import DribbleDetector
from .dribble_event import DribbleEvent, DribbleStatistics


class DribbleAnalyzer:
    """
    Comprehensive dribble analysis system that integrates with the main pipeline.
    
    This analyzer provides:
    1. Integration with existing tracker and detection systems
    2. Frame-by-frame dribble detection coordination
    3. Statistics aggregation and analysis
    4. CSV export functionality
    """
    
    def __init__(self, 
                 min_dribble_distance: float = 30.0,
                 min_dribble_duration: int = 10,
                 confidence_threshold: float = 0.4):
        """
        Initialize the dribble analyzer.
        
        Args:
            min_dribble_distance: Minimum distance for a valid dribble (pixels)
            min_dribble_duration: Minimum duration for a valid dribble (frames)
            confidence_threshold: Minimum confidence for dribble detection
        """
        self.detector = DribbleDetector(
            min_dribble_distance=min_dribble_distance,
            min_dribble_duration=min_dribble_duration,
            confidence_threshold=confidence_threshold
        )
        
        # Analysis results
        self.dribble_events = []
        self.frame_analysis = []
        
        # Integration with existing systems
        self.pass_counter = None
        self.tackle_counter = None
        
    def analyze_frame(self, 
                     frame_num: int,
                     players: Dict[int, Dict],
                     ball_position: Optional[tuple],
                     ball_possessor: int,
                     ball_team: int) -> Optional[DribbleEvent]:
        """
        Analyze a single frame for dribble detection.
        
        Args:
            frame_num: Current frame number
            players: Dictionary of player data {player_id: {bbox, team, ...}}
            ball_position: Current ball position (x, y)
            ball_possessor: ID of player currently possessing the ball
            ball_team: Team of player currently possessing the ball
            
        Returns:
            DribbleEvent if a dribble is detected, None otherwise
        """
        # Detect dribble in current frame
        dribble_event = self.detector.detect_dribble(
            frame_num, players, ball_position, ball_possessor, ball_team
        )
        
        # Store frame analysis data
        frame_data = {
            'frame_num': frame_num,
            'ball_possessor': ball_possessor,
            'ball_team': ball_team,
            'dribble_detected': dribble_event is not None,
            'dribble_confidence': dribble_event.confidence if dribble_event else 0.0
        }
        self.frame_analysis.append(frame_data)
        
        # Store dribble event if detected
        if dribble_event:
            self.dribble_events.append(dribble_event)
        
        return dribble_event
    
    def analyze_complete_match(self, tracks: Dict, player_assignments: List[int]) -> Dict:
        """
        Analyze complete match data for dribbles.
        
        Args:
            tracks: Complete tracking data including ball and player tracks
            player_assignments: Player assignments per frame
            
        Returns:
            Dict: Comprehensive dribble analysis results
        """
        print("🏃 Analyzing dribbles from complete match data...")
        
        total_frames = len(tracks.get("players", []))
        dribbles_detected = 0
        
        for frame_num in range(total_frames):
            # Skip if frame data is missing
            if frame_num >= len(tracks.get("players", [])) or frame_num >= len(tracks.get("ball", [])):
                continue
            
            # Get frame data
            players = tracks["players"][frame_num]
            ball_data = tracks["ball"][frame_num].get(1, {})
            ball_position = None
            
            if ball_data and "bbox" in ball_data:
                bbox = ball_data["bbox"]
                if len(bbox) == 4:
                    ball_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Get ball possession
            ball_possessor = player_assignments[frame_num] if frame_num < len(player_assignments) else -1
            ball_team = None
            
            if ball_possessor != -1 and ball_possessor in players:
                ball_team = players[ball_possessor].get("team", None)
            
            # Analyze frame
            dribble_event = self.analyze_frame(
                frame_num, players, ball_position, ball_possessor, ball_team
            )
            
            if dribble_event:
                dribbles_detected += 1
                if dribbles_detected % 5 == 0:  # Progress update every 5 dribbles
                    print(f"  📊 Detected {dribbles_detected} dribbles so far...")
        
        print(f"✅ Dribble analysis complete: {dribbles_detected} dribbles detected")
        
        # Generate comprehensive results
        return self.generate_analysis_results()
    
    def generate_analysis_results(self) -> Dict:
        """Generate comprehensive analysis results."""
        team_stats = self.detector.get_team_statistics()
        player_stats = self.detector.get_player_statistics()
        completed_dribbles = self.detector.get_completed_dribbles()
        
        # Calculate additional metrics
        total_dribbles = sum(stats.total_dribbles for stats in team_stats.values())
        avg_confidence = sum(event.confidence for event in completed_dribbles) / len(completed_dribbles) if completed_dribbles else 0.0
        
        # Team-level analysis
        team_analysis = {}
        for team_id, stats in team_stats.items():
            team_analysis[team_id] = {
                'total_dribbles': stats.total_dribbles,
                'successful_dribbles': stats.successful_dribbles,
                'success_rate': stats.success_rate,
                'avg_confidence': stats.avg_confidence,
                'avg_duration': stats.avg_duration,
                'avg_distance': stats.avg_distance,
                'avg_speed': stats.avg_speed,
                'events_count': stats.events_count
            }
        
        # Player-level analysis
        player_analysis = {}
        for player_id, stats in player_stats.items():
            player_analysis[player_id] = {
                'total_dribbles': stats.total_dribbles,
                'successful_dribbles': stats.successful_dribbles,
                'success_rate': stats.success_rate,
                'avg_confidence': stats.avg_confidence,
                'events_count': stats.events_count
            }
        
        return {
            'team_statistics': team_analysis,
            'player_statistics': player_analysis,
            'dribble_events': completed_dribbles,
            'total_dribbles': total_dribbles,
            'avg_confidence': avg_confidence,
            'frame_analysis': self.frame_analysis
        }
    
    def export_to_csv(self, output_dir: str = "data/output", video_name: str = "match") -> Dict[str, str]:
        """
        Export dribble analysis results to CSV files.
        
        Args:
            output_dir: Output directory for CSV files
            video_name: Video name for file naming
            
        Returns:
            Dict mapping export type to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        # Export team dribble statistics
        team_stats_path = os.path.join(output_dir, f"{video_name}_dribble_team_stats.csv")
        self._export_team_statistics(team_stats_path)
        exported_files["team_dribble_stats"] = team_stats_path
        
        # Export player dribble statistics
        player_stats_path = os.path.join(output_dir, f"{video_name}_dribble_player_stats.csv")
        self._export_player_statistics(player_stats_path)
        exported_files["player_dribble_stats"] = player_stats_path
        
        # Export dribble events
        events_path = os.path.join(output_dir, f"{video_name}_dribble_events.csv")
        self._export_dribble_events(events_path)
        exported_files["dribble_events"] = events_path
        
        print(f"📊 Dribble analysis exported to {output_dir}")
        return exported_files
    
    def _export_team_statistics(self, file_path: str):
        """Export team dribble statistics to CSV."""
        team_stats = self.detector.get_team_statistics()
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [
                'team', 'dribbles_detected', 'successful_dribbles', 'dribble_success_rate',
                'avg_dribble_confidence', 'dribble_events_count', 'avg_dribble_duration',
                'avg_dribble_distance', 'avg_dribble_speed'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for team_id, stats in team_stats.items():
                writer.writerow({
                    'team': team_id,
                    'dribbles_detected': stats.total_dribbles,
                    'successful_dribbles': stats.successful_dribbles,
                    'dribble_success_rate': round(stats.success_rate, 3),
                    'avg_dribble_confidence': round(stats.avg_confidence, 3),
                    'dribble_events_count': stats.events_count,
                    'avg_dribble_duration': round(stats.avg_duration, 2),
                    'avg_dribble_distance': round(stats.avg_distance, 2),
                    'avg_dribble_speed': round(stats.avg_speed, 2)
                })
    
    def _export_player_statistics(self, file_path: str):
        """Export player dribble statistics to CSV."""
        player_stats = self.detector.get_player_statistics()
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [
                'player_id', 'dribbles_detected', 'successful_dribbles', 'dribble_success_rate',
                'avg_dribble_confidence', 'dribble_events_count'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for player_id, stats in player_stats.items():
                writer.writerow({
                    'player_id': player_id,
                    'dribbles_detected': stats.total_dribbles,
                    'successful_dribbles': stats.successful_dribbles,
                    'dribble_success_rate': round(stats.success_rate, 3),
                    'avg_dribble_confidence': round(stats.avg_confidence, 3),
                    'dribble_events_count': stats.events_count
                })
    
    def _export_dribble_events(self, file_path: str):
        """Export individual dribble events to CSV."""
        dribble_events = self.detector.get_completed_dribbles()
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame_num', 'team', 'player_id', 'confidence', 'start_frame', 'end_frame',
                'duration_frames', 'distance_covered', 'direction_changes', 'avg_speed',
                'opponent_player_id', 'opponent_distance', 'ball_control_consistency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in dribble_events:
                writer.writerow({
                    'frame_num': event.frame_num,
                    'team': event.team,
                    'player_id': event.player_id,
                    'confidence': round(event.confidence, 3),
                    'start_frame': event.start_frame,
                    'end_frame': event.end_frame,
                    'duration_frames': event.duration_frames,
                    'distance_covered': round(event.distance_covered, 2),
                    'direction_changes': event.direction_changes,
                    'avg_speed': round(event.avg_speed, 2),
                    'opponent_player_id': event.opponent_player_id,
                    'opponent_distance': round(event.opponent_distance, 2) if event.opponent_distance else None,
                    'ball_control_consistency': round(event.ball_control_consistency, 3)
                })
    
    def get_team_dribble_counts(self) -> Dict[int, int]:
        """Get dribble counts by team for integration with existing statistics."""
        team_stats = self.detector.get_team_statistics()
        return {team_id: stats.total_dribbles for team_id, stats in team_stats.items()}
    
    def get_team_dribble_confidence(self) -> Dict[int, float]:
        """Get average dribble confidence by team for integration with existing statistics."""
        team_stats = self.detector.get_team_statistics()
        return {team_id: stats.avg_confidence for team_id, stats in team_stats.items()}
    
    def get_team_dribble_events_count(self) -> Dict[int, int]:
        """Get dribble events count by team for integration with existing statistics."""
        team_stats = self.detector.get_team_statistics()
        return {team_id: stats.events_count for team_id, stats in team_stats.items()}
