#!/usr/bin/env python3
"""
Goal Detection CSV Generator

This module generates the two required CSV files for goal detection:
1. goal_detected.csv - Frame-by-frame goal detection results
2. goal_detected_scoreboard.csv - Summarized goal events for scoreboard updates

The system integrates with existing ball detection, player detection, and field keypoint detection
to provide comprehensive goal detection with temporal validation.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GoalDetectionFrame:
    """Frame-by-frame goal detection data."""
    
    frame_num: int
    ball_position: Optional[Tuple[int, int]]
    ball_confidence: float
    ball_in_goal_area: bool
    goal_side: Optional[str]  # 'left', 'right', or None
    player_id: Optional[int]
    player_team: Optional[int]
    player_ball_distance: Optional[float]
    kick_detected: bool
    goal_detected: bool
    goal_confidence: float
    detection_method: str
    temporal_validation: bool
    sequence_id: Optional[int]  # Groups frames that are part of the same goal sequence


@dataclass
class GoalEvent:
    """Summarized goal event for scoreboard updates."""
    
    goal_id: int
    frame_start: int
    frame_end: int
    frame_goal: int  # Frame where goal was actually scored
    team: int
    player_id: Optional[int]
    goal_side: str
    ball_final_position: Tuple[int, int]
    confidence: float
    validation_score: float
    kick_frame: Optional[int]
    sequence_frames: List[int]
    detection_method: str


class GoalCSVGenerator:
    """
    Generates CSV files for goal detection results.
    
    This class handles the generation of two CSV files:
    1. goal_detected.csv - Contains frame-by-frame analysis
    2. goal_detected_scoreboard.csv - Contains summarized goal events
    """
    
    def __init__(self, output_dir: str = "data/output"):
        """
        Initialize the CSV generator.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_data: List[GoalDetectionFrame] = []
        self.goal_events: List[GoalEvent] = []
        
        self.logger = logging.getLogger(__name__)
        
    def add_frame_data(
        self,
        frame_num: int,
        ball_position: Optional[Tuple[int, int]],
        ball_confidence: float,
        ball_in_goal_area: bool,
        goal_side: Optional[str],
        player_id: Optional[int],
        player_team: Optional[int],
        player_ball_distance: Optional[float],
        kick_detected: bool,
        goal_detected: bool,
        goal_confidence: float,
        detection_method: str,
        temporal_validation: bool,
        sequence_id: Optional[int] = None,
    ):
        """Add frame-by-frame goal detection data."""
        frame_data = GoalDetectionFrame(
            frame_num=frame_num,
            ball_position=ball_position,
            ball_confidence=ball_confidence,
            ball_in_goal_area=ball_in_goal_area,
            goal_side=goal_side,
            player_id=player_id,
            player_team=player_team,
            player_ball_distance=player_ball_distance,
            kick_detected=kick_detected,
            goal_detected=goal_detected,
            goal_confidence=goal_confidence,
            detection_method=detection_method,
            temporal_validation=temporal_validation,
            sequence_id=sequence_id,
        )
        self.frame_data.append(frame_data)
        
    def add_goal_event(
        self,
        goal_id: int,
        frame_start: int,
        frame_end: int,
        frame_goal: int,
        team: int,
        player_id: Optional[int],
        goal_side: str,
        ball_final_position: Tuple[int, int],
        confidence: float,
        validation_score: float,
        kick_frame: Optional[int],
        sequence_frames: List[int],
        detection_method: str,
    ):
        """Add a goal event for scoreboard updates."""
        goal_event = GoalEvent(
            goal_id=goal_id,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_goal=frame_goal,
            team=team,
            player_id=player_id,
            goal_side=goal_side,
            ball_final_position=ball_final_position,
            confidence=confidence,
            validation_score=validation_score,
            kick_frame=kick_frame,
            sequence_frames=sequence_frames,
            detection_method=detection_method,
        )
        self.goal_events.append(goal_event)
        
    def generate_frame_csv(self, video_name: str) -> str:
        """
        Generate goal_detected.csv with frame-by-frame results.
        
        Args:
            video_name: Name of the video being processed
            
        Returns:
            Path to the generated CSV file
        """
        csv_path = self.output_dir / f"{video_name}_goal_detected.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame_num',
                'ball_x',
                'ball_y', 
                'ball_confidence',
                'ball_in_goal_area',
                'goal_side',
                'player_id',
                'player_team',
                'player_ball_distance',
                'kick_detected',
                'goal_detected',
                'goal_confidence',
                'detection_method',
                'temporal_validation',
                'sequence_id'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame in self.frame_data:
                ball_x, ball_y = frame.ball_position if frame.ball_position else (None, None)
                
                writer.writerow({
                    'frame_num': frame.frame_num,
                    'ball_x': ball_x,
                    'ball_y': ball_y,
                    'ball_confidence': round(frame.ball_confidence, 3),
                    'ball_in_goal_area': frame.ball_in_goal_area,
                    'goal_side': frame.goal_side or '',
                    'player_id': frame.player_id or '',
                    'player_team': frame.player_team or '',
                    'player_ball_distance': round(frame.player_ball_distance, 2) if frame.player_ball_distance else '',
                    'kick_detected': frame.kick_detected,
                    'goal_detected': frame.goal_detected,
                    'goal_confidence': round(frame.goal_confidence, 3),
                    'detection_method': frame.detection_method,
                    'temporal_validation': frame.temporal_validation,
                    'sequence_id': frame.sequence_id or ''
                })
                
        self.logger.info(f"Generated frame-by-frame CSV: {csv_path}")
        return str(csv_path)
        
    def generate_scoreboard_csv(self, video_name: str) -> str:
        """
        Generate goal_detected_scoreboard.csv with summarized goal events.
        
        Args:
            video_name: Name of the video being processed
            
        Returns:
            Path to the generated CSV file
        """
        csv_path = self.output_dir / f"{video_name}_goal_detected_scoreboard.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'goal_id',
                'frame_start',
                'frame_end', 
                'frame_goal',
                'team',
                'player_id',
                'goal_side',
                'ball_final_x',
                'ball_final_y',
                'confidence',
                'validation_score',
                'kick_frame',
                'sequence_length',
                'detection_method',
                'timestamp_seconds'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for goal in self.goal_events:
                ball_x, ball_y = goal.ball_final_position
                
                writer.writerow({
                    'goal_id': goal.goal_id,
                    'frame_start': goal.frame_start,
                    'frame_end': goal.frame_end,
                    'frame_goal': goal.frame_goal,
                    'team': goal.team,
                    'player_id': goal.player_id or '',
                    'goal_side': goal.goal_side,
                    'ball_final_x': ball_x,
                    'ball_final_y': ball_y,
                    'confidence': round(goal.confidence, 3),
                    'validation_score': round(goal.validation_score, 3),
                    'kick_frame': goal.kick_frame or '',
                    'sequence_length': len(goal.sequence_frames),
                    'detection_method': goal.detection_method,
                    'timestamp_seconds': round(goal.frame_goal / 30.0, 2)  # Assuming 30 FPS
                })
                
        self.logger.info(f"Generated scoreboard CSV: {csv_path}")
        return str(csv_path)
        
    def generate_both_csvs(self, video_name: str) -> Tuple[str, str]:
        """
        Generate both required CSV files.
        
        Args:
            video_name: Name of the video being processed
            
        Returns:
            Tuple of (frame_csv_path, scoreboard_csv_path)
        """
        frame_csv = self.generate_frame_csv(video_name)
        scoreboard_csv = self.generate_scoreboard_csv(video_name)
        
        return frame_csv, scoreboard_csv
        
    def clear_data(self):
        """Clear all stored data for processing a new video."""
        self.frame_data.clear()
        self.goal_events.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the goal detection results."""
        total_frames = len(self.frame_data)
        goal_frames = len([f for f in self.frame_data if f.goal_detected])
        total_goals = len(self.goal_events)
        
        team_goals = {}
        for goal in self.goal_events:
            team_goals[goal.team] = team_goals.get(goal.team, 0) + 1
            
        avg_confidence = np.mean([g.confidence for g in self.goal_events]) if self.goal_events else 0.0
        
        return {
            'total_frames_processed': total_frames,
            'frames_with_goal_detection': goal_frames,
            'total_goals_detected': total_goals,
            'team_goals': team_goals,
            'average_goal_confidence': avg_confidence,
            'detection_methods': list(set(g.detection_method for g in self.goal_events))
        }
