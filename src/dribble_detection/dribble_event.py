#!/usr/bin/env python3
"""
Dribble Event Data Structure

This module defines the data structures for dribble events in football video analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class DribbleEvent:
    """
    Comprehensive dribble event with detailed tracking information.
    
    A dribble is defined as a player successfully moving past an opponent
    while maintaining ball control.
    """
    
    frame_num: int
    team: int
    player_id: int
    ball_position: Tuple[int, int]
    player_position: Tuple[int, int]
    confidence: float
    
    # Dribble-specific metrics
    start_frame: int
    end_frame: int
    duration_frames: int
    distance_covered: float
    direction_changes: int
    speed_changes: int
    
    # Opponent interaction
    opponent_player_id: Optional[int] = None
    opponent_team: Optional[int] = None
    opponent_distance: Optional[float] = None
    
    # Movement analysis
    avg_speed: float = 0.0
    max_speed: float = 0.0
    acceleration_changes: int = 0
    trajectory_smoothness: float = 0.0
    
    # Ball control quality
    ball_control_consistency: float = 0.0
    ball_touches: int = 0
    ball_distance_variance: float = 0.0
    
    # Detection method and validation
    detection_method: str = "comprehensive"
    validation_score: float = 0.0
    
    # Additional metadata
    field_zone: Optional[str] = None  # e.g., "penalty_area", "midfield", "wing"
    dribble_type: Optional[str] = None  # e.g., "speed", "skill", "direction_change"


@dataclass
class DribbleSequence:
    """
    A sequence of related dribble movements that form a complete dribble action.
    """
    
    sequence_id: str
    events: List[DribbleEvent]
    start_frame: int
    end_frame: int
    total_duration: int
    
    # Sequence-level metrics
    total_distance: float
    avg_confidence: float
    success_rate: float
    
    # Player and team info
    player_id: int
    team: int
    
    # Outcome
    successful: bool
    outcome: str  # e.g., "pass", "shot", "tackle", "out_of_bounds"


@dataclass
class DribbleStatistics:
    """
    Comprehensive dribble statistics for teams and players.
    """
    
    # Basic counts
    total_dribbles: int = 0
    successful_dribbles: int = 0
    failed_dribbles: int = 0
    
    # Success metrics
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    
    # Performance metrics
    avg_duration: float = 0.0
    avg_distance: float = 0.0
    avg_speed: float = 0.0
    
    # Quality metrics
    avg_ball_control_quality: float = 0.0
    avg_trajectory_smoothness: float = 0.0
    
    # Tactical metrics
    dribbles_in_penalty_area: int = 0
    dribbles_in_midfield: int = 0
    dribbles_on_wings: int = 0
    
    # Event counts
    events_count: int = 0
    sequences_count: int = 0


class DribbleEventValidator:
    """
    Validates dribble events to ensure quality and consistency.
    """
    
    def __init__(self, min_confidence: float = 0.3, min_duration: int = 5):
        self.min_confidence = min_confidence
        self.min_duration = min_duration
    
    def validate_event(self, event: DribbleEvent) -> bool:
        """
        Validate a single dribble event.
        
        Args:
            event: DribbleEvent to validate
            
        Returns:
            bool: True if event is valid
        """
        # Basic validation checks
        if event.confidence < self.min_confidence:
            return False
            
        if event.duration_frames < self.min_duration:
            return False
            
        if event.distance_covered <= 0:
            return False
            
        # Ensure reasonable values
        if event.avg_speed < 0 or event.avg_speed > 50:  # Reasonable speed limits
            return False
            
        return True
    
    def validate_sequence(self, sequence: DribbleSequence) -> bool:
        """
        Validate a dribble sequence.
        
        Args:
            sequence: DribbleSequence to validate
            
        Returns:
            bool: True if sequence is valid
        """
        if not sequence.events:
            return False
            
        if sequence.total_duration < self.min_duration:
            return False
            
        # Validate all events in sequence
        for event in sequence.events:
            if not self.validate_event(event):
                return False
                
        return True
