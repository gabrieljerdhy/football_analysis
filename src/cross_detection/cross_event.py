"""
Cross Event Data Structure

Defines the data structure for representing cross events in football analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class CrossType(Enum):
    """Types of crosses based on characteristics."""
    LOW_CROSS = "low_cross"
    HIGH_CROSS = "high_cross"
    CUTBACK = "cutback"
    CORNER_CROSS = "corner_cross"
    DEEP_CROSS = "deep_cross"
    PULLBACK = "pullback"


class CrossOrigin(Enum):
    """Origin zones for crosses."""
    LEFT_WING = "left_wing"
    RIGHT_WING = "right_wing"
    LEFT_BYLINE = "left_byline"
    RIGHT_BYLINE = "right_byline"
    LEFT_HALF_SPACE = "left_half_space"
    RIGHT_HALF_SPACE = "right_half_space"


class CrossTarget(Enum):
    """Target areas for crosses."""
    PENALTY_BOX = "penalty_box"
    SIX_YARD_BOX = "six_yard_box"
    NEAR_POST = "near_post"
    FAR_POST = "far_post"
    CENTRAL_BOX = "central_box"
    EDGE_OF_BOX = "edge_of_box"


@dataclass
class CrossEvent:
    """
    Represents a cross event with comprehensive metadata.
    """
    # Basic event information
    frame_num: int
    player_id: int
    team_id: int
    
    # Spatial information
    origin_position: Tuple[int, int]
    target_position: Optional[Tuple[int, int]]
    ball_trajectory: list  # List of ball positions during cross
    
    # Cross characteristics
    cross_type: CrossType
    origin_zone: CrossOrigin
    target_zone: CrossTarget
    
    # Trajectory metrics
    distance: float
    max_height: float
    arc_curvature: float
    avg_velocity: float
    peak_velocity: float
    
    # Quality metrics
    confidence_score: float
    trajectory_quality: float
    accuracy_score: float  # How close to intended target
    
    # Timing information
    start_frame: int
    end_frame: int
    duration_frames: int
    
    # Success metrics
    is_successful: bool = False
    receiver_player_id: Optional[int] = None
    receiver_team_id: Optional[int] = None
    
    # Additional metadata
    field_context: Optional[Dict[str, Any]] = None
    detection_method: str = "trajectory_analysis"
    validation_flags: Optional[Dict[str, bool]] = None
    
    def __post_init__(self):
        """Post-initialization validation and calculations."""
        # Calculate duration if not provided
        if self.duration_frames == 0:
            self.duration_frames = self.end_frame - self.start_frame
        
        # Initialize validation flags if not provided
        if self.validation_flags is None:
            self.validation_flags = {
                "valid_origin": True,
                "valid_trajectory": True,
                "valid_target": True,
                "sufficient_distance": True,
                "appropriate_height": True
            }
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds (assuming 24 fps)."""
        return self.duration_frames / 24.0
    
    @property
    def is_wing_cross(self) -> bool:
        """Check if this is a wing cross."""
        return self.origin_zone in [CrossOrigin.LEFT_WING, CrossOrigin.RIGHT_WING]
    
    @property
    def is_byline_cross(self) -> bool:
        """Check if this is a byline cross."""
        return self.origin_zone in [CrossOrigin.LEFT_BYLINE, CrossOrigin.RIGHT_BYLINE]
    
    @property
    def is_high_cross(self) -> bool:
        """Check if this is a high cross based on trajectory."""
        return self.cross_type in [CrossType.HIGH_CROSS, CrossType.DEEP_CROSS]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cross event to dictionary for export."""
        return {
            "frame_num": self.frame_num,
            "player_id": self.player_id,
            "team_id": self.team_id,
            "origin_x": self.origin_position[0],
            "origin_y": self.origin_position[1],
            "target_x": self.target_position[0] if self.target_position else None,
            "target_y": self.target_position[1] if self.target_position else None,
            "cross_type": self.cross_type.value,
            "origin_zone": self.origin_zone.value,
            "target_zone": self.target_zone.value,
            "distance": self.distance,
            "max_height": self.max_height,
            "arc_curvature": self.arc_curvature,
            "avg_velocity": self.avg_velocity,
            "peak_velocity": self.peak_velocity,
            "confidence_score": self.confidence_score,
            "trajectory_quality": self.trajectory_quality,
            "accuracy_score": self.accuracy_score,
            "duration_frames": self.duration_frames,
            "duration_seconds": self.duration_seconds,
            "is_successful": self.is_successful,
            "receiver_player_id": self.receiver_player_id,
            "receiver_team_id": self.receiver_team_id,
            "detection_method": self.detection_method,
            "is_wing_cross": self.is_wing_cross,
            "is_byline_cross": self.is_byline_cross,
            "is_high_cross": self.is_high_cross
        }


@dataclass
class CrossStatistics:
    """
    Aggregated cross statistics for teams and players.
    """
    # Basic counts
    crosses_attempted: int = 0
    crosses_successful: int = 0
    crosses_failed: int = 0
    
    # Accuracy metrics
    cross_accuracy_percentage: float = 0.0
    avg_cross_distance: float = 0.0
    avg_cross_height: float = 0.0
    
    # Type breakdown
    wing_crosses: int = 0
    byline_crosses: int = 0
    high_crosses: int = 0
    low_crosses: int = 0
    cutbacks: int = 0
    
    # Target area breakdown
    penalty_box_crosses: int = 0
    six_yard_crosses: int = 0
    near_post_crosses: int = 0
    far_post_crosses: int = 0
    
    # Quality metrics
    avg_confidence_score: float = 0.0
    avg_trajectory_quality: float = 0.0
    avg_accuracy_score: float = 0.0
    
    def calculate_accuracy(self):
        """Calculate cross accuracy percentage."""
        if self.crosses_attempted > 0:
            self.cross_accuracy_percentage = (self.crosses_successful / self.crosses_attempted) * 100
        else:
            self.cross_accuracy_percentage = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for export."""
        return {
            "crosses_attempted": self.crosses_attempted,
            "crosses_successful": self.crosses_successful,
            "crosses_failed": self.crosses_failed,
            "cross_accuracy_percentage": self.cross_accuracy_percentage,
            "avg_cross_distance": self.avg_cross_distance,
            "avg_cross_height": self.avg_cross_height,
            "wing_crosses": self.wing_crosses,
            "byline_crosses": self.byline_crosses,
            "high_crosses": self.high_crosses,
            "low_crosses": self.low_crosses,
            "cutbacks": self.cutbacks,
            "penalty_box_crosses": self.penalty_box_crosses,
            "six_yard_crosses": self.six_yard_crosses,
            "near_post_crosses": self.near_post_crosses,
            "far_post_crosses": self.far_post_crosses,
            "avg_confidence_score": self.avg_confidence_score,
            "avg_trajectory_quality": self.avg_trajectory_quality,
            "avg_accuracy_score": self.avg_accuracy_score
        }
