"""
Cross Detector

Main cross detection system that integrates field zone analysis, trajectory analysis,
and cross classification to detect crosses in football video analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .cross_event import CrossEvent, CrossStatistics, CrossType, CrossOrigin, CrossTarget
from .field_zone_analyzer import FieldZoneAnalyzer
from .cross_trajectory_analyzer import CrossTrajectoryAnalyzer


class CrossDetector:
    """
    Main cross detection system that analyzes ball movements to identify crosses.
    """
    
    def __init__(self, video_width: int = 1920, video_height: int = 1080, frame_rate: float = 24.0):
        """
        Initialize the cross detector.
        
        Args:
            video_width: Width of the video frame
            video_height: Height of the video frame
            frame_rate: Video frame rate
        """
        self.video_width = video_width
        self.video_height = video_height
        self.frame_rate = frame_rate
        
        # Initialize sub-components
        self.field_zone_analyzer = FieldZoneAnalyzer(video_width, video_height)
        self.trajectory_analyzer = CrossTrajectoryAnalyzer(frame_rate)
        
        # Cross detection state
        self.detected_crosses = []
        self.current_trajectory = []
        self.last_player_id = -1
        self.last_team_id = None
        self.possession_start_frame = -1
        
        # Statistics tracking
        self.team_statistics = {1: CrossStatistics(), 2: CrossStatistics()}
        self.player_statistics = defaultdict(CrossStatistics)
        
        # Configuration
        self.config = {
            "min_possession_frames": 5,  # Minimum frames to consider stable possession
            "max_trajectory_gap": 3,     # Maximum frames gap in trajectory
            "min_cross_confidence": 0.6, # Minimum confidence for cross detection
            "enable_validation": True,   # Enable additional validation
        }
        
        # Field keypoints detector integration
        self.field_keypoints_detector = None
    
    def set_field_keypoints_detector(self, detector):
        """
        Set the field keypoints detector for enhanced zone detection.
        
        Args:
            detector: FieldKeypointsDetector instance
        """
        self.field_keypoints_detector = detector
        self.field_zone_analyzer.set_field_keypoints_detector(detector)
    
    def update_keypoints(self, frame):
        """Update field keypoints for current frame."""
        if self.field_keypoints_detector:
            self.field_keypoints_detector.detect_keypoints(frame)
            self.field_zone_analyzer.update_field_zones()
    
    def detect_cross(self, ball_position: Tuple[int, int], player_id: int, team_id: int, 
                    frame_num: int, ball_confidence: float = 0.8, 
                    ball_source: str = "detection") -> Optional[CrossEvent]:
        """
        Detect crosses based on ball position and player possession.
        
        Args:
            ball_position: Current ball position (x, y)
            player_id: ID of player possessing the ball
            team_id: Team ID of the player
            frame_num: Current frame number
            ball_confidence: Confidence of ball detection
            ball_source: Source of ball detection
            
        Returns:
            CrossEvent if cross is detected, None otherwise
        """
        # Handle no possession
        if player_id == -1 or team_id is None:
            self._handle_no_possession(frame_num)
            return None
        
        # Update trajectory
        ball_info = {
            "position": ball_position,
            "frame_num": frame_num,
            "confidence": ball_confidence,
            "source": ball_source,
            "player_id": player_id,
            "team_id": team_id
        }
        
        # Check for possession change
        if player_id != self.last_player_id:
            cross_event = self._handle_possession_change(player_id, team_id, frame_num, ball_info)
            if cross_event:
                return cross_event
        else:
            # Continue current possession
            self._continue_possession(ball_info)
        
        return None
    
    def _handle_possession_change(self, new_player_id: int, new_team_id: int, 
                                 frame_num: int, ball_info: Dict) -> Optional[CrossEvent]:
        """Handle possession change and check for cross completion."""
        cross_event = None
        
        # Check if previous possession resulted in a cross
        if (self.last_player_id != -1 and 
            len(self.current_trajectory) >= self.config["min_possession_frames"]):
            cross_event = self._analyze_trajectory_for_cross()
        
        # Start new possession tracking
        self._start_new_possession(new_player_id, new_team_id, frame_num, ball_info)
        
        return cross_event
    
    def _handle_no_possession(self, frame_num: int):
        """Handle frames where no player has possession."""
        # Check if we should end current trajectory
        if (self.current_trajectory and 
            frame_num - self.current_trajectory[-1]["frame_num"] > self.config["max_trajectory_gap"]):
            # Analyze trajectory before clearing
            if len(self.current_trajectory) >= self.config["min_possession_frames"]:
                cross_event = self._analyze_trajectory_for_cross()
                if cross_event:
                    self.detected_crosses.append(cross_event)
            
            self._clear_current_trajectory()
    
    def _start_new_possession(self, player_id: int, team_id: int, frame_num: int, ball_info: Dict):
        """Start tracking a new possession."""
        self.last_player_id = player_id
        self.last_team_id = team_id
        self.possession_start_frame = frame_num
        self.current_trajectory = [ball_info]
    
    def _continue_possession(self, ball_info: Dict):
        """Continue tracking current possession."""
        self.current_trajectory.append(ball_info)
        
        # Limit trajectory length for performance
        if len(self.current_trajectory) > 50:
            self.current_trajectory = self.current_trajectory[-30:]
    
    def _clear_current_trajectory(self):
        """Clear current trajectory tracking."""
        self.current_trajectory = []
        self.last_player_id = -1
        self.last_team_id = None
        self.possession_start_frame = -1
    
    def _analyze_trajectory_for_cross(self) -> Optional[CrossEvent]:
        """Analyze current trajectory to determine if it's a cross."""
        if len(self.current_trajectory) < self.config["min_possession_frames"]:
            return None
        
        # Get trajectory analysis
        trajectory_analysis = self.trajectory_analyzer.analyze_trajectory_for_cross(self.current_trajectory)
        
        if not trajectory_analysis or not trajectory_analysis.get("is_cross", False):
            return None
        
        # Get origin and target positions
        origin_position = self.current_trajectory[0]["position"]
        target_position = self.current_trajectory[-1]["position"]
        
        # Analyze field zones
        origin_zone = self.field_zone_analyzer.get_cross_origin_zone(origin_position)
        target_zone = self.field_zone_analyzer.get_cross_target_zone(target_position)
        
        # Validate cross
        if not self._validate_cross(origin_zone, target_zone, origin_position, target_position, trajectory_analysis):
            return None
        
        # Create cross event
        cross_event = self._create_cross_event(
            origin_position, target_position, origin_zone, target_zone, trajectory_analysis
        )
        
        # Update statistics
        self._update_statistics(cross_event)
        
        return cross_event
    
    def _validate_cross(self, origin_zone: Optional[CrossOrigin], target_zone: Optional[CrossTarget],
                       origin_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                       trajectory_analysis: Dict) -> bool:
        """Validate if the detected movement is actually a cross."""
        if not self.config["enable_validation"]:
            return True
        
        # Must have valid origin and target zones
        if not origin_zone or not target_zone:
            return False
        
        # Check direction validity
        if not self.field_zone_analyzer.is_cross_direction_valid(origin_pos, target_pos):
            return False
        
        # Check minimum distance
        distance = self.field_zone_analyzer.calculate_cross_distance(origin_pos, target_pos)
        if distance < 100:  # Minimum cross distance
            return False
        
        # Check trajectory quality
        if trajectory_analysis["cross_score"] < self.config["min_cross_confidence"]:
            return False
        
        return True
    
    def _create_cross_event(self, origin_pos: Tuple[int, int], target_pos: Tuple[int, int],
                           origin_zone: CrossOrigin, target_zone: CrossTarget,
                           trajectory_analysis: Dict) -> CrossEvent:
        """Create a CrossEvent from analysis results."""
        # Extract trajectory positions
        ball_trajectory = [pos["position"] for pos in self.current_trajectory]
        
        # Calculate metrics
        distance = self.field_zone_analyzer.calculate_cross_distance(origin_pos, target_pos)
        
        # Determine success (simplified - could be enhanced with receiver detection)
        is_successful = target_zone in [CrossTarget.PENALTY_BOX, CrossTarget.SIX_YARD_BOX, 
                                       CrossTarget.NEAR_POST, CrossTarget.FAR_POST]
        
        return CrossEvent(
            frame_num=self.current_trajectory[-1]["frame_num"],
            player_id=self.last_player_id,
            team_id=self.last_team_id,
            origin_position=origin_pos,
            target_position=target_pos,
            ball_trajectory=ball_trajectory,
            cross_type=trajectory_analysis["cross_type"],
            origin_zone=origin_zone,
            target_zone=target_zone,
            distance=distance,
            max_height=trajectory_analysis["height_analysis"]["estimated_max_height"],
            arc_curvature=trajectory_analysis["arc_analysis"]["curvature"],
            avg_velocity=trajectory_analysis["velocity_analysis"]["avg_velocity"],
            peak_velocity=trajectory_analysis["velocity_analysis"]["max_velocity"],
            confidence_score=trajectory_analysis["cross_score"],
            trajectory_quality=trajectory_analysis["trajectory_quality"],
            accuracy_score=0.8 if is_successful else 0.4,  # Simplified accuracy
            start_frame=self.current_trajectory[0]["frame_num"],
            end_frame=self.current_trajectory[-1]["frame_num"],
            duration_frames=trajectory_analysis["duration_frames"],
            is_successful=is_successful,
            field_context=self.field_zone_analyzer.get_field_context(origin_pos)
        )
    
    def _update_statistics(self, cross_event: CrossEvent):
        """Update team and player statistics with the cross event."""
        # Update team statistics
        team_stats = self.team_statistics[cross_event.team_id]
        team_stats.crosses_attempted += 1
        
        if cross_event.is_successful:
            team_stats.crosses_successful += 1
        else:
            team_stats.crosses_failed += 1
        
        # Update type-specific counts
        if cross_event.is_wing_cross:
            team_stats.wing_crosses += 1
        if cross_event.is_byline_cross:
            team_stats.byline_crosses += 1
        if cross_event.is_high_cross:
            team_stats.high_crosses += 1
        elif cross_event.cross_type == CrossType.LOW_CROSS:
            team_stats.low_crosses += 1
        elif cross_event.cross_type == CrossType.CUTBACK:
            team_stats.cutbacks += 1
        
        # Update target area counts
        if cross_event.target_zone == CrossTarget.PENALTY_BOX:
            team_stats.penalty_box_crosses += 1
        elif cross_event.target_zone == CrossTarget.SIX_YARD_BOX:
            team_stats.six_yard_crosses += 1
        elif cross_event.target_zone == CrossTarget.NEAR_POST:
            team_stats.near_post_crosses += 1
        elif cross_event.target_zone == CrossTarget.FAR_POST:
            team_stats.far_post_crosses += 1
        
        # Update averages
        team_stats.avg_cross_distance = (
            (team_stats.avg_cross_distance * (team_stats.crosses_attempted - 1) + cross_event.distance) /
            team_stats.crosses_attempted
        )
        team_stats.avg_cross_height = (
            (team_stats.avg_cross_height * (team_stats.crosses_attempted - 1) + cross_event.max_height) /
            team_stats.crosses_attempted
        )
        team_stats.avg_confidence_score = (
            (team_stats.avg_confidence_score * (team_stats.crosses_attempted - 1) + cross_event.confidence_score) /
            team_stats.crosses_attempted
        )
        
        # Calculate accuracy
        team_stats.calculate_accuracy()
        
        # Update player statistics (similar logic)
        player_stats = self.player_statistics[cross_event.player_id]
        player_stats.crosses_attempted += 1
        if cross_event.is_successful:
            player_stats.crosses_successful += 1
        else:
            player_stats.crosses_failed += 1
        player_stats.calculate_accuracy()
    
    def get_cross_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cross statistics."""
        return {
            "team_statistics": {team_id: stats.to_dict() for team_id, stats in self.team_statistics.items()},
            "player_statistics": {player_id: stats.to_dict() for player_id, stats in self.player_statistics.items()},
            "total_crosses": len(self.detected_crosses),
            "crosses_by_type": self._get_crosses_by_type(),
            "crosses_by_zone": self._get_crosses_by_zone()
        }
    
    def _get_crosses_by_type(self) -> Dict[str, int]:
        """Get cross counts by type."""
        type_counts = defaultdict(int)
        for cross in self.detected_crosses:
            type_counts[cross.cross_type.value] += 1
        return dict(type_counts)
    
    def _get_crosses_by_zone(self) -> Dict[str, int]:
        """Get cross counts by origin zone."""
        zone_counts = defaultdict(int)
        for cross in self.detected_crosses:
            zone_counts[cross.origin_zone.value] += 1
        return dict(zone_counts)
