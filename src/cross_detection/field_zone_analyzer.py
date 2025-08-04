"""
Field Zone Analyzer for Cross Detection

Analyzes field positions to determine if they are suitable for cross origins or targets.
Extends the existing field keypoints detection to identify specific zones.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .cross_event import CrossOrigin, CrossTarget


class FieldZoneAnalyzer:
    """
    Analyzes field positions to determine zones relevant for cross detection.
    """
    
    def __init__(self, video_width: int = 1920, video_height: int = 1080):
        """
        Initialize the field zone analyzer.
        
        Args:
            video_width: Width of the video frame
            video_height: Height of the video frame
        """
        self.video_width = video_width
        self.video_height = video_height
        
        # Field zone boundaries (as percentages of field width/height)
        self.zone_config = {
            # Wing zones (outer 25% of field width)
            "left_wing_x_max": 0.25,
            "right_wing_x_min": 0.75,
            
            # Byline zones (final 15% of field length)
            "left_byline_x_max": 0.15,
            "right_byline_x_min": 0.85,
            
            # Half-space zones (between wing and center)
            "left_half_space_x_min": 0.25,
            "left_half_space_x_max": 0.40,
            "right_half_space_x_min": 0.60,
            "right_half_space_x_max": 0.75,
            
            # Central zones
            "central_x_min": 0.40,
            "central_x_max": 0.60,
            
            # Penalty box zones (approximate)
            "penalty_box_width": 0.25,  # 25% of field width
            "penalty_box_depth": 0.20,  # 20% of field length
            
            # Six-yard box zones
            "six_yard_width": 0.12,   # 12% of field width
            "six_yard_depth": 0.08,   # 8% of field length
        }
        
        # Cache for field keypoints detector integration
        self.field_keypoints_detector = None
        self.penalty_areas = {"left": None, "right": None}
        self.goal_areas = {"left": None, "right": None}
        
    def set_field_keypoints_detector(self, detector):
        """
        Set the field keypoints detector for enhanced zone detection.
        
        Args:
            detector: FieldKeypointsDetector instance
        """
        self.field_keypoints_detector = detector
        
    def update_field_zones(self):
        """Update field zones based on keypoints detector if available."""
        if self.field_keypoints_detector:
            self.penalty_areas = self.field_keypoints_detector.get_penalty_areas()
            self.goal_areas = self.field_keypoints_detector.get_goal_areas()
    
    def get_cross_origin_zone(self, position: Tuple[int, int]) -> Optional[CrossOrigin]:
        """
        Determine the cross origin zone for a given position.
        
        Args:
            position: (x, y) position on the field
            
        Returns:
            CrossOrigin enum value or None if not a valid cross origin
        """
        x, y = position
        x_ratio = x / self.video_width
        
        # Check for byline crosses first (most specific)
        if x_ratio <= self.zone_config["left_byline_x_max"]:
            return CrossOrigin.LEFT_BYLINE
        elif x_ratio >= self.zone_config["right_byline_x_min"]:
            return CrossOrigin.RIGHT_BYLINE
        
        # Check for wing crosses
        elif x_ratio <= self.zone_config["left_wing_x_max"]:
            return CrossOrigin.LEFT_WING
        elif x_ratio >= self.zone_config["right_wing_x_min"]:
            return CrossOrigin.RIGHT_WING
        
        # Check for half-space crosses
        elif (self.zone_config["left_half_space_x_min"] <= x_ratio <= 
              self.zone_config["left_half_space_x_max"]):
            return CrossOrigin.LEFT_HALF_SPACE
        elif (self.zone_config["right_half_space_x_min"] <= x_ratio <= 
              self.zone_config["right_half_space_x_max"]):
            return CrossOrigin.RIGHT_HALF_SPACE
        
        # Not a valid cross origin zone
        return None
    
    def get_cross_target_zone(self, position: Tuple[int, int]) -> Optional[CrossTarget]:
        """
        Determine the cross target zone for a given position.
        
        Args:
            position: (x, y) position on the field
            
        Returns:
            CrossTarget enum value or None if not a valid cross target
        """
        x, y = position
        
        # Use keypoints detector if available for more accurate detection
        if self.field_keypoints_detector:
            # Check if in goal area (six-yard box)
            goal_area = self.field_keypoints_detector.is_ball_in_goal_area(position)
            if goal_area:
                return self._get_goal_area_target(position, goal_area)
            
            # Check if in penalty area
            penalty_area = self.field_keypoints_detector.is_ball_in_penalty_area(position)
            if penalty_area:
                return self._get_penalty_area_target(position, penalty_area)
        
        # Fallback to ratio-based detection
        return self._get_fallback_target_zone(position)
    
    def _get_goal_area_target(self, position: Tuple[int, int], goal_side: str) -> CrossTarget:
        """Get specific target within goal area."""
        x, y = position
        
        if goal_side == "left":
            goal_area = self.goal_areas.get("left")
        else:
            goal_area = self.goal_areas.get("right")
        
        if goal_area:
            # Determine near/far post based on position within goal area
            goal_center_y = (goal_area["y_min"] + goal_area["y_max"]) / 2
            if y < goal_center_y:
                return CrossTarget.NEAR_POST
            else:
                return CrossTarget.FAR_POST
        
        return CrossTarget.SIX_YARD_BOX
    
    def _get_penalty_area_target(self, position: Tuple[int, int], penalty_side: str) -> CrossTarget:
        """Get specific target within penalty area."""
        x, y = position
        
        if penalty_side == "left":
            penalty_area = self.penalty_areas.get("left")
        else:
            penalty_area = self.penalty_areas.get("right")
        
        if penalty_area:
            # Determine position within penalty area
            area_width = penalty_area["x_max"] - penalty_area["x_min"]
            area_height = penalty_area["y_max"] - penalty_area["y_min"]
            
            # Check if near the goal line (central box area)
            goal_line_threshold = 0.3  # 30% from goal line
            if penalty_side == "left":
                distance_from_goal = (x - penalty_area["x_min"]) / area_width
            else:
                distance_from_goal = (penalty_area["x_max"] - x) / area_width
            
            if distance_from_goal <= goal_line_threshold:
                return CrossTarget.CENTRAL_BOX
            else:
                return CrossTarget.EDGE_OF_BOX
        
        return CrossTarget.PENALTY_BOX
    
    def _get_fallback_target_zone(self, position: Tuple[int, int]) -> Optional[CrossTarget]:
        """Fallback target zone detection using ratios."""
        x, y = position
        x_ratio = x / self.video_width
        
        # Approximate penalty box areas
        left_penalty_x_max = self.zone_config["penalty_box_depth"]
        right_penalty_x_min = 1.0 - self.zone_config["penalty_box_depth"]
        
        # Check if in penalty box area
        if x_ratio <= left_penalty_x_max or x_ratio >= right_penalty_x_min:
            # Further classify within penalty box
            six_yard_depth = self.zone_config["six_yard_depth"]
            
            if x_ratio <= six_yard_depth or x_ratio >= (1.0 - six_yard_depth):
                return CrossTarget.SIX_YARD_BOX
            else:
                return CrossTarget.PENALTY_BOX
        
        # Check if near penalty box (edge area)
        edge_threshold = 0.05  # 5% buffer around penalty box
        if (x_ratio <= left_penalty_x_max + edge_threshold or 
            x_ratio >= right_penalty_x_min - edge_threshold):
            return CrossTarget.EDGE_OF_BOX
        
        return None
    
    def is_valid_cross_origin(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is a valid cross origin.
        
        Args:
            position: (x, y) position on the field
            
        Returns:
            True if position is suitable for crossing
        """
        return self.get_cross_origin_zone(position) is not None
    
    def is_valid_cross_target(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is a valid cross target.
        
        Args:
            position: (x, y) position on the field
            
        Returns:
            True if position is suitable as cross target
        """
        return self.get_cross_target_zone(position) is not None
    
    def calculate_cross_distance(self, origin: Tuple[int, int], target: Tuple[int, int]) -> float:
        """
        Calculate the distance of a potential cross.
        
        Args:
            origin: Starting position of the cross
            target: Target position of the cross
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((target[0] - origin[0])**2 + (target[1] - origin[1])**2)
    
    def is_cross_direction_valid(self, origin: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """
        Check if the direction from origin to target is valid for a cross.
        
        Args:
            origin: Starting position
            target: Target position
            
        Returns:
            True if direction is valid for a cross
        """
        origin_zone = self.get_cross_origin_zone(origin)
        target_zone = self.get_cross_target_zone(target)
        
        if not origin_zone or not target_zone:
            return False
        
        # Check if cross is going toward the goal
        origin_x, origin_y = origin
        target_x, target_y = target
        
        # Left side crosses should go toward right (positive x direction)
        if origin_zone in [CrossOrigin.LEFT_WING, CrossOrigin.LEFT_BYLINE, CrossOrigin.LEFT_HALF_SPACE]:
            return target_x > origin_x
        
        # Right side crosses should go toward left (negative x direction)
        elif origin_zone in [CrossOrigin.RIGHT_WING, CrossOrigin.RIGHT_BYLINE, CrossOrigin.RIGHT_HALF_SPACE]:
            return target_x < origin_x
        
        return False
    
    def get_field_context(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Get comprehensive field context for a position.
        
        Args:
            position: (x, y) position on the field
            
        Returns:
            Dictionary with field context information
        """
        return {
            "cross_origin_zone": self.get_cross_origin_zone(position),
            "cross_target_zone": self.get_cross_target_zone(position),
            "is_valid_cross_origin": self.is_valid_cross_origin(position),
            "is_valid_cross_target": self.is_valid_cross_target(position),
            "x_ratio": position[0] / self.video_width,
            "y_ratio": position[1] / self.video_height
        }
