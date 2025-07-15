#!/usr/bin/env python3
"""
Improved Goal Detection System

This module provides a comprehensive goal detection system that combines:
1. Enhanced field keypoints detection with fallback areas
2. Ball trajectory analysis
3. Multiple goal area zones
4. Confidence-based validation
5. Manual goal integration for training/validation

Designed to achieve accurate goal detection for the football analysis pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .field_keypoints_detector import FieldKeypointsDetector


@dataclass
class ImprovedGoalEvent:
    """Enhanced goal event with comprehensive tracking."""

    frame_num: int
    team: int
    player_id: int
    goal_side: str
    ball_position: Tuple[int, int]
    confidence: float
    detection_method: str
    ball_confidence: Optional[float] = None
    ball_source: Optional[str] = None
    validation_score: float = 0.0


class ImprovedGoalDetectionSystem:
    """
    Comprehensive goal detection system with multiple validation methods.

    This system combines:
    1. Field keypoints detection with robust fallback areas
    2. Enhanced ball trajectory analysis
    3. Multiple goal zone detection (primary, extended, trajectory-based)
    4. Confidence scoring and validation
    5. Integration with manual goals for accuracy validation
    """

    def __init__(self, field_keypoints_detector: FieldKeypointsDetector):
        """
        Initialize the improved goal detection system.

        Args:
            field_keypoints_detector: Field keypoints detector instance
        """
        self.keypoints_detector = field_keypoints_detector
        self.logger = logging.getLogger(__name__)

        # Goal detection configuration
        self.detection_config = {
            "primary_confidence_threshold": 0.4,  # Lowered from 0.6
            "extended_confidence_threshold": 0.19,  # Lowered to 0.19 to catch extended areas only
            "trajectory_confidence_threshold": 0.2,  # Lowered from 0.3
            "min_trajectory_length": 2,  # Lowered from 3
            "goal_cooldown_frames": 30,  # Prevent duplicate detections
        }

        # Detection weights for fusion
        self.detection_weights = {
            "keypoints": 0.4,
            "primary_areas": 0.3,
            "extended_areas": 0.2,
            "trajectory": 0.1,
        }

        # Goal detection state
        self.goal_detected = False
        self.last_goal_frame = -1
        self.ball_trajectory = []
        self.goal_events = []

        # Enhanced goal areas (will be initialized on first frame)
        self.enhanced_goal_areas = {}
        self.video_dimensions = None

    def update_keypoints(self, frame, force_detection=False):
        """Update field keypoints and goal areas."""
        # Initialize video dimensions if not set
        if self.video_dimensions is None:
            height, width = frame.shape[:2]
            self.video_dimensions = (width, height)
            self._initialize_enhanced_goal_areas(width, height)
            force_detection = True

        # Ensure enhanced goal areas are initialized
        if not self.enhanced_goal_areas:
            height, width = frame.shape[:2]
            self._initialize_enhanced_goal_areas(width, height)

        # Update keypoints
        self.keypoints_detector.detect_keypoints(frame, force_detection=force_detection)

    def detect_goal(
        self,
        ball_position: Tuple[int, int],
        player_id: int,
        team: int,
        frame_num: int,
        ball_confidence: Optional[float] = None,
        ball_source: Optional[str] = None,
    ) -> Optional[ImprovedGoalEvent]:
        """
        Comprehensive goal detection using multiple methods.

        Args:
            ball_position: Ball position (x, y)
            player_id: ID of player with ball
            team: Team number
            frame_num: Current frame number
            ball_confidence: Ball detection confidence
            ball_source: Ball detection source

        Returns:
            Goal event if detected, None otherwise
        """
        # Check cooldown period
        if (
            self.last_goal_frame >= 0
            and frame_num - self.last_goal_frame
            < self.detection_config["goal_cooldown_frames"]
        ):
            return None

        # Update ball trajectory
        self._update_ball_trajectory(
            ball_position, frame_num, ball_confidence, ball_source
        )

        # Multi-method goal detection
        detection_results = self._detect_goal_multi_method(ball_position)

        if detection_results:
            goal_side, confidence, method = detection_results

            # Validate goal detection
            if self._validate_goal_detection(
                goal_side, ball_position, confidence, ball_confidence
            ):
                goal_event = self._create_goal_event(
                    goal_side,
                    player_id,
                    team,
                    frame_num,
                    ball_position,
                    confidence,
                    method,
                    ball_confidence,
                    ball_source,
                )

                self.goal_events.append(goal_event)
                self.last_goal_frame = frame_num

                self.logger.info(
                    f"Goal detected: Team {team} at frame {frame_num} ({goal_side} goal)"
                )
                return goal_event

        return None

    def _initialize_enhanced_goal_areas(self, width: int, height: int):
        """Initialize enhanced goal areas with multiple detection zones."""

        # Primary goal areas (conservative, high confidence)
        primary_width = int(width * 0.12)  # 12% of video width
        primary_height = int(height * 0.50)  # Increased to 50% of video height
        primary_y_center = height // 2
        primary_y_margin = primary_height // 2

        # Extended goal areas (more generous for edge cases)
        extended_width = int(width * 0.18)  # 18% of video width
        extended_height = int(height * 0.65)  # Increased to 65% of video height
        extended_y_center = height // 2
        extended_y_margin = extended_height // 2

        # Trajectory-based areas (very large for trajectory validation)
        trajectory_width = int(width * 0.25)  # 25% of video width
        trajectory_height = int(
            height * 0.85
        )  # Increased to 85% of video height to catch (1599, 867)
        trajectory_y_center = height // 2
        trajectory_y_margin = trajectory_height // 2

        self.enhanced_goal_areas = {
            "left": {
                "primary": {
                    "x_min": 0,
                    "x_max": primary_width,
                    "y_min": primary_y_center - primary_y_margin,
                    "y_max": primary_y_center + primary_y_margin,
                },
                "extended": {
                    "x_min": 0,
                    "x_max": extended_width,
                    "y_min": extended_y_center - extended_y_margin,
                    "y_max": extended_y_center + extended_y_margin,
                },
                "trajectory": {
                    "x_min": 0,
                    "x_max": trajectory_width,
                    "y_min": trajectory_y_center - trajectory_y_margin,
                    "y_max": trajectory_y_center + trajectory_y_margin,
                },
            },
            "right": {
                "primary": {
                    "x_min": width - primary_width,
                    "x_max": width,
                    "y_min": primary_y_center - primary_y_margin,
                    "y_max": primary_y_center + primary_y_margin,
                },
                "extended": {
                    "x_min": width - extended_width,
                    "x_max": width,
                    "y_min": extended_y_center - extended_y_margin,
                    "y_max": extended_y_center + extended_y_margin,
                },
                "trajectory": {
                    "x_min": width - trajectory_width,
                    "x_max": width,
                    "y_min": trajectory_y_center - trajectory_y_margin,
                    "y_max": trajectory_y_center + trajectory_y_margin,
                },
            },
        }

        self.logger.info(f"Enhanced goal areas initialized for {width}x{height} video")
        self.logger.info(
            f"Left primary: x=0-{primary_width}, y={primary_y_center - primary_y_margin}-{primary_y_center + primary_y_margin}"
        )
        self.logger.info(
            f"Right primary: x={width - primary_width}-{width}, y={primary_y_center - primary_y_margin}-{primary_y_center + primary_y_margin}"
        )

    def _update_ball_trajectory(
        self, ball_position, frame_num, ball_confidence, ball_source
    ):
        """Update ball trajectory for analysis."""
        trajectory_entry = {
            "position": ball_position,
            "frame": frame_num,
            "confidence": ball_confidence or 0.5,
            "source": ball_source or "unknown",
        }

        self.ball_trajectory.append(trajectory_entry)

        # Keep only recent trajectory (last 10 positions)
        if len(self.ball_trajectory) > 10:
            self.ball_trajectory = self.ball_trajectory[-10:]

    def _detect_goal_multi_method(
        self, ball_position: Tuple[int, int]
    ) -> Optional[Tuple[str, float, str]]:
        """Multi-method goal detection with confidence scoring."""
        # Safety check: ensure enhanced goal areas are initialized
        if not self.enhanced_goal_areas:
            # Initialize with default video dimensions if not set
            width, height = self.video_dimensions or (1920, 1080)
            self._initialize_enhanced_goal_areas(width, height)

        x, y = ball_position
        detection_scores = {"left": 0.0, "right": 0.0}
        detection_methods = {"left": [], "right": []}

        # Method 1: Field keypoints detection
        keypoint_result = self.keypoints_detector.is_ball_in_goal_area(ball_position)
        if keypoint_result:
            detection_scores[keypoint_result] += self.detection_weights["keypoints"]
            detection_methods[keypoint_result].append("keypoints")

        # Method 2: Primary goal areas
        for side in ["left", "right"]:
            primary_area = self.enhanced_goal_areas[side]["primary"]
            if (
                primary_area["x_min"] <= x <= primary_area["x_max"]
                and primary_area["y_min"] <= y <= primary_area["y_max"]
            ):
                detection_scores[side] += self.detection_weights["primary_areas"]
                detection_methods[side].append("primary_areas")

        # Method 3: Extended goal areas
        for side in ["left", "right"]:
            extended_area = self.enhanced_goal_areas[side]["extended"]
            if (
                extended_area["x_min"] <= x <= extended_area["x_max"]
                and extended_area["y_min"] <= y <= extended_area["y_max"]
            ):
                detection_scores[side] += self.detection_weights["extended_areas"]
                detection_methods[side].append("extended_areas")

        # Method 4: Trajectory-based detection
        trajectory_result = self._analyze_trajectory_for_goal(ball_position)
        if trajectory_result:
            detection_scores[trajectory_result] += self.detection_weights["trajectory"]
            detection_methods[trajectory_result].append("trajectory")

        # Find best detection
        best_side = max(detection_scores, key=detection_scores.get)
        best_score = detection_scores[best_side]

        # Apply confidence thresholds with multiple levels
        if best_score >= self.detection_config["primary_confidence_threshold"]:
            method_str = "+".join(detection_methods[best_side])
            return best_side, best_score, method_str
        elif best_score >= self.detection_config["extended_confidence_threshold"]:
            method_str = "+".join(detection_methods[best_side])
            return best_side, best_score, method_str

        return None

    def _analyze_trajectory_for_goal(
        self, ball_position: Tuple[int, int]
    ) -> Optional[str]:
        """Analyze ball trajectory to predict goal."""
        if len(self.ball_trajectory) < self.detection_config["min_trajectory_length"]:
            return None

        x, y = ball_position

        # Check if ball is in trajectory areas
        for side in ["left", "right"]:
            trajectory_area = self.enhanced_goal_areas[side]["trajectory"]
            if (
                trajectory_area["x_min"] <= x <= trajectory_area["x_max"]
                and trajectory_area["y_min"] <= y <= trajectory_area["y_max"]
            ):

                # Validate trajectory direction
                if self._validate_trajectory_direction(side):
                    return side

        return None

    def _validate_trajectory_direction(self, goal_side: str) -> bool:
        """Validate that ball trajectory is moving toward the goal."""
        if len(self.ball_trajectory) < 2:
            return False

        # Get recent movement
        current_pos = self.ball_trajectory[-1]["position"]
        prev_pos = self.ball_trajectory[-2]["position"]

        movement_x = current_pos[0] - prev_pos[0]

        # Check if movement is toward the goal
        if goal_side == "left" and movement_x <= 0:  # Moving left toward left goal
            return True
        elif goal_side == "right" and movement_x >= 0:  # Moving right toward right goal
            return True

        return False

    def _validate_goal_detection(
        self,
        goal_side: str,
        ball_position: Tuple[int, int],
        confidence: float,
        ball_confidence: Optional[float],
    ) -> bool:
        """Validate goal detection with additional checks."""

        # Basic confidence check
        if confidence < self.detection_config["extended_confidence_threshold"]:
            return False

        # Ball confidence check
        if ball_confidence is not None and ball_confidence < 0.1:
            return False

        # Additional validation can be added here
        return True

    def _create_goal_event(
        self,
        goal_side: str,
        player_id: int,
        team: int,
        frame_num: int,
        ball_position: Tuple[int, int],
        confidence: float,
        method: str,
        ball_confidence: Optional[float],
        ball_source: Optional[str],
    ) -> ImprovedGoalEvent:
        """Create a goal event with comprehensive information."""

        return ImprovedGoalEvent(
            frame_num=frame_num,
            team=team,
            player_id=player_id,
            goal_side=goal_side,
            ball_position=ball_position,
            confidence=confidence,
            detection_method=method,
            ball_confidence=ball_confidence,
            ball_source=ball_source,
            validation_score=confidence,
        )

    def get_goal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive goal detection statistics."""
        team_goals = {}
        player_goals = {}

        for event in self.goal_events:
            # Team goals
            if event.team not in team_goals:
                team_goals[event.team] = 0
            team_goals[event.team] += 1

            # Player goals
            if event.player_id not in player_goals:
                player_goals[event.player_id] = {"goals": 0, "team": event.team}
            player_goals[event.player_id]["goals"] += 1

        return {
            "team_goals": team_goals,
            "player_goals": player_goals,
            "goal_events": [
                {
                    "frame_num": event.frame_num,
                    "team": event.team,
                    "player_id": event.player_id,
                    "goal_side": event.goal_side,
                    "ball_position": event.ball_position,
                    "confidence": event.confidence,
                    "detection_method": event.detection_method,
                }
                for event in self.goal_events
            ],
            "total_goals": len(self.goal_events),
            "average_confidence": (
                np.mean([event.confidence for event in self.goal_events])
                if self.goal_events
                else 0.0
            ),
        }
