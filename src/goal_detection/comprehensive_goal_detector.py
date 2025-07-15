#!/usr/bin/env python3
"""
Comprehensive Goal Detection System

This module provides a complete goal detection system that integrates:
1. Ball detection using existing specialized models
2. Player detection and ball-player interaction analysis
3. Field keypoint detection for accurate goal area identification
4. Temporal sequence validation for goal confirmation
5. CSV output generation for frame-by-frame and scoreboard results

The system is designed to minimize false positives by requiring:
- Ball crossing into goal net area (spatial analysis)
- Player kick detection (not just any ball movement)
- Temporal validation of goal sequences
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .field_keypoints_detector import FieldKeypointsDetector
from .goal_csv_generator import GoalCSVGenerator


class ComprehensiveGoalDetector:
    """
    Comprehensive goal detection system with CSV output generation.

    This detector combines multiple validation methods:
    1. Spatial analysis for ball-in-goal detection
    2. Player-ball interaction analysis for kick validation
    3. Temporal sequence validation
    4. Integration with existing detection models
    5. CSV output generation
    """

    def __init__(
        self,
        field_keypoints_detector: FieldKeypointsDetector,
        csv_generator: GoalCSVGenerator,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the comprehensive goal detector.

        Args:
            field_keypoints_detector: Field keypoints detector instance
            csv_generator: CSV generator for output files
            config: Configuration parameters
        """
        self.keypoints_detector = field_keypoints_detector
        self.csv_generator = csv_generator

        # Default configuration
        default_config = {
            "goal_cooldown_frames": 90,  # 3 seconds at 30fps
            "kick_detection_distance": 50,  # pixels
            "kick_detection_frames": 5,  # frames to look back for kick
            "sequence_validation_frames": 15,  # frames for temporal validation
            "min_ball_confidence": 0.3,
            "min_goal_confidence": 0.5,
            "player_ball_max_distance": 100,  # pixels
            "ball_speed_threshold": 10,  # pixels per frame
            "goal_area_buffer": 20,  # pixels buffer around goal area
        }

        self.config = {**default_config, **(config or {})}

        # State tracking
        self.frame_count = 0
        self.last_goal_frame = -1
        self.goal_sequence_id = 0
        self.current_sequence_frames: List[int] = []
        self.ball_trajectory: List[Tuple[int, int, int]] = []  # (x, y, frame)
        self.player_positions: Dict[int, List[Tuple[int, int, int]]] = (
            {}
        )  # player_id -> [(x, y, frame)]

        # Goal tracking
        self.detected_goals: List[Dict[str, Any]] = []
        self.goal_events_count = 0

        self.logger = logging.getLogger(__name__)

    def process_frame(
        self,
        frame_num: int,
        ball_position: Optional[Tuple[int, int]],
        ball_confidence: float,
        player_detections: List[Dict[str, Any]],
        frame_data: Optional[Any] = None,
    ) -> bool:
        """
        Process a single frame for goal detection.

        Args:
            frame_num: Current frame number
            ball_position: Ball position (x, y) or None if not detected
            ball_confidence: Ball detection confidence
            player_detections: List of player detections with positions and IDs
            frame_data: Additional frame data for keypoint detection

        Returns:
            True if goal detected in this frame, False otherwise
        """
        self.frame_count = frame_num

        # Initialize frame analysis variables
        ball_in_goal_area = False
        goal_side = None
        player_id = None
        player_team = None
        player_ball_distance = None
        kick_detected = False
        goal_detected = False
        goal_confidence = 0.0
        detection_method = "none"
        temporal_validation = False
        sequence_id = None

        # Update ball trajectory
        if ball_position and ball_confidence >= self.config["min_ball_confidence"]:
            self._update_ball_trajectory(ball_position, frame_num)

            # Check if ball is in goal area
            ball_in_goal_area, goal_side = self._check_ball_in_goal_area(ball_position)

            if ball_in_goal_area:
                # Find closest player and check for kick
                closest_player = self._find_closest_player(
                    ball_position, player_detections
                )
                if closest_player:
                    player_id = closest_player["player_id"]
                    player_team = closest_player.get("team", None)
                    player_ball_distance = closest_player["distance"]

                    # Check for kick detection
                    kick_detected = self._detect_player_kick(
                        player_id, ball_position, frame_num, player_detections
                    )

                    if kick_detected:
                        # Perform temporal validation
                        temporal_validation = self._validate_goal_sequence(
                            ball_position, goal_side, frame_num
                        )

                        if temporal_validation:
                            # Check cooldown period
                            if (
                                frame_num - self.last_goal_frame
                                >= self.config["goal_cooldown_frames"]
                            ):
                                goal_detected = True
                                goal_confidence = self._calculate_goal_confidence(
                                    ball_confidence,
                                    player_ball_distance,
                                    kick_detected,
                                    temporal_validation,
                                )
                                detection_method = "comprehensive"
                                sequence_id = self._start_goal_sequence(frame_num)

                                # Register goal event
                                self._register_goal_event(
                                    frame_num,
                                    player_team,
                                    player_id,
                                    goal_side,
                                    ball_position,
                                    goal_confidence,
                                    detection_method,
                                )

                                self.last_goal_frame = frame_num
                                self.logger.info(
                                    f"Goal detected at frame {frame_num}: Team {player_team}, "
                                    f"Player {player_id}, Side {goal_side}, Confidence {goal_confidence:.3f}"
                                )

        # Update player positions
        self._update_player_positions(player_detections, frame_num)

        # Add frame data to CSV generator
        self.csv_generator.add_frame_data(
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

        return goal_detected

    def _update_ball_trajectory(self, ball_position: Tuple[int, int], frame_num: int):
        """Update ball trajectory for analysis."""
        self.ball_trajectory.append((ball_position[0], ball_position[1], frame_num))

        # Keep only recent trajectory points
        max_trajectory_length = self.config["sequence_validation_frames"] * 2
        if len(self.ball_trajectory) > max_trajectory_length:
            self.ball_trajectory = self.ball_trajectory[-max_trajectory_length:]

    def _check_ball_in_goal_area(
        self, ball_position: Tuple[int, int]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if ball is in goal area using field keypoints.

        Returns:
            Tuple of (is_in_goal_area, goal_side)
        """
        try:
            # Use existing keypoints detector
            goal_side = self.keypoints_detector.is_ball_in_goal_area(ball_position)
            return goal_side is not None, goal_side
        except Exception as e:
            # Fallback to simple position-based detection if keypoints fail
            self.logger.warning(f"Keypoints detection failed, using fallback: {e}")
            return self._fallback_goal_area_check(ball_position)

    def _fallback_goal_area_check(
        self, ball_position: Tuple[int, int]
    ) -> Tuple[bool, Optional[str]]:
        """
        Fallback goal area detection using simple position-based logic.

        This is used when field keypoints detection is not available.
        Assumes standard video dimensions and goal positions.
        """
        x, y = ball_position

        # Assume standard video dimensions (adjust as needed)
        # Left goal area: x < 100, center area for y
        # Right goal area: x > video_width - 100, center area for y

        # Simple fallback - detect goals at extreme left/right positions
        if x < 80:  # Left goal area
            return True, "left"
        elif x > 1200:  # Right goal area (assuming ~1280 width)
            return True, "right"
        else:
            return False, None

    def _find_closest_player(
        self, ball_position: Tuple[int, int], player_detections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the closest player to the ball."""
        if not player_detections:
            return None

        closest_player = None
        min_distance = float("inf")

        ball_x, ball_y = ball_position

        for player in player_detections:
            if "bbox" in player:
                # Calculate player center from bbox
                bbox = player["bbox"]
                player_x = (bbox[0] + bbox[2]) / 2
                player_y = (bbox[1] + bbox[3]) / 2
            elif "position" in player:
                player_x, player_y = player["position"]
            else:
                continue

            distance = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

            if (
                distance < min_distance
                and distance <= self.config["player_ball_max_distance"]
            ):
                min_distance = distance
                closest_player = {
                    **player,
                    "distance": distance,
                    "position": (player_x, player_y),
                }

        return closest_player

    def _detect_player_kick(
        self,
        player_id: int,
        ball_position: Tuple[int, int],
        frame_num: int,
        player_detections: List[Dict[str, Any]],
    ) -> bool:
        """
        Detect if a player kick caused the ball movement.

        This analyzes recent ball trajectory and player positions to determine
        if the ball movement was caused by a player kick rather than other factors.
        """
        if len(self.ball_trajectory) < 3:
            return False

        # Check if ball speed increased recently (indicating a kick)
        recent_trajectory = [
            t
            for t in self.ball_trajectory
            if frame_num - t[2] <= self.config["kick_detection_frames"]
        ]

        if len(recent_trajectory) < 2:
            return False

        # Calculate ball speed changes
        speeds = []
        for i in range(1, len(recent_trajectory)):
            prev_x, prev_y, prev_frame = recent_trajectory[i - 1]
            curr_x, curr_y, curr_frame = recent_trajectory[i]

            distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            frame_diff = max(1, curr_frame - prev_frame)
            speed = distance / frame_diff
            speeds.append(speed)

        if not speeds:
            return False

        # Check if there was a significant speed increase (kick)
        max_speed = max(speeds)
        avg_speed = sum(speeds) / len(speeds)

        speed_increase = max_speed > self.config["ball_speed_threshold"]

        # Check if player was close to ball when speed increased
        player_was_close = False
        if player_id in self.player_positions:
            player_trajectory = self.player_positions[player_id]
            for pos_x, pos_y, pos_frame in player_trajectory:
                if abs(pos_frame - frame_num) <= self.config["kick_detection_frames"]:
                    distance = math.sqrt(
                        (pos_x - ball_position[0]) ** 2
                        + (pos_y - ball_position[1]) ** 2
                    )
                    if distance <= self.config["kick_detection_distance"]:
                        player_was_close = True
                        break

        return speed_increase and player_was_close

    def _validate_goal_sequence(
        self, ball_position: Tuple[int, int], goal_side: str, frame_num: int
    ) -> bool:
        """
        Validate that this is part of a valid goal sequence.

        This checks temporal consistency and trajectory analysis.
        """
        if len(self.ball_trajectory) < self.config["sequence_validation_frames"]:
            return False

        # Check trajectory consistency towards goal
        recent_trajectory = [
            t
            for t in self.ball_trajectory
            if frame_num - t[2] <= self.config["sequence_validation_frames"]
        ]

        if len(recent_trajectory) < 3:
            return False

        # Analyze trajectory direction consistency
        directions_to_goal = []
        for x, y, _ in recent_trajectory:
            # Simple direction analysis - ball should be moving towards goal
            if goal_side == "left":
                direction_score = -x  # Lower x values are better for left goal
            else:  # right goal
                direction_score = x  # Higher x values are better for right goal
            directions_to_goal.append(direction_score)

        # Check if trajectory is generally consistent towards goal
        if len(directions_to_goal) >= 2:
            direction_trend = directions_to_goal[-1] - directions_to_goal[0]
            if goal_side == "left":
                consistent_direction = direction_trend < 0  # Moving left
            else:
                consistent_direction = direction_trend > 0  # Moving right

            return consistent_direction

        return True  # Default to valid if we can't determine

    def _calculate_goal_confidence(
        self,
        ball_confidence: float,
        player_distance: Optional[float],
        kick_detected: bool,
        temporal_validation: bool,
    ) -> float:
        """Calculate overall goal detection confidence."""
        confidence = ball_confidence * 0.3  # Base ball detection confidence

        if player_distance is not None:
            # Closer player = higher confidence
            distance_score = max(
                0, 1 - (player_distance / self.config["player_ball_max_distance"])
            )
            confidence += distance_score * 0.2

        if kick_detected:
            confidence += 0.3

        if temporal_validation:
            confidence += 0.2

        return min(1.0, confidence)

    def _start_goal_sequence(self, frame_num: int) -> int:
        """Start a new goal sequence and return sequence ID."""
        self.goal_sequence_id += 1
        self.current_sequence_frames = [frame_num]
        return self.goal_sequence_id

    def _register_goal_event(
        self,
        frame_num: int,
        team: int,
        player_id: int,
        goal_side: str,
        ball_position: Tuple[int, int],
        confidence: float,
        detection_method: str,
    ):
        """Register a goal event for scoreboard CSV."""
        self.goal_events_count += 1

        # Find kick frame (look back in trajectory)
        kick_frame = None
        for x, y, f in reversed(
            self.ball_trajectory[-self.config["kick_detection_frames"] :]
        ):
            if f < frame_num:
                kick_frame = f
                break

        # Determine sequence frames
        sequence_start = max(0, frame_num - self.config["sequence_validation_frames"])
        sequence_frames = list(range(sequence_start, frame_num + 1))

        # Add to CSV generator
        self.csv_generator.add_goal_event(
            goal_id=self.goal_events_count,
            frame_start=sequence_start,
            frame_end=frame_num,
            frame_goal=frame_num,
            team=team,
            player_id=player_id,
            goal_side=goal_side,
            ball_final_position=ball_position,
            confidence=confidence,
            validation_score=confidence,  # Using same value for now
            kick_frame=kick_frame,
            sequence_frames=sequence_frames,
            detection_method=detection_method,
        )

    def _update_player_positions(
        self, player_detections: List[Dict[str, Any]], frame_num: int
    ):
        """Update player position tracking."""
        for player in player_detections:
            player_id = player.get("player_id")
            if player_id is None:
                continue

            if "bbox" in player:
                bbox = player["bbox"]
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
            elif "position" in player:
                x, y = player["position"]
            else:
                continue

            if player_id not in self.player_positions:
                self.player_positions[player_id] = []

            self.player_positions[player_id].append((x, y, frame_num))

            # Keep only recent positions
            max_positions = self.config["sequence_validation_frames"] * 2
            if len(self.player_positions[player_id]) > max_positions:
                self.player_positions[player_id] = self.player_positions[player_id][
                    -max_positions:
                ]

    def generate_csv_outputs(self, video_name: str) -> Tuple[str, str]:
        """Generate both required CSV files."""
        return self.csv_generator.generate_both_csvs(video_name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal detection statistics."""
        return {
            **self.csv_generator.get_statistics(),
            "total_goals_detected": self.goal_events_count,
            "frames_processed": self.frame_count,
        }

    def reset(self):
        """Reset detector state for new video."""
        self.frame_count = 0
        self.last_goal_frame = -1
        self.goal_sequence_id = 0
        self.current_sequence_frames.clear()
        self.ball_trajectory.clear()
        self.player_positions.clear()
        self.detected_goals.clear()
        self.goal_events_count = 0
        self.csv_generator.clear_data()
