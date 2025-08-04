#!/usr/bin/env python3
"""
Dribble Detection System

This module implements computer vision algorithms to detect dribbles in football videos.
A dribble is detected when a player successfully moves past an opponent while maintaining ball control.

The detection system analyzes:
1. Player movement patterns and direction changes
2. Ball possession transitions between players
3. Proximity between attacking and defending players
4. Speed and acceleration changes during ball control
"""

import logging
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.bbox_utils import get_center_of_bbox, measure_distance
from .dribble_event import (
    DribbleEvent,
    DribbleEventValidator,
    DribbleSequence,
    DribbleStatistics,
)


class DribbleDetector:
    """
    Main dribble detection system that analyzes player and ball movements
    to identify successful dribble attempts.
    """

    def __init__(
        self,
        min_dribble_distance: float = 30.0,
        min_dribble_duration: int = 10,
        max_dribble_duration: int = 120,
        opponent_proximity_threshold: float = 80.0,
        ball_control_threshold: float = 50.0,
        confidence_threshold: float = 0.4,
    ):
        """
        Initialize the dribble detection system.

        Args:
            min_dribble_distance: Minimum distance for a valid dribble (pixels)
            min_dribble_duration: Minimum duration for a valid dribble (frames)
            max_dribble_duration: Maximum duration for a valid dribble (frames)
            opponent_proximity_threshold: Distance threshold for opponent interaction (pixels)
            ball_control_threshold: Maximum distance between player and ball for control (pixels)
            confidence_threshold: Minimum confidence for dribble detection
        """
        self.min_dribble_distance = min_dribble_distance
        self.min_dribble_duration = min_dribble_duration
        self.max_dribble_duration = max_dribble_duration
        self.opponent_proximity_threshold = opponent_proximity_threshold
        self.ball_control_threshold = ball_control_threshold
        self.confidence_threshold = confidence_threshold

        # Tracking state
        self.current_possessions = {}  # player_id -> possession data
        self.potential_dribbles = {}  # player_id -> potential dribble data
        self.completed_dribbles = []
        self.frame_count = 0

        # Statistics
        self.team_statistics = {1: DribbleStatistics(), 2: DribbleStatistics()}
        self.player_statistics = defaultdict(DribbleStatistics)

        # Event validator
        self.validator = DribbleEventValidator(
            min_confidence=confidence_threshold, min_duration=min_dribble_duration
        )

        # Movement tracking
        self.player_positions_history = defaultdict(lambda: deque(maxlen=30))
        self.ball_positions_history = deque(maxlen=30)

        self.logger = logging.getLogger(__name__)

    def detect_dribble(
        self,
        frame_num: int,
        players: Dict[int, Dict],
        ball_position: Optional[Tuple[int, int]],
        ball_possessor: int,
        ball_team: int,
    ) -> Optional[DribbleEvent]:
        """
        Detect dribbles in the current frame.

        Args:
            frame_num: Current frame number
            players: Dictionary of player data {player_id: {bbox, team, ...}}
            ball_position: Current ball position (x, y)
            ball_possessor: ID of player currently possessing the ball
            ball_team: Team of player currently possessing the ball

        Returns:
            DribbleEvent if a dribble is detected, None otherwise
        """
        self.frame_count = frame_num

        # Update position histories
        self._update_position_histories(players, ball_position)

        # Skip if no ball possession
        if ball_possessor == -1 or ball_team is None or ball_position is None:
            return None

        # Update current possessions
        self._update_possessions(
            frame_num, players, ball_position, ball_possessor, ball_team
        )

        # Check for potential dribble starts
        self._check_dribble_initiation(
            frame_num, players, ball_position, ball_possessor, ball_team
        )

        # Update ongoing potential dribbles
        self._update_potential_dribbles(
            frame_num, players, ball_position, ball_possessor, ball_team
        )

        # Check for completed dribbles
        completed_dribble = self._check_dribble_completion(
            frame_num, players, ball_possessor, ball_team
        )

        # Also check for ongoing dribbles that meet completion criteria
        if not completed_dribble:
            completed_dribble = self._check_ongoing_dribble_completion(
                frame_num, ball_possessor
            )

        return completed_dribble

    def _update_position_histories(
        self, players: Dict[int, Dict], ball_position: Optional[Tuple[int, int]]
    ):
        """Update position histories for movement analysis."""
        # Update player positions
        for player_id, player_data in players.items():
            if "bbox" in player_data:
                position = get_center_of_bbox(player_data["bbox"])
                self.player_positions_history[player_id].append(position)

        # Update ball position
        if ball_position:
            self.ball_positions_history.append(ball_position)

    def _update_possessions(
        self,
        frame_num: int,
        players: Dict[int, Dict],
        ball_position: Tuple[int, int],
        ball_possessor: int,
        ball_team: int,
    ):
        """Update current ball possession tracking."""
        if ball_possessor not in self.current_possessions:
            # New possession
            player_position = get_center_of_bbox(players[ball_possessor]["bbox"])
            self.current_possessions[ball_possessor] = {
                "start_frame": frame_num,
                "start_position": player_position,
                "team": ball_team,
                "positions": [player_position],
                "ball_positions": [ball_position],
                "frames": [frame_num],
            }
        else:
            # Continue possession
            player_position = get_center_of_bbox(players[ball_possessor]["bbox"])
            possession = self.current_possessions[ball_possessor]
            possession["positions"].append(player_position)
            possession["ball_positions"].append(ball_position)
            possession["frames"].append(frame_num)

        # Clean up old possessions
        to_remove = []
        for player_id, possession in self.current_possessions.items():
            if (
                player_id != ball_possessor
                and frame_num - possession["frames"][-1] > 30
            ):
                to_remove.append(player_id)

        for player_id in to_remove:
            del self.current_possessions[player_id]

    def _check_dribble_initiation(
        self,
        frame_num: int,
        players: Dict[int, Dict],
        ball_position: Tuple[int, int],
        ball_possessor: int,
        ball_team: int,
    ):
        """Check if a new dribble is being initiated."""
        if ball_possessor in self.potential_dribbles:
            return  # Already tracking a potential dribble for this player

        # Check if player has had possession long enough to start a dribble
        if ball_possessor not in self.current_possessions:
            return

        possession = self.current_possessions[ball_possessor]
        possession_duration = frame_num - possession["start_frame"]

        if possession_duration < 5:  # Need minimum possession time
            return

        # Check for nearby opponents
        nearby_opponents = self._find_nearby_opponents(
            players, ball_possessor, ball_team
        )

        if not nearby_opponents:
            return  # No opponents nearby, not a dribble situation

        # Initiate potential dribble tracking
        player_position = get_center_of_bbox(players[ball_possessor]["bbox"])
        self.potential_dribbles[ball_possessor] = {
            "player_id": ball_possessor,
            "start_frame": frame_num,
            "start_position": player_position,
            "start_ball_position": ball_position,
            "team": ball_team,
            "opponents": nearby_opponents,
            "positions": [player_position],
            "ball_positions": [ball_position],
            "frames": [frame_num],
            "direction_changes": 0,
            "speed_changes": 0,
            "max_speed": 0.0,
            "distances": [0.0],
        }

    def _find_nearby_opponents(
        self, players: Dict[int, Dict], ball_possessor: int, ball_team: int
    ) -> List[Dict]:
        """Find opponents within proximity threshold of the ball possessor."""
        if ball_possessor not in players:
            return []

        possessor_position = get_center_of_bbox(players[ball_possessor]["bbox"])
        nearby_opponents = []

        for player_id, player_data in players.items():
            if player_id == ball_possessor:
                continue

            player_team = player_data.get("team", None)
            if player_team == ball_team:
                continue  # Same team, not an opponent

            player_position = get_center_of_bbox(player_data["bbox"])
            distance = measure_distance(possessor_position, player_position)

            if distance <= self.opponent_proximity_threshold:
                nearby_opponents.append(
                    {
                        "player_id": player_id,
                        "team": player_team,
                        "position": player_position,
                        "distance": distance,
                    }
                )

        return nearby_opponents

    def _update_potential_dribbles(
        self,
        frame_num: int,
        players: Dict[int, Dict],
        ball_position: Tuple[int, int],
        ball_possessor: int,
        ball_team: int,
    ):
        """Update ongoing potential dribbles."""
        to_remove = []

        for player_id, dribble_data in self.potential_dribbles.items():
            if player_id != ball_possessor:
                # Player lost possession, check if dribble was completed
                self._evaluate_dribble_completion(player_id, frame_num)
                to_remove.append(player_id)
                continue

            # Update dribble tracking data
            if player_id in players:
                player_position = get_center_of_bbox(players[player_id]["bbox"])

                # Calculate movement metrics
                prev_position = dribble_data["positions"][-1]
                distance = measure_distance(prev_position, player_position)
                total_distance = dribble_data["distances"][-1] + distance

                # Calculate speed
                speed = distance  # pixels per frame
                if speed > dribble_data["max_speed"]:
                    dribble_data["max_speed"] = speed

                # Detect direction changes
                if len(dribble_data["positions"]) >= 3:
                    direction_change = self._detect_direction_change(
                        dribble_data["positions"][-2:] + [player_position]
                    )
                    if direction_change:
                        dribble_data["direction_changes"] += 1

                # Update tracking data
                dribble_data["positions"].append(player_position)
                dribble_data["ball_positions"].append(ball_position)
                dribble_data["frames"].append(frame_num)
                dribble_data["distances"].append(total_distance)

                # Check if dribble has gone on too long
                duration = frame_num - dribble_data["start_frame"]
                if duration > self.max_dribble_duration:
                    to_remove.append(player_id)

        # Remove completed or failed dribbles
        for player_id in to_remove:
            if player_id in self.potential_dribbles:
                del self.potential_dribbles[player_id]

    def _detect_direction_change(self, positions: List[Tuple[int, int]]) -> bool:
        """Detect significant direction changes in player movement."""
        if len(positions) < 3:
            return False

        # Calculate vectors
        v1 = (positions[1][0] - positions[0][0], positions[1][1] - positions[0][1])
        v2 = (positions[2][0] - positions[1][0], positions[2][1] - positions[1][1])

        # Calculate angle between vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return False

        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle = math.acos(cos_angle)

        # Consider it a direction change if angle > 45 degrees
        return angle > math.pi / 4

    def _check_dribble_completion(
        self,
        frame_num: int,
        players: Dict[int, Dict],
        ball_possessor: int,
        ball_team: int,
    ) -> Optional[DribbleEvent]:
        """Check if any potential dribbles have been completed."""
        for player_id, dribble_data in list(self.potential_dribbles.items()):
            if player_id != ball_possessor:
                # Player lost possession, evaluate if it was a successful dribble
                return self._evaluate_dribble_completion(player_id, frame_num)

        return None

    def _check_ongoing_dribble_completion(
        self, frame_num: int, ball_possessor: int
    ) -> Optional[DribbleEvent]:
        """Check if an ongoing dribble meets completion criteria even without possession loss."""
        if ball_possessor not in self.potential_dribbles:
            return None

        dribble_data = self.potential_dribbles[ball_possessor]

        # Calculate current metrics
        duration = frame_num - dribble_data["start_frame"]
        total_distance = (
            dribble_data["distances"][-1] if dribble_data["distances"] else 0
        )

        # Check if dribble meets completion criteria
        if (
            duration
            >= self.min_dribble_duration * 2  # Longer duration for ongoing check
            and total_distance >= self.min_dribble_distance * 2
        ):  # More distance for ongoing check

            # Calculate confidence
            confidence = self._calculate_dribble_confidence(
                dribble_data, duration, total_distance
            )

            if confidence >= self.confidence_threshold:
                # Create and validate event
                dribble_event = self._create_dribble_event(
                    dribble_data, frame_num, confidence
                )

                if self.validator.validate_event(dribble_event):
                    # Update statistics
                    self._update_statistics(dribble_event)

                    # Add to completed dribbles
                    self.completed_dribbles.append(dribble_event)

                    # Remove from potential dribbles to avoid duplicate detection
                    del self.potential_dribbles[ball_possessor]

                    self.logger.info(
                        f"Ongoing dribble completed: Player {ball_possessor}, "
                        f"Confidence {confidence:.3f}, Duration {duration} frames"
                    )

                    return dribble_event

        return None

    def _evaluate_dribble_completion(
        self, player_id: int, frame_num: int
    ) -> Optional[DribbleEvent]:
        """Evaluate if a potential dribble was successfully completed."""
        if player_id not in self.potential_dribbles:
            return None

        dribble_data = self.potential_dribbles[player_id]

        # Calculate dribble metrics
        duration = frame_num - dribble_data["start_frame"]
        total_distance = (
            dribble_data["distances"][-1] if dribble_data["distances"] else 0
        )

        # Check minimum requirements
        if (
            duration < self.min_dribble_duration
            or total_distance < self.min_dribble_distance
        ):
            return None

        # Calculate confidence score
        confidence = self._calculate_dribble_confidence(
            dribble_data, duration, total_distance
        )

        if confidence < self.confidence_threshold:
            return None

        # Create dribble event
        dribble_event = self._create_dribble_event(dribble_data, frame_num, confidence)

        # Validate event
        if not self.validator.validate_event(dribble_event):
            return None

        # Update statistics
        self._update_statistics(dribble_event)

        # Add to completed dribbles
        self.completed_dribbles.append(dribble_event)

        self.logger.info(
            f"Dribble detected: Player {player_id}, Team {dribble_event.team}, "
            f"Confidence {confidence:.3f}, Duration {duration} frames"
        )

        return dribble_event

    def _calculate_dribble_confidence(
        self, dribble_data: Dict, duration: int, total_distance: float
    ) -> float:
        """Calculate confidence score for a dribble event."""
        confidence = 0.0

        # Distance factor (30% weight)
        distance_score = min(1.0, total_distance / 100.0)  # Normalize to 100 pixels
        confidence += distance_score * 0.3

        # Duration factor (20% weight)
        duration_score = min(1.0, duration / 60.0)  # Normalize to 60 frames
        confidence += duration_score * 0.2

        # Direction changes factor (25% weight)
        direction_score = min(
            1.0, dribble_data["direction_changes"] / 3.0
        )  # Normalize to 3 changes
        confidence += direction_score * 0.25

        # Speed factor (15% weight)
        speed_score = min(
            1.0, dribble_data["max_speed"] / 10.0
        )  # Normalize to 10 pixels/frame
        confidence += speed_score * 0.15

        # Opponent proximity factor (10% weight)
        opponent_score = min(
            1.0, len(dribble_data["opponents"]) / 2.0
        )  # Normalize to 2 opponents
        confidence += opponent_score * 0.1

        return min(1.0, confidence)

    def _create_dribble_event(
        self, dribble_data: Dict, end_frame: int, confidence: float
    ) -> DribbleEvent:
        """Create a DribbleEvent from dribble tracking data."""
        start_frame = dribble_data["start_frame"]
        duration = end_frame - start_frame
        total_distance = (
            dribble_data["distances"][-1] if dribble_data["distances"] else 0
        )

        # Calculate average speed
        avg_speed = total_distance / duration if duration > 0 else 0

        # Find closest opponent
        closest_opponent = None
        min_distance = float("inf")
        for opponent in dribble_data["opponents"]:
            if opponent["distance"] < min_distance:
                min_distance = opponent["distance"]
                closest_opponent = opponent

        # Calculate trajectory smoothness
        trajectory_smoothness = self._calculate_trajectory_smoothness(
            dribble_data["positions"]
        )

        # Calculate ball control consistency
        ball_control_consistency = self._calculate_ball_control_consistency(
            dribble_data["positions"], dribble_data["ball_positions"]
        )

        return DribbleEvent(
            frame_num=end_frame,
            team=dribble_data["team"],
            player_id=dribble_data["player_id"],
            ball_position=dribble_data["ball_positions"][-1],
            player_position=dribble_data["positions"][-1],
            confidence=confidence,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration,
            distance_covered=total_distance,
            direction_changes=dribble_data["direction_changes"],
            speed_changes=dribble_data["speed_changes"],
            opponent_player_id=(
                closest_opponent["player_id"] if closest_opponent else None
            ),
            opponent_team=closest_opponent["team"] if closest_opponent else None,
            opponent_distance=(
                closest_opponent["distance"] if closest_opponent else None
            ),
            avg_speed=avg_speed,
            max_speed=dribble_data["max_speed"],
            trajectory_smoothness=trajectory_smoothness,
            ball_control_consistency=ball_control_consistency,
            ball_touches=len(dribble_data["ball_positions"]),
            validation_score=confidence,
        )

    def _calculate_trajectory_smoothness(
        self, positions: List[Tuple[int, int]]
    ) -> float:
        """Calculate trajectory smoothness score (0-1)."""
        if len(positions) < 3:
            return 0.0

        # Calculate direction changes
        direction_changes = 0
        total_segments = len(positions) - 1

        for i in range(len(positions) - 2):
            if self._detect_direction_change(positions[i : i + 3]):
                direction_changes += 1

        # Smoothness is inverse of direction change frequency
        if total_segments == 0:
            return 0.0

        change_frequency = direction_changes / total_segments
        smoothness = max(0.0, 1.0 - change_frequency)

        return smoothness

    def _calculate_ball_control_consistency(
        self,
        player_positions: List[Tuple[int, int]],
        ball_positions: List[Tuple[int, int]],
    ) -> float:
        """Calculate ball control consistency score (0-1)."""
        if len(player_positions) != len(ball_positions) or len(player_positions) < 2:
            return 0.0

        distances = []
        for player_pos, ball_pos in zip(player_positions, ball_positions):
            distance = measure_distance(player_pos, ball_pos)
            distances.append(distance)

        # Calculate variance in ball-player distances
        mean_distance = sum(distances) / len(distances)
        variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)

        # Normalize variance to 0-1 scale (lower variance = higher consistency)
        max_variance = 50.0  # Reasonable maximum variance
        consistency = max(0.0, 1.0 - (variance / max_variance))

        return consistency

    def _update_statistics(self, dribble_event: DribbleEvent):
        """Update team and player statistics with the new dribble event."""
        # Update team statistics
        team_stats = self.team_statistics[dribble_event.team]
        team_stats.total_dribbles += 1
        team_stats.successful_dribbles += 1
        team_stats.events_count += 1

        # Update averages
        total_events = team_stats.events_count
        team_stats.avg_confidence = (
            team_stats.avg_confidence * (total_events - 1) + dribble_event.confidence
        ) / total_events
        team_stats.avg_duration = (
            team_stats.avg_duration * (total_events - 1) + dribble_event.duration_frames
        ) / total_events
        team_stats.avg_distance = (
            team_stats.avg_distance * (total_events - 1)
            + dribble_event.distance_covered
        ) / total_events
        team_stats.avg_speed = (
            team_stats.avg_speed * (total_events - 1) + dribble_event.avg_speed
        ) / total_events
        team_stats.avg_ball_control_quality = (
            team_stats.avg_ball_control_quality * (total_events - 1)
            + dribble_event.ball_control_consistency
        ) / total_events
        team_stats.avg_trajectory_smoothness = (
            team_stats.avg_trajectory_smoothness * (total_events - 1)
            + dribble_event.trajectory_smoothness
        ) / total_events

        # Calculate success rate
        team_stats.success_rate = (
            team_stats.successful_dribbles / team_stats.total_dribbles
        )

        # Update player statistics
        player_stats = self.player_statistics[dribble_event.player_id]
        player_stats.total_dribbles += 1
        player_stats.successful_dribbles += 1
        player_stats.events_count += 1

        # Update player averages (similar to team stats)
        player_total = player_stats.events_count
        player_stats.avg_confidence = (
            player_stats.avg_confidence * (player_total - 1) + dribble_event.confidence
        ) / player_total
        player_stats.avg_duration = (
            player_stats.avg_duration * (player_total - 1)
            + dribble_event.duration_frames
        ) / player_total
        player_stats.avg_distance = (
            player_stats.avg_distance * (player_total - 1)
            + dribble_event.distance_covered
        ) / player_total
        player_stats.success_rate = (
            player_stats.successful_dribbles / player_stats.total_dribbles
        )

    def get_team_statistics(self) -> Dict[int, DribbleStatistics]:
        """Get current team dribble statistics."""
        return dict(self.team_statistics)

    def get_player_statistics(self) -> Dict[int, DribbleStatistics]:
        """Get current player dribble statistics."""
        return dict(self.player_statistics)

    def get_completed_dribbles(self) -> List[DribbleEvent]:
        """Get list of all completed dribble events."""
        return self.completed_dribbles.copy()

    def reset_statistics(self):
        """Reset all statistics and tracking data."""
        self.team_statistics = {1: DribbleStatistics(), 2: DribbleStatistics()}
        self.player_statistics = defaultdict(DribbleStatistics)
        self.completed_dribbles = []
        self.current_possessions = {}
        self.potential_dribbles = {}
        self.player_positions_history = defaultdict(lambda: deque(maxlen=30))
        self.ball_positions_history = deque(maxlen=30)
        self.frame_count = 0
