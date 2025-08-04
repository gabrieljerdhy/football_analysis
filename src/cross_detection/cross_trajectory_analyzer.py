"""
Cross Trajectory Analyzer

Analyzes ball trajectories to detect cross-specific movement patterns including
arc-like movements, height estimation, and speed characteristics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cross_event import CrossType


@dataclass
class TrajectorySegment:
    """Represents a segment of ball trajectory."""

    start_frame: int
    end_frame: int
    positions: List[Tuple[int, int]]
    velocities: List[float]
    directions: List[float]
    confidence_scores: List[float]

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def avg_velocity(self) -> float:
        return np.mean(self.velocities) if self.velocities else 0.0

    @property
    def max_velocity(self) -> float:
        return np.max(self.velocities) if self.velocities else 0.0


class CrossTrajectoryAnalyzer:
    """
    Analyzes ball trajectories to identify cross-specific movement patterns.
    """

    def __init__(self, frame_rate: float = 24.0):
        """
        Initialize the trajectory analyzer.

        Args:
            frame_rate: Video frame rate for velocity calculations
        """
        self.frame_rate = frame_rate

        # Cross detection thresholds
        self.config = {
            # Minimum trajectory length for cross detection
            "min_trajectory_frames": 8,
            # Velocity thresholds (pixels per second)
            "min_cross_velocity": 100.0,
            "max_cross_velocity": 800.0,
            # Height estimation parameters
            "height_estimation_factor": 0.3,  # Factor for estimating ball height from y-movement
            "min_cross_height": 20.0,  # Minimum estimated height for crosses
            # Arc detection parameters
            "min_arc_curvature": 0.1,
            "max_direction_change": 45.0,  # Maximum direction change in degrees
            # Distance thresholds
            "min_cross_distance": 100.0,  # Minimum distance in pixels
            "max_cross_distance": 600.0,  # Maximum distance in pixels
            # Smoothness thresholds
            "max_velocity_variance": 0.5,
            "min_trajectory_smoothness": 0.6,
        }

    def analyze_trajectory_for_cross(
        self, trajectory: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a ball trajectory to determine if it represents a cross.

        Args:
            trajectory: List of ball position data with frame info

        Returns:
            Dictionary with cross analysis results or None if not a cross
        """
        if len(trajectory) < self.config["min_trajectory_frames"]:
            return None

        # Extract positions and metadata
        positions = [pos["position"] for pos in trajectory if pos.get("position")]
        frames = [
            pos["frame_num"] for pos in trajectory if pos.get("frame_num") is not None
        ]
        confidences = [pos.get("confidence", 0.5) for pos in trajectory]

        # Ensure frames and positions have same length
        min_length = min(len(positions), len(frames))
        positions = positions[:min_length]
        frames = frames[:min_length]
        confidences = confidences[:min_length]

        if len(positions) < self.config["min_trajectory_frames"]:
            return None

        # Calculate trajectory metrics
        velocities = self._calculate_velocities(positions, frames)
        directions = self._calculate_directions(positions)
        distance = self._calculate_total_distance(positions)

        # Analyze trajectory characteristics
        height_analysis = self._analyze_height_characteristics(positions)
        arc_analysis = self._analyze_arc_characteristics(positions, directions)
        velocity_analysis = self._analyze_velocity_characteristics(velocities)
        smoothness_analysis = self._analyze_trajectory_smoothness(positions, velocities)

        # Determine if trajectory matches cross patterns
        cross_score = self._calculate_cross_score(
            distance,
            height_analysis,
            arc_analysis,
            velocity_analysis,
            smoothness_analysis,
        )

        if cross_score >= 0.6:  # Threshold for cross detection
            return {
                "is_cross": True,
                "cross_score": cross_score,
                "distance": distance,
                "height_analysis": height_analysis,
                "arc_analysis": arc_analysis,
                "velocity_analysis": velocity_analysis,
                "smoothness_analysis": smoothness_analysis,
                "cross_type": self._determine_cross_type(
                    height_analysis, arc_analysis, distance
                ),
                "trajectory_quality": np.mean(confidences),
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "duration_frames": frames[-1] - frames[0],
            }

        return None

    def _calculate_velocities(
        self, positions: List[Tuple[int, int]], frames: List[int]
    ) -> List[float]:
        """Calculate velocities between consecutive positions."""
        velocities = []

        for i in range(1, len(positions)):
            pos1, pos2 = positions[i - 1], positions[i]
            frame_diff = frames[i] - frames[i - 1]

            if frame_diff > 0:
                distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                velocity = (
                    distance / frame_diff
                ) * self.frame_rate  # pixels per second
                velocities.append(velocity)
            else:
                velocities.append(0.0)

        return velocities

    def _calculate_directions(self, positions: List[Tuple[int, int]]) -> List[float]:
        """Calculate movement directions in degrees."""
        directions = []

        for i in range(1, len(positions)):
            pos1, pos2 = positions[i - 1], positions[i]
            dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]

            if dx != 0 or dy != 0:
                angle = np.degrees(np.arctan2(dy, dx))
                directions.append(angle)
            else:
                directions.append(0.0)

        return directions

    def _calculate_total_distance(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate total distance of trajectory."""
        total_distance = 0.0

        for i in range(1, len(positions)):
            pos1, pos2 = positions[i - 1], positions[i]
            distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
            total_distance += distance

        return total_distance

    def _analyze_height_characteristics(
        self, positions: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """Analyze height characteristics of the trajectory."""
        y_positions = [pos[1] for pos in positions]

        # Calculate height metrics
        y_range = max(y_positions) - min(y_positions)
        y_variance = np.var(y_positions)

        # Estimate ball height based on y-movement (simplified)
        estimated_max_height = y_range * self.config["height_estimation_factor"]

        # Analyze height pattern (arc-like vs linear)
        mid_point = len(y_positions) // 2
        start_y, mid_y, end_y = y_positions[0], y_positions[mid_point], y_positions[-1]

        # Check for arc pattern (ball goes up then down, or down then up)
        height_pattern_score = 0.0
        if abs(mid_y - start_y) > 10 and abs(mid_y - end_y) > 10:
            # Arc-like pattern detected
            height_pattern_score = min(abs(mid_y - start_y), abs(mid_y - end_y)) / max(
                abs(mid_y - start_y), abs(mid_y - end_y)
            )

        return {
            "y_range": y_range,
            "y_variance": y_variance,
            "estimated_max_height": estimated_max_height,
            "height_pattern_score": height_pattern_score,
            "is_high_trajectory": estimated_max_height
            >= self.config["min_cross_height"],
        }

    def _analyze_arc_characteristics(
        self, positions: List[Tuple[int, int]], directions: List[float]
    ) -> Dict[str, float]:
        """Analyze arc characteristics of the trajectory."""
        if len(directions) < 3:
            return {"curvature": 0.0, "direction_changes": 0, "arc_score": 0.0}

        # Calculate direction changes
        direction_changes = []
        for i in range(1, len(directions)):
            change = abs(directions[i] - directions[i - 1])
            # Handle angle wraparound
            if change > 180:
                change = 360 - change
            direction_changes.append(change)

        # Calculate curvature (simplified)
        total_direction_change = sum(direction_changes)
        avg_direction_change = np.mean(direction_changes)

        # Calculate arc score based on smooth curvature
        arc_score = 0.0
        if len(direction_changes) > 0:
            # Good arc has moderate, consistent direction changes
            consistency = 1.0 - (
                np.std(direction_changes) / (np.mean(direction_changes) + 1e-6)
            )
            magnitude = min(
                avg_direction_change / self.config["max_direction_change"], 1.0
            )
            arc_score = consistency * magnitude

        return {
            "curvature": total_direction_change,
            "avg_direction_change": avg_direction_change,
            "direction_changes": len([c for c in direction_changes if c > 5.0]),
            "arc_score": arc_score,
        }

    def _analyze_velocity_characteristics(
        self, velocities: List[float]
    ) -> Dict[str, float]:
        """Analyze velocity characteristics of the trajectory."""
        if not velocities:
            return {
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "velocity_variance": 0.0,
                "velocity_score": 0.0,
            }

        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        velocity_variance = np.var(velocities)

        # Calculate velocity score based on cross-appropriate speeds
        velocity_score = 0.0
        if (
            self.config["min_cross_velocity"]
            <= avg_velocity
            <= self.config["max_cross_velocity"]
        ):
            # Velocity is in appropriate range
            velocity_score = 1.0 - abs(
                avg_velocity
                - (
                    self.config["min_cross_velocity"]
                    + self.config["max_cross_velocity"]
                )
                / 2
            ) / (
                (self.config["max_cross_velocity"] - self.config["min_cross_velocity"])
                / 2
            )

        return {
            "avg_velocity": avg_velocity,
            "max_velocity": max_velocity,
            "velocity_variance": velocity_variance,
            "velocity_score": velocity_score,
        }

    def _analyze_trajectory_smoothness(
        self, positions: List[Tuple[int, int]], velocities: List[float]
    ) -> Dict[str, float]:
        """Analyze smoothness of the trajectory."""
        if len(positions) < 3:
            return {"smoothness_score": 0.0, "position_variance": 0.0}

        # Calculate position variance (how much the trajectory deviates from a straight line)
        start_pos, end_pos = positions[0], positions[-1]

        # Calculate expected positions on straight line
        expected_positions = []
        for i, pos in enumerate(positions):
            t = i / (len(positions) - 1)
            expected_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            expected_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            expected_positions.append((expected_x, expected_y))

        # Calculate deviations
        deviations = []
        for actual, expected in zip(positions, expected_positions):
            deviation = np.sqrt(
                (actual[0] - expected[0]) ** 2 + (actual[1] - expected[1]) ** 2
            )
            deviations.append(deviation)

        position_variance = np.var(deviations)

        # Calculate velocity smoothness
        velocity_smoothness = 0.0
        if len(velocities) > 1:
            velocity_changes = [
                abs(velocities[i] - velocities[i - 1])
                for i in range(1, len(velocities))
            ]
            velocity_smoothness = 1.0 / (1.0 + np.mean(velocity_changes))

        # Combined smoothness score
        smoothness_score = velocity_smoothness * (
            1.0 / (1.0 + position_variance / 100.0)
        )

        return {
            "smoothness_score": smoothness_score,
            "position_variance": position_variance,
            "velocity_smoothness": velocity_smoothness,
        }

    def _calculate_cross_score(
        self,
        distance: float,
        height_analysis: Dict,
        arc_analysis: Dict,
        velocity_analysis: Dict,
        smoothness_analysis: Dict,
    ) -> float:
        """Calculate overall cross detection score."""
        score = 0.0

        # Distance score (25% weight)
        if (
            self.config["min_cross_distance"]
            <= distance
            <= self.config["max_cross_distance"]
        ):
            distance_score = 1.0
        else:
            distance_score = 0.5
        score += 0.25 * distance_score

        # Height score (20% weight)
        height_score = 0.5
        if height_analysis["is_high_trajectory"]:
            height_score = 0.8 + 0.2 * height_analysis["height_pattern_score"]
        score += 0.20 * height_score

        # Arc score (25% weight)
        score += 0.25 * arc_analysis["arc_score"]

        # Velocity score (20% weight)
        score += 0.20 * velocity_analysis["velocity_score"]

        # Smoothness score (10% weight)
        score += 0.10 * smoothness_analysis["smoothness_score"]

        return min(score, 1.0)

    def _determine_cross_type(
        self, height_analysis: Dict, arc_analysis: Dict, distance: float
    ) -> CrossType:
        """Determine the type of cross based on trajectory characteristics."""
        # High cross if significant height
        if height_analysis["estimated_max_height"] > 40:
            if distance > 300:
                return CrossType.DEEP_CROSS
            else:
                return CrossType.HIGH_CROSS

        # Low cross if minimal height
        elif height_analysis["estimated_max_height"] < 20:
            return CrossType.LOW_CROSS

        # Cutback if significant direction change
        elif arc_analysis["avg_direction_change"] > 30:
            return CrossType.CUTBACK

        # Default to high cross
        return CrossType.HIGH_CROSS
