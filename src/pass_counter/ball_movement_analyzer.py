import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math

from src.utils import get_center_of_bbox, measure_distance


@dataclass
class BallMovementEvent:
    """Represents a ball movement event with enhanced tracking data."""
    frame_num: int
    position: Tuple[float, float]
    velocity: float
    direction: float
    confidence: float
    source: str
    acceleration: float = 0.0
    is_possession_change: bool = False
    event_type: str = "movement"  # movement, pass, tackle, interception, shot


@dataclass
class BallTrajectorySegment:
    """Represents a segment of ball trajectory between possession changes."""
    start_frame: int
    end_frame: int
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    avg_velocity: float
    max_velocity: float
    direction_change: float
    confidence_score: float
    trajectory_type: str  # pass, dribble, shot, clearance


class EnhancedBallMovementAnalyzer:
    """
    Analyzes ball movement patterns using enhanced ball tracking data.
    Provides trajectory analysis, velocity calculations, and movement pattern recognition.
    """
    
    def __init__(self, frame_rate: float = 24.0, field_scale: float = 1.0):
        """
        Initialize the Enhanced Ball Movement Analyzer.
        
        Args:
            frame_rate (float): Video frame rate for velocity calculations
            field_scale (float): Scale factor for converting pixels to real-world units
        """
        self.frame_rate = frame_rate
        self.field_scale = field_scale
        
        # Movement analysis parameters
        self.min_velocity_threshold = 0.5  # m/s - minimum velocity to consider as movement
        self.max_velocity_threshold = 30.0  # m/s - maximum realistic ball velocity
        self.direction_change_threshold = 45.0  # degrees - significant direction change
        self.confidence_threshold = 0.3  # minimum confidence for reliable tracking
        
        # Trajectory analysis parameters
        self.min_trajectory_frames = 5  # minimum frames for trajectory analysis
        self.velocity_smoothing_window = 3  # frames for velocity smoothing
        
        # Movement event storage
        self.movement_events: List[BallMovementEvent] = []
        self.trajectory_segments: List[BallTrajectorySegment] = []
        
        # Analysis cache
        self._velocity_cache = {}
        self._direction_cache = {}
        
    def analyze_ball_tracks(self, ball_tracks: List[Dict]) -> Dict:
        """
        Analyze ball tracks to extract movement patterns and statistics.
        
        Args:
            ball_tracks (List[Dict]): Ball tracking data from enhanced tracker
            
        Returns:
            Dict: Comprehensive ball movement analysis results
        """
        # Clear previous analysis
        self.movement_events.clear()
        self.trajectory_segments.clear()
        self._velocity_cache.clear()
        self._direction_cache.clear()
        
        # Extract ball positions and metadata
        positions, confidences, sources = self._extract_ball_data(ball_tracks)
        
        if len(positions) < 2:
            return self._empty_analysis_result()
        
        # Calculate velocities and directions
        velocities = self._calculate_velocities(positions)
        directions = self._calculate_directions(positions)
        accelerations = self._calculate_accelerations(velocities)
        
        # Create movement events
        self._create_movement_events(
            positions, velocities, directions, accelerations, confidences, sources
        )
        
        # Analyze trajectory segments
        self._analyze_trajectory_segments()
        
        # Generate comprehensive analysis
        return self._generate_analysis_results()
    
    def _extract_ball_data(self, ball_tracks: List[Dict]) -> Tuple[List, List, List]:
        """Extract positions, confidences, and sources from ball tracks."""
        positions = []
        confidences = []
        sources = []
        
        for frame_data in ball_tracks:
            if 1 in frame_data and frame_data[1]:
                ball_info = frame_data[1]
                
                # Get position
                if "position" in ball_info:
                    position = ball_info["position"]
                elif "bbox" in ball_info:
                    position = get_center_of_bbox(ball_info["bbox"])
                else:
                    position = None
                
                positions.append(position)
                confidences.append(ball_info.get("confidence", 0.5))
                sources.append(ball_info.get("source", "unknown"))
            else:
                positions.append(None)
                confidences.append(0.0)
                sources.append("missing")
        
        return positions, confidences, sources
    
    def _calculate_velocities(self, positions: List[Optional[Tuple]]) -> List[float]:
        """Calculate ball velocities between consecutive frames."""
        velocities = [0.0]  # First frame has zero velocity
        
        for i in range(1, len(positions)):
            if positions[i] is None or positions[i-1] is None:
                velocities.append(0.0)
                continue
            
            # Calculate distance and velocity
            distance = measure_distance(positions[i-1], positions[i]) * self.field_scale
            velocity = distance * self.frame_rate  # Convert to units per second
            
            # Apply velocity constraints
            velocity = max(0.0, min(velocity, self.max_velocity_threshold))
            velocities.append(velocity)
        
        # Apply smoothing
        return self._smooth_velocities(velocities)
    
    def _calculate_directions(self, positions: List[Optional[Tuple]]) -> List[float]:
        """Calculate ball movement directions in degrees."""
        directions = [0.0]  # First frame has no direction
        
        for i in range(1, len(positions)):
            if positions[i] is None or positions[i-1] is None:
                directions.append(directions[-1] if directions else 0.0)
                continue
            
            # Calculate direction angle
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                # No movement, keep previous direction
                directions.append(directions[-1])
            else:
                direction = math.degrees(math.atan2(dy, dx))
                directions.append(direction)
        
        return directions
    
    def _calculate_accelerations(self, velocities: List[float]) -> List[float]:
        """Calculate ball accelerations from velocity changes."""
        accelerations = [0.0]  # First frame has zero acceleration
        
        for i in range(1, len(velocities)):
            acceleration = (velocities[i] - velocities[i-1]) * self.frame_rate
            accelerations.append(acceleration)
        
        return accelerations
    
    def _smooth_velocities(self, velocities: List[float]) -> List[float]:
        """Apply smoothing to velocity data to reduce noise."""
        if len(velocities) < self.velocity_smoothing_window:
            return velocities
        
        smoothed = velocities.copy()
        window = self.velocity_smoothing_window
        
        for i in range(window, len(velocities) - window):
            window_values = velocities[i-window:i+window+1]
            smoothed[i] = sum(window_values) / len(window_values)
        
        return smoothed
    
    def _create_movement_events(self, positions, velocities, directions, 
                              accelerations, confidences, sources):
        """Create movement events from calculated data."""
        for i in range(len(positions)):
            if positions[i] is None:
                continue
                
            event = BallMovementEvent(
                frame_num=i,
                position=positions[i],
                velocity=velocities[i],
                direction=directions[i],
                confidence=confidences[i],
                source=sources[i],
                acceleration=accelerations[i]
            )
            
            self.movement_events.append(event)
    
    def _analyze_trajectory_segments(self):
        """Analyze ball trajectory segments between significant events."""
        if len(self.movement_events) < self.min_trajectory_frames:
            return
        
        # Find trajectory breakpoints (significant direction/velocity changes)
        breakpoints = [0]  # Start with first frame
        
        for i in range(1, len(self.movement_events) - 1):
            current_event = self.movement_events[i]
            prev_event = self.movement_events[i-1]
            
            # Check for significant direction change
            direction_diff = abs(current_event.direction - prev_event.direction)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff
            
            # Check for significant velocity change
            velocity_ratio = (current_event.velocity / max(prev_event.velocity, 0.1))
            
            if (direction_diff > self.direction_change_threshold or 
                velocity_ratio > 2.0 or velocity_ratio < 0.5):
                breakpoints.append(i)
        
        breakpoints.append(len(self.movement_events) - 1)  # End with last frame
        
        # Create trajectory segments
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            
            if end_idx - start_idx < self.min_trajectory_frames:
                continue
            
            segment_events = self.movement_events[start_idx:end_idx + 1]
            self._create_trajectory_segment(segment_events)
    
    def _create_trajectory_segment(self, events: List[BallMovementEvent]):
        """Create a trajectory segment from movement events."""
        if len(events) < 2:
            return
        
        start_event = events[0]
        end_event = events[-1]
        
        # Calculate segment statistics
        velocities = [e.velocity for e in events]
        avg_velocity = sum(velocities) / len(velocities)
        max_velocity = max(velocities)
        
        # Calculate direction change
        direction_change = abs(end_event.direction - start_event.direction)
        if direction_change > 180:
            direction_change = 360 - direction_change
        
        # Calculate confidence score
        confidences = [e.confidence for e in events]
        confidence_score = sum(confidences) / len(confidences)
        
        # Determine trajectory type
        trajectory_type = self._classify_trajectory_type(events)
        
        segment = BallTrajectorySegment(
            start_frame=start_event.frame_num,
            end_frame=end_event.frame_num,
            start_position=start_event.position,
            end_position=end_event.position,
            avg_velocity=avg_velocity,
            max_velocity=max_velocity,
            direction_change=direction_change,
            confidence_score=confidence_score,
            trajectory_type=trajectory_type
        )
        
        self.trajectory_segments.append(segment)
    
    def _classify_trajectory_type(self, events: List[BallMovementEvent]) -> str:
        """Classify the type of ball trajectory based on movement patterns."""
        if len(events) < 2:
            return "unknown"
        
        avg_velocity = sum(e.velocity for e in events) / len(events)
        max_velocity = max(e.velocity for e in events)
        
        # High velocity suggests pass or shot
        if max_velocity > 15.0:
            return "shot" if max_velocity > 20.0 else "pass"
        
        # Low velocity with consistent direction suggests dribble
        if avg_velocity < 3.0:
            return "dribble"
        
        # Medium velocity suggests pass
        if avg_velocity > 5.0:
            return "pass"
        
        return "movement"
    
    def _generate_analysis_results(self) -> Dict:
        """Generate comprehensive analysis results."""
        if not self.movement_events:
            return self._empty_analysis_result()
        
        # Calculate overall statistics
        velocities = [e.velocity for e in self.movement_events]
        confidences = [e.confidence for e in self.movement_events]
        
        results = {
            "total_frames": len(self.movement_events),
            "avg_velocity": sum(velocities) / len(velocities),
            "max_velocity": max(velocities),
            "avg_confidence": sum(confidences) / len(confidences),
            "trajectory_segments": len(self.trajectory_segments),
            "movement_events": self.movement_events,
            "trajectory_data": self.trajectory_segments,
            "quality_metrics": self._calculate_quality_metrics()
        }
        
        return results
    
    def _calculate_quality_metrics(self) -> Dict:
        """Calculate data quality metrics for the analysis."""
        if not self.movement_events:
            return {}
        
        high_conf_events = [e for e in self.movement_events if e.confidence > 0.5]
        interpolated_events = [e for e in self.movement_events if "interpolated" in e.source]
        
        return {
            "high_confidence_ratio": len(high_conf_events) / len(self.movement_events),
            "interpolated_ratio": len(interpolated_events) / len(self.movement_events),
            "tracking_continuity": self._calculate_tracking_continuity(),
            "velocity_consistency": self._calculate_velocity_consistency()
        }
    
    def _calculate_tracking_continuity(self) -> float:
        """Calculate tracking continuity score."""
        if len(self.movement_events) < 2:
            return 0.0
        
        continuous_frames = 0
        for i in range(1, len(self.movement_events)):
            if self.movement_events[i].frame_num == self.movement_events[i-1].frame_num + 1:
                continuous_frames += 1
        
        return continuous_frames / (len(self.movement_events) - 1)
    
    def _calculate_velocity_consistency(self) -> float:
        """Calculate velocity consistency score."""
        velocities = [e.velocity for e in self.movement_events]
        if len(velocities) < 2:
            return 1.0
        
        velocity_changes = []
        for i in range(1, len(velocities)):
            if velocities[i-1] > 0:
                change_ratio = abs(velocities[i] - velocities[i-1]) / velocities[i-1]
                velocity_changes.append(min(change_ratio, 2.0))  # Cap at 200% change
        
        if not velocity_changes:
            return 1.0
        
        avg_change = sum(velocity_changes) / len(velocity_changes)
        return max(0.0, 1.0 - avg_change)  # Higher score for more consistent velocities
    
    def _empty_analysis_result(self) -> Dict:
        """Return empty analysis result structure."""
        return {
            "total_frames": 0,
            "avg_velocity": 0.0,
            "max_velocity": 0.0,
            "avg_confidence": 0.0,
            "trajectory_segments": 0,
            "movement_events": [],
            "trajectory_data": [],
            "quality_metrics": {}
        }
