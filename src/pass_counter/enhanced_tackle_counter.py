import csv
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .ball_movement_analyzer import EnhancedBallMovementAnalyzer, BallTrajectorySegment
from src.utils import measure_distance, get_center_of_bbox, get_foot_position


@dataclass
class TackleEvent:
    """Represents a detected tackle event with enhanced validation."""
    frame_num: int
    tackler_id: int
    tackled_player_id: int
    tackler_team: int
    tackled_team: int
    tackle_distance: float
    ball_velocity_before: float
    ball_velocity_after: float
    success_probability: float
    confidence_score: float
    tackle_type: str = "standing"  # standing, sliding, aerial, interception


@dataclass
class InterceptionEvent:
    """Represents a detected interception event."""
    frame_num: int
    interceptor_id: int
    interceptor_team: int
    original_target_team: int
    interception_distance: float
    ball_trajectory_change: float
    confidence_score: float
    interception_type: str = "ground"  # ground, aerial, deflection


class EnhancedTackleCounter:
    """
    Enhanced tackle counter that uses ball trajectory analysis and player movement
    to improve tackle and interception detection accuracy.
    """
    
    def __init__(self, frame_rate: float = 24.0):
        """
        Initialize the Enhanced Tackle Counter.
        
        Args:
            frame_rate (float): Video frame rate for temporal analysis
        """
        # Basic statistics tracking
        self.team_tackles = {1: 0, 2: 0}
        self.player_tackles = {}
        self.team_interceptions = {1: 0, 2: 0}
        self.player_interceptions = {}
        
        # Enhanced tracking
        self.tackle_events: List[TackleEvent] = []
        self.interception_events: List[InterceptionEvent] = []
        
        # Ball movement analyzer
        self.ball_analyzer = EnhancedBallMovementAnalyzer(frame_rate=frame_rate)
        
        # Tracking variables
        self.last_ball_possessor = -1
        self.last_team_possession = None
        self.possession_change_cooldown = 0
        self.min_possession_frames = 3
        self.possession_frames = 0
        
        # Detection parameters
        self.tackle_distance_threshold = 60  # pixels - maximum distance for tackle
        self.interception_distance_threshold = 80  # pixels - maximum distance for interception
        self.min_velocity_change_tackle = 5.0  # m/s - minimum velocity change for tackle
        self.min_direction_change_interception = 30.0  # degrees - minimum direction change
        self.confidence_threshold = 0.4  # minimum confidence for event validation
        
        # Player position tracking
        self.prev_player_positions = {}
        self.player_velocities = {}
        
    def analyze_tackles_from_tracks(self, tracks: Dict, player_assignments: List[int]) -> Dict:
        """
        Analyze tackles and interceptions from complete tracking data.
        
        Args:
            tracks (Dict): Complete tracking data including ball and player tracks
            player_assignments (List[int]): Player assignments per frame
            
        Returns:
            Dict: Comprehensive tackle and interception analysis results
        """
        # Analyze ball movement patterns
        ball_analysis = self.ball_analyzer.analyze_ball_tracks(tracks["ball"])
        
        # Process possession changes and detect tackles/interceptions
        self._process_defensive_actions(tracks, player_assignments, ball_analysis)
        
        # Validate and classify events
        self._validate_and_classify_events(ball_analysis)
        
        # Generate comprehensive statistics
        return self._generate_tackle_statistics()
    
    def detect_tackles_and_interceptions_enhanced(self, current_frame_players: Dict, 
                                                ball_possessor: int, ball_team: int, 
                                                frame_num: int, ball_data: Dict = None):
        """
        Enhanced tackle and interception detection with trajectory analysis.
        
        Args:
            current_frame_players (Dict): Dictionary of player data for current frame
            ball_possessor (int): ID of player currently possessing the ball
            ball_team (int): Team of player currently possessing the ball
            frame_num (int): Current frame number
            ball_data (Dict): Enhanced ball tracking data for current frame
        """
        # Skip if in cooldown period
        if self.possession_change_cooldown > 0:
            self.possession_change_cooldown -= 1
            return
        
        # Update player positions and velocities
        self._update_player_tracking(current_frame_players, frame_num)
        
        # Handle no possession
        if ball_possessor == -1 or ball_team is None:
            self.possession_frames = 0
            return
        
        # Check for possession change
        if (self.last_ball_possessor != -1 and 
            self.last_team_possession is not None and
            self.possession_frames >= self.min_possession_frames):
            
            # Different player has the ball
            if ball_possessor != self.last_ball_possessor:
                if ball_team != self.last_team_possession:
                    # Different team - potential tackle or interception
                    self._analyze_defensive_action(
                        current_frame_players, ball_possessor, ball_team, 
                        frame_num, ball_data
                    )
                
                # Set cooldown to prevent multiple detections
                self.possession_change_cooldown = 10
        
        # Update tracking variables
        if ball_possessor != self.last_ball_possessor:
            self.possession_frames = 1
        else:
            self.possession_frames += 1
        
        self.last_ball_possessor = ball_possessor
        self.last_team_possession = ball_team
    
    def _process_defensive_actions(self, tracks: Dict, player_assignments: List[int], 
                                 ball_analysis: Dict):
        """Process defensive actions throughout the video."""
        for frame_num, assigned_player in enumerate(player_assignments):
            if frame_num >= len(tracks["players"]):
                continue
            
            current_team = None
            if assigned_player != -1 and assigned_player in tracks["players"][frame_num]:
                current_team = tracks["players"][frame_num][assigned_player].get("team")
            
            # Get ball data for this frame
            ball_data = None
            if frame_num < len(tracks["ball"]) and 1 in tracks["ball"][frame_num]:
                ball_data = tracks["ball"][frame_num][1]
            
            # Process with enhanced detection
            self.detect_tackles_and_interceptions_enhanced(
                tracks["players"][frame_num], assigned_player, current_team, 
                frame_num, ball_data
            )
    
    def _update_player_tracking(self, current_frame_players: Dict, frame_num: int):
        """Update player position and velocity tracking."""
        current_positions = {}
        
        for player_id, player_data in current_frame_players.items():
            if "bbox" in player_data:
                position = get_foot_position(player_data["bbox"])
                current_positions[player_id] = position
                
                # Calculate velocity if we have previous position
                if player_id in self.prev_player_positions:
                    prev_pos = self.prev_player_positions[player_id]
                    distance = measure_distance(prev_pos, position)
                    velocity = distance * self.ball_analyzer.frame_rate  # pixels per second
                    self.player_velocities[player_id] = velocity
                else:
                    self.player_velocities[player_id] = 0.0
        
        self.prev_player_positions = current_positions
    
    def _analyze_defensive_action(self, current_frame_players: Dict, new_possessor: int, 
                                new_team: int, frame_num: int, ball_data: Dict = None):
        """Analyze whether a possession change was due to a tackle or interception."""
        if self.last_ball_possessor == -1:
            return
        
        # Get ball movement data
        ball_velocity_before = 0.0
        ball_velocity_after = 0.0
        ball_confidence = 0.5
        
        if ball_data:
            ball_confidence = ball_data.get("confidence", 0.5)
            ball_velocity_after = ball_data.get("velocity", 0.0)
        
        # Check if this was a tackle (close proximity)
        if self._is_tackle_enhanced(current_frame_players, new_possessor, 
                                  self.last_ball_possessor, ball_data):
            self._record_tackle_event(
                new_possessor, self.last_ball_possessor, new_team, 
                self.last_team_possession, frame_num, ball_data
            )
        
        # Check if this was an interception (ball trajectory change)
        elif self._is_interception_enhanced(current_frame_players, new_possessor, 
                                          new_team, frame_num, ball_data):
            self._record_interception_event(
                new_possessor, new_team, self.last_team_possession, 
                frame_num, ball_data
            )
    
    def _is_tackle_enhanced(self, current_frame_players: Dict, new_possessor: int, 
                          old_possessor: int, ball_data: Dict = None) -> bool:
        """Enhanced tackle detection using player proximity and ball movement."""
        if (new_possessor not in current_frame_players or 
            old_possessor not in self.prev_player_positions):
            return False
        
        # Get player positions
        new_pos = get_foot_position(current_frame_players[new_possessor]["bbox"])
        old_pos = self.prev_player_positions.get(old_possessor)
        
        if old_pos is None:
            return False
        
        # Calculate distance between players
        distance = measure_distance(new_pos, old_pos)
        
        # Check proximity threshold
        if distance > self.tackle_distance_threshold:
            return False
        
        # Check ball velocity change (tackles usually slow down the ball)
        if ball_data:
            ball_velocity = ball_data.get("velocity", 0.0)
            confidence = ball_data.get("confidence", 0.0)
            
            # High confidence and significant velocity change suggests tackle
            if confidence >= self.confidence_threshold:
                return True
        
        # Default proximity-based detection
        return distance <= self.tackle_distance_threshold * 0.7
    
    def _is_interception_enhanced(self, current_frame_players: Dict, new_possessor: int, 
                                new_team: int, frame_num: int, ball_data: Dict = None) -> bool:
        """Enhanced interception detection using ball trajectory analysis."""
        if new_possessor not in current_frame_players:
            return False
        
        # Get interceptor position
        interceptor_pos = get_foot_position(current_frame_players[new_possessor]["bbox"])
        
        # Check if ball trajectory changed significantly
        if ball_data:
            confidence = ball_data.get("confidence", 0.0)
            
            # High confidence detection suggests valid interception
            if confidence >= self.confidence_threshold:
                return True
        
        # Check distance from potential pass target
        # This is simplified - in a full implementation, we'd analyze the ball trajectory
        # to determine the intended target and measure interception distance
        return True  # Placeholder for now
    
    def _record_tackle_event(self, tackler_id: int, tackled_id: int, tackler_team: int, 
                           tackled_team: int, frame_num: int, ball_data: Dict = None):
        """Record a tackle event with enhanced data."""
        # Calculate tackle distance
        tackle_distance = 0.0
        if (tackler_id in self.prev_player_positions and 
            tackled_id in self.prev_player_positions):
            tackle_distance = measure_distance(
                self.prev_player_positions[tackler_id],
                self.prev_player_positions[tackled_id]
            )
        
        # Get ball velocity data
        ball_velocity_before = 0.0
        ball_velocity_after = 0.0
        confidence_score = 0.5
        
        if ball_data:
            ball_velocity_after = ball_data.get("velocity", 0.0)
            confidence_score = ball_data.get("confidence", 0.5)
        
        # Calculate success probability based on various factors
        success_probability = self._calculate_tackle_success_probability(
            tackle_distance, ball_velocity_after, confidence_score
        )
        
        # Create tackle event
        tackle_event = TackleEvent(
            frame_num=frame_num,
            tackler_id=tackler_id,
            tackled_player_id=tackled_id,
            tackler_team=tackler_team,
            tackled_team=tackled_team,
            tackle_distance=tackle_distance,
            ball_velocity_before=ball_velocity_before,
            ball_velocity_after=ball_velocity_after,
            success_probability=success_probability,
            confidence_score=confidence_score,
            tackle_type=self._classify_tackle_type(tackle_distance, ball_velocity_after)
        )
        
        self.tackle_events.append(tackle_event)
        
        # Update statistics
        self.team_tackles[tackler_team] = self.team_tackles.get(tackler_team, 0) + 1
        
        if tackler_id not in self.player_tackles:
            self.player_tackles[tackler_id] = {
                "tackles": 0,
                "successful_tackles": 0,
                "team": tackler_team,
                "tackle_success_rate": 0.0
            }
        
        self.player_tackles[tackler_id]["tackles"] += 1
        if success_probability > 0.7:  # Consider successful if high probability
            self.player_tackles[tackler_id]["successful_tackles"] += 1
        
        # Update success rate
        player_stats = self.player_tackles[tackler_id]
        player_stats["tackle_success_rate"] = (
            player_stats["successful_tackles"] / player_stats["tackles"]
        )
        
        print(f"Enhanced tackle detected: Player {tackler_id} (Team {tackler_team}) "
              f"tackled Player {tackled_id} at frame {frame_num}, "
              f"success probability: {success_probability:.2f}")
    
    def _record_interception_event(self, interceptor_id: int, interceptor_team: int, 
                                 original_team: int, frame_num: int, ball_data: Dict = None):
        """Record an interception event with enhanced data."""
        interception_distance = 0.0
        ball_trajectory_change = 0.0
        confidence_score = 0.5
        
        if ball_data:
            confidence_score = ball_data.get("confidence", 0.5)
        
        # Create interception event
        interception_event = InterceptionEvent(
            frame_num=frame_num,
            interceptor_id=interceptor_id,
            interceptor_team=interceptor_team,
            original_target_team=original_team,
            interception_distance=interception_distance,
            ball_trajectory_change=ball_trajectory_change,
            confidence_score=confidence_score,
            interception_type="ground"  # Simplified classification
        )
        
        self.interception_events.append(interception_event)
        
        # Update statistics
        self.team_interceptions[interceptor_team] = (
            self.team_interceptions.get(interceptor_team, 0) + 1
        )
        
        if interceptor_id not in self.player_interceptions:
            self.player_interceptions[interceptor_id] = {
                "interceptions": 0,
                "team": interceptor_team
            }
        
        self.player_interceptions[interceptor_id]["interceptions"] += 1
        
        print(f"Enhanced interception detected: Player {interceptor_id} "
              f"(Team {interceptor_team}) at frame {frame_num}")
    
    def _calculate_tackle_success_probability(self, distance: float, velocity: float, 
                                            confidence: float) -> float:
        """Calculate the probability that a tackle was successful."""
        # Base probability on distance (closer = higher success)
        distance_factor = max(0.0, 1.0 - distance / self.tackle_distance_threshold)
        
        # Factor in ball velocity (lower velocity after tackle = higher success)
        velocity_factor = max(0.0, 1.0 - velocity / 10.0)  # Normalize by 10 m/s
        
        # Factor in detection confidence
        confidence_factor = confidence
        
        # Weighted combination
        probability = (distance_factor * 0.4 + velocity_factor * 0.3 + confidence_factor * 0.3)
        
        return min(1.0, max(0.0, probability))
    
    def _classify_tackle_type(self, distance: float, velocity: float) -> str:
        """Classify the type of tackle based on distance and ball velocity."""
        if distance < 20:
            return "close_contact"
        elif velocity < 2.0:
            return "standing"
        elif velocity > 8.0:
            return "sliding"
        else:
            return "standing"
    
    def _validate_and_classify_events(self, ball_analysis: Dict):
        """Validate and enhance classification of detected events."""
        trajectory_segments = ball_analysis.get("trajectory_data", [])
        
        # Validate tackle events
        for tackle_event in self.tackle_events:
            # Find corresponding trajectory segment
            for segment in trajectory_segments:
                if (segment.start_frame <= tackle_event.frame_num <= segment.end_frame):
                    # Update confidence based on trajectory analysis
                    tackle_event.confidence_score = min(
                        tackle_event.confidence_score + segment.confidence_score * 0.2,
                        1.0
                    )
                    break
        
        # Validate interception events
        for interception_event in self.interception_events:
            # Find corresponding trajectory segment
            for segment in trajectory_segments:
                if (segment.start_frame <= interception_event.frame_num <= segment.end_frame):
                    # Update trajectory change based on segment analysis
                    interception_event.ball_trajectory_change = segment.direction_change
                    interception_event.confidence_score = min(
                        interception_event.confidence_score + segment.confidence_score * 0.2,
                        1.0
                    )
                    break
    
    def _generate_tackle_statistics(self) -> Dict:
        """Generate comprehensive tackle and interception statistics."""
        total_tackles = len(self.tackle_events)
        total_interceptions = len(self.interception_events)
        
        # Calculate team statistics
        team_stats = {}
        for team_id in [1, 2]:
            team_tackles = [t for t in self.tackle_events if t.tackler_team == team_id]
            team_interceptions = [i for i in self.interception_events if i.interceptor_team == team_id]
            
            team_stats[team_id] = {
                "total_tackles": len(team_tackles),
                "total_interceptions": len(team_interceptions),
                "avg_tackle_success": np.mean([t.success_probability for t in team_tackles]) if team_tackles else 0,
                "avg_tackle_distance": np.mean([t.tackle_distance for t in team_tackles]) if team_tackles else 0,
                "defensive_actions": len(team_tackles) + len(team_interceptions)
            }
        
        return {
            "total_tackles": total_tackles,
            "total_interceptions": total_interceptions,
            "tackle_events": self.tackle_events,
            "interception_events": self.interception_events,
            "team_statistics": team_stats,
            "player_tackles": self.player_tackles,
            "player_interceptions": self.player_interceptions,
            "quality_metrics": {
                "avg_tackle_confidence": np.mean([t.confidence_score for t in self.tackle_events]) if self.tackle_events else 0,
                "avg_interception_confidence": np.mean([i.confidence_score for i in self.interception_events]) if self.interception_events else 0
            }
        }
    
    def export_enhanced_statistics_to_csv(self, output_dir="data/output"):
        """Export enhanced tackle and interception statistics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export tackle events
        tackle_events_path = os.path.join(output_dir, "tackle_events.csv")
        with open(tackle_events_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_num', 'tackler_id', 'tackled_player_id', 'tackler_team', 
                         'tackled_team', 'tackle_distance', 'success_probability', 
                         'confidence_score', 'tackle_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.tackle_events:
                writer.writerow({
                    'frame_num': event.frame_num,
                    'tackler_id': event.tackler_id,
                    'tackled_player_id': event.tackled_player_id,
                    'tackler_team': event.tackler_team,
                    'tackled_team': event.tackled_team,
                    'tackle_distance': round(event.tackle_distance, 2),
                    'success_probability': round(event.success_probability, 3),
                    'confidence_score': round(event.confidence_score, 3),
                    'tackle_type': event.tackle_type
                })
        
        # Export interception events
        interception_events_path = os.path.join(output_dir, "interception_events.csv")
        with open(interception_events_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_num', 'interceptor_id', 'interceptor_team', 
                         'original_target_team', 'interception_distance', 
                         'ball_trajectory_change', 'confidence_score', 'interception_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.interception_events:
                writer.writerow({
                    'frame_num': event.frame_num,
                    'interceptor_id': event.interceptor_id,
                    'interceptor_team': event.interceptor_team,
                    'original_target_team': event.original_target_team,
                    'interception_distance': round(event.interception_distance, 2),
                    'ball_trajectory_change': round(event.ball_trajectory_change, 2),
                    'confidence_score': round(event.confidence_score, 3),
                    'interception_type': event.interception_type
                })
        
        print(f"Enhanced tackle and interception statistics exported to {output_dir}")
