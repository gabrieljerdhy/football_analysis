import csv
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .ball_movement_analyzer import EnhancedBallMovementAnalyzer, BallTrajectorySegment
from src.utils import measure_distance


@dataclass
class PassEvent:
    """Represents a detected pass event with enhanced validation."""
    frame_num: int
    passer_id: int
    receiver_id: int
    passer_team: int
    receiver_team: int
    pass_distance: float
    pass_velocity: float
    pass_accuracy: float
    confidence_score: float
    trajectory_segment: Optional[BallTrajectorySegment] = None
    pass_type: str = "short"  # short, medium, long, through_ball, cross


@dataclass
class PossessionEvent:
    """Represents a ball possession event."""
    start_frame: int
    end_frame: int
    player_id: int
    team_id: int
    duration_frames: int
    ball_touches: int
    possession_quality: float


class EnhancedPassCounter:
    """
    Enhanced pass counter that uses ball trajectory analysis and confidence-based validation
    to improve pass detection accuracy and provide detailed pass statistics.
    """
    
    def __init__(self, video_width=None, video_height=None, frame_rate=24.0):
        """
        Initialize the Enhanced Pass Counter.
        
        Args:
            video_width (int): Width of the video in pixels
            video_height (int): Height of the video in pixels
            frame_rate (float): Video frame rate for temporal analysis
        """
        # Basic statistics tracking
        self.team_passes = {1: 0, 2: 0}
        self.player_passes = {}
        self.team_goals = {1: 0, 2: 0}
        self.player_goals = {}
        
        # Enhanced tracking
        self.pass_events: List[PassEvent] = []
        self.possession_events: List[PossessionEvent] = []
        self.pass_counts_per_frame = []
        
        # Ball movement analyzer
        self.ball_analyzer = EnhancedBallMovementAnalyzer(frame_rate=frame_rate)
        
        # Possession tracking
        self.current_possession = None
        self.last_player_id = -1
        self.last_team = None
        self.possession_start_frame = -1
        self.possession_frames = 0
        self.ball_touches = 0
        
        # Pass detection parameters
        self.min_pass_distance = 5.0  # pixels - minimum distance for a pass
        self.max_pass_distance = 200.0  # pixels - maximum distance for a pass
        self.min_pass_velocity = 2.0  # m/s - minimum velocity for a pass
        self.min_possession_frames = 5  # minimum frames for stable possession
        self.pass_confidence_threshold = 0.4  # minimum confidence for pass validation
        
        # Pass classification thresholds
        self.short_pass_distance = 30.0  # pixels
        self.medium_pass_distance = 60.0  # pixels
        self.long_pass_distance = 100.0  # pixels
        
        # Video dimensions for field analysis
        self.video_width = video_width
        self.video_height = video_height
        
    def analyze_passes_from_tracks(self, tracks: Dict, player_assignments: List[int]) -> Dict:
        """
        Analyze passes from complete tracking data using enhanced ball movement analysis.
        
        Args:
            tracks (Dict): Complete tracking data including ball tracks
            player_assignments (List[int]): Player assignments per frame
            
        Returns:
            Dict: Comprehensive pass analysis results
        """
        # Analyze ball movement patterns
        ball_analysis = self.ball_analyzer.analyze_ball_tracks(tracks["ball"])
        
        # Process possession changes and detect passes
        self._process_possession_changes(tracks, player_assignments, ball_analysis)
        
        # Validate and classify passes
        self._validate_and_classify_passes(ball_analysis)
        
        # Generate comprehensive statistics
        return self._generate_pass_statistics()
    
    def count_passes_enhanced(self, current_player_id: int, current_team: int, 
                            frame_num: int, ball_data: Dict = None) -> int:
        """
        Enhanced pass counting with trajectory analysis and confidence validation.
        
        Args:
            current_player_id (int): ID of player currently possessing the ball
            current_team (int): Team of player currently possessing the ball
            frame_num (int): Current frame number
            ball_data (Dict): Enhanced ball tracking data for current frame
            
        Returns:
            int: Current pass count for the team
        """
        # Handle no possession
        if current_player_id == -1 or current_team is None:
            self._handle_no_possession(frame_num)
            return self.team_passes.get(current_team or 1, 0)
        
        # Convert to int if needed
        current_team = int(current_team)
        
        # Initialize possession tracking if needed
        if self.last_player_id == -1:
            self._start_new_possession(current_player_id, current_team, frame_num)
            return self.team_passes.get(current_team, 0)
        
        # Check for possession change
        if current_player_id != self.last_player_id:
            self._handle_possession_change(
                current_player_id, current_team, frame_num, ball_data
            )
        else:
            # Same player continues possession
            self._continue_possession(frame_num)
        
        # Store current pass counts for this frame
        self.pass_counts_per_frame.append({
            1: self.team_passes.get(1, 0), 
            2: self.team_passes.get(2, 0)
        })
        
        return self.team_passes.get(current_team, 0)
    
    def _process_possession_changes(self, tracks: Dict, player_assignments: List[int], 
                                  ball_analysis: Dict):
        """Process possession changes throughout the video."""
        trajectory_segments = ball_analysis.get("trajectory_data", [])
        
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
            
            # Process with enhanced counting
            self.count_passes_enhanced(assigned_player, current_team, frame_num, ball_data)
    
    def _handle_no_possession(self, frame_num: int):
        """Handle frames where no player has possession."""
        if self.current_possession:
            # End current possession
            self._end_possession(frame_num - 1)
        
        self.possession_frames = 0
        self.ball_touches = 0
    
    def _start_new_possession(self, player_id: int, team_id: int, frame_num: int):
        """Start tracking a new possession."""
        self.last_player_id = player_id
        self.last_team = team_id
        self.possession_start_frame = frame_num
        self.possession_frames = 1
        self.ball_touches = 1
        
        self.current_possession = PossessionEvent(
            start_frame=frame_num,
            end_frame=frame_num,
            player_id=player_id,
            team_id=team_id,
            duration_frames=1,
            ball_touches=1,
            possession_quality=1.0
        )
    
    def _continue_possession(self, frame_num: int):
        """Continue current possession."""
        self.possession_frames += 1
        if self.current_possession:
            self.current_possession.end_frame = frame_num
            self.current_possession.duration_frames = self.possession_frames
    
    def _handle_possession_change(self, new_player_id: int, new_team_id: int, 
                                frame_num: int, ball_data: Dict = None):
        """Handle possession change between players."""
        # End previous possession
        if self.current_possession:
            self._end_possession(frame_num - 1)
        
        # Check if this is a pass (same team) or turnover (different team)
        if new_team_id == self.last_team and self.possession_frames >= self.min_possession_frames:
            # Potential pass detected
            self._detect_pass(new_player_id, new_team_id, frame_num, ball_data)
        
        # Start new possession
        self._start_new_possession(new_player_id, new_team_id, frame_num)
    
    def _detect_pass(self, receiver_id: int, team_id: int, frame_num: int, 
                    ball_data: Dict = None):
        """Detect and validate a pass event."""
        if self.last_player_id == -1:
            return
        
        # Calculate pass distance (basic validation)
        pass_distance = 0.0
        pass_velocity = 0.0
        confidence_score = 0.5
        
        if ball_data:
            confidence_score = ball_data.get("confidence", 0.5)
            
            # Get ball position for distance calculation
            if "position" in ball_data:
                # Would need previous position for distance calculation
                # This is simplified for now
                pass_distance = 50.0  # Placeholder
                pass_velocity = ball_data.get("velocity", 5.0)
        
        # Validate pass based on confidence and parameters
        if (confidence_score >= self.pass_confidence_threshold and
            self.min_pass_distance <= pass_distance <= self.max_pass_distance):
            
            # Create pass event
            pass_event = PassEvent(
                frame_num=frame_num,
                passer_id=self.last_player_id,
                receiver_id=receiver_id,
                passer_team=team_id,
                receiver_team=team_id,
                pass_distance=pass_distance,
                pass_velocity=pass_velocity,
                pass_accuracy=confidence_score,
                confidence_score=confidence_score,
                pass_type=self._classify_pass_type(pass_distance, pass_velocity)
            )
            
            self.pass_events.append(pass_event)
            
            # Update team and player statistics
            self.team_passes[team_id] = self.team_passes.get(team_id, 0) + 1
            
            if self.last_player_id not in self.player_passes:
                self.player_passes[self.last_player_id] = {
                    "passes": 0,
                    "team": team_id,
                    "successful_passes": 0,
                    "pass_accuracy": 0.0
                }
            
            self.player_passes[self.last_player_id]["passes"] += 1
            self.player_passes[self.last_player_id]["successful_passes"] += 1
            
            # Update pass accuracy
            player_stats = self.player_passes[self.last_player_id]
            player_stats["pass_accuracy"] = (
                player_stats["successful_passes"] / player_stats["passes"]
            )
            
            print(f"Enhanced pass detected: Player {self.last_player_id} -> {receiver_id} "
                  f"(Team {team_id}) at frame {frame_num}, confidence: {confidence_score:.2f}")
    
    def _classify_pass_type(self, distance: float, velocity: float) -> str:
        """Classify pass type based on distance and velocity."""
        if distance < self.short_pass_distance:
            return "short"
        elif distance < self.medium_pass_distance:
            return "medium"
        elif distance < self.long_pass_distance:
            return "long"
        else:
            return "long"
    
    def _end_possession(self, frame_num: int):
        """End current possession and store possession event."""
        if self.current_possession:
            self.current_possession.end_frame = frame_num
            self.current_possession.duration_frames = self.possession_frames
            self.current_possession.ball_touches = self.ball_touches
            
            # Calculate possession quality based on duration and touches
            self.current_possession.possession_quality = min(
                1.0, self.possession_frames / 30.0  # Normalize by 30 frames
            )
            
            self.possession_events.append(self.current_possession)
            self.current_possession = None
    
    def _validate_and_classify_passes(self, ball_analysis: Dict):
        """Validate and enhance classification of detected passes."""
        trajectory_segments = ball_analysis.get("trajectory_data", [])
        
        for pass_event in self.pass_events:
            # Find corresponding trajectory segment
            matching_segment = None
            for segment in trajectory_segments:
                if (segment.start_frame <= pass_event.frame_num <= segment.end_frame):
                    matching_segment = segment
                    break
            
            if matching_segment:
                pass_event.trajectory_segment = matching_segment
                pass_event.pass_velocity = matching_segment.avg_velocity
                pass_event.confidence_score = min(
                    pass_event.confidence_score + matching_segment.confidence_score * 0.3,
                    1.0
                )
    
    def _generate_pass_statistics(self) -> Dict:
        """Generate comprehensive pass statistics."""
        total_passes = len(self.pass_events)
        
        if total_passes == 0:
            return self._empty_pass_statistics()
        
        # Calculate team statistics
        team_stats = {}
        for team_id in [1, 2]:
            team_passes = [p for p in self.pass_events if p.passer_team == team_id]
            
            team_stats[team_id] = {
                "total_passes": len(team_passes),
                "avg_pass_distance": np.mean([p.pass_distance for p in team_passes]) if team_passes else 0,
                "avg_pass_velocity": np.mean([p.pass_velocity for p in team_passes]) if team_passes else 0,
                "pass_accuracy": np.mean([p.pass_accuracy for p in team_passes]) if team_passes else 0,
                "short_passes": len([p for p in team_passes if p.pass_type == "short"]),
                "medium_passes": len([p for p in team_passes if p.pass_type == "medium"]),
                "long_passes": len([p for p in team_passes if p.pass_type == "long"])
            }
        
        # Calculate possession statistics
        possession_stats = self._calculate_possession_statistics()
        
        return {
            "total_passes": total_passes,
            "pass_events": self.pass_events,
            "team_statistics": team_stats,
            "possession_statistics": possession_stats,
            "player_statistics": self.player_passes,
            "quality_metrics": {
                "avg_confidence": np.mean([p.confidence_score for p in self.pass_events]),
                "trajectory_validated_passes": len([p for p in self.pass_events if p.trajectory_segment])
            }
        }
    
    def _calculate_possession_statistics(self) -> Dict:
        """Calculate possession-related statistics."""
        if not self.possession_events:
            return {}
        
        team_possessions = {1: [], 2: []}
        for poss in self.possession_events:
            if poss.team_id in team_possessions:
                team_possessions[poss.team_id].append(poss)
        
        stats = {}
        for team_id, possessions in team_possessions.items():
            if possessions:
                total_duration = sum(p.duration_frames for p in possessions)
                stats[team_id] = {
                    "total_possessions": len(possessions),
                    "total_possession_time": total_duration,
                    "avg_possession_duration": total_duration / len(possessions),
                    "avg_possession_quality": np.mean([p.possession_quality for p in possessions])
                }
            else:
                stats[team_id] = {
                    "total_possessions": 0,
                    "total_possession_time": 0,
                    "avg_possession_duration": 0,
                    "avg_possession_quality": 0
                }
        
        return stats
    
    def _empty_pass_statistics(self) -> Dict:
        """Return empty statistics structure."""
        return {
            "total_passes": 0,
            "pass_events": [],
            "team_statistics": {1: {}, 2: {}},
            "possession_statistics": {},
            "player_statistics": {},
            "quality_metrics": {}
        }
    
    def export_enhanced_statistics_to_csv(self, output_dir="data/output"):
        """Export enhanced pass statistics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export pass events
        pass_events_path = os.path.join(output_dir, "pass_events.csv")
        with open(pass_events_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_num', 'passer_id', 'receiver_id', 'team', 'pass_distance', 
                         'pass_velocity', 'pass_accuracy', 'confidence_score', 'pass_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.pass_events:
                writer.writerow({
                    'frame_num': event.frame_num,
                    'passer_id': event.passer_id,
                    'receiver_id': event.receiver_id,
                    'team': event.passer_team,
                    'pass_distance': round(event.pass_distance, 2),
                    'pass_velocity': round(event.pass_velocity, 2),
                    'pass_accuracy': round(event.pass_accuracy, 3),
                    'confidence_score': round(event.confidence_score, 3),
                    'pass_type': event.pass_type
                })
        
        # Export possession events
        possession_events_path = os.path.join(output_dir, "possession_events.csv")
        with open(possession_events_path, 'w', newline='') as csvfile:
            fieldnames = ['start_frame', 'end_frame', 'player_id', 'team_id', 
                         'duration_frames', 'ball_touches', 'possession_quality']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.possession_events:
                writer.writerow({
                    'start_frame': event.start_frame,
                    'end_frame': event.end_frame,
                    'player_id': event.player_id,
                    'team_id': event.team_id,
                    'duration_frames': event.duration_frames,
                    'ball_touches': event.ball_touches,
                    'possession_quality': round(event.possession_quality, 3)
                })
        
        print(f"Enhanced pass statistics exported to {output_dir}")
