import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
import os

from .ball_movement_analyzer import EnhancedBallMovementAnalyzer
from .enhanced_pass_counter import EnhancedPassCounter, PassEvent, PossessionEvent
from .enhanced_tackle_counter import EnhancedTackleCounter, TackleEvent, InterceptionEvent


@dataclass
class TeamStatistics:
    """Comprehensive team statistics."""
    team_id: int
    
    # Possession statistics
    total_possession_time: float
    possession_percentage: float
    avg_possession_duration: float
    possession_count: int
    
    # Passing statistics
    total_passes: int
    successful_passes: int
    pass_accuracy: float
    avg_pass_distance: float
    short_passes: int
    medium_passes: int
    long_passes: int
    
    # Defensive statistics
    total_tackles: int
    successful_tackles: int
    tackle_success_rate: float
    total_interceptions: int
    defensive_actions: int
    
    # Ball control statistics
    ball_touches: int
    avg_ball_control_duration: float
    ball_control_quality: float
    
    # Advanced metrics
    pass_completion_rate: float
    defensive_efficiency: float
    ball_retention_rate: float


@dataclass
class PlayerStatistics:
    """Comprehensive player statistics."""
    player_id: int
    team_id: int
    
    # Passing statistics
    passes_attempted: int
    passes_completed: int
    pass_accuracy: float
    avg_pass_distance: float
    key_passes: int
    
    # Defensive statistics
    tackles_attempted: int
    tackles_successful: int
    tackle_success_rate: float
    interceptions: int
    defensive_actions: int
    
    # Possession statistics
    ball_touches: int
    possession_time: float
    avg_possession_duration: float
    ball_control_quality: float
    
    # Movement statistics
    distance_covered: float
    avg_speed: float
    max_speed: float


class EnhancedFootballStatistics:
    """
    Comprehensive football statistics analyzer that combines all enhanced tracking data
    to provide detailed team and player performance metrics.
    """
    
    def __init__(self, frame_rate: float = 24.0):
        """
        Initialize the Enhanced Football Statistics analyzer.
        
        Args:
            frame_rate (float): Video frame rate for temporal calculations
        """
        self.frame_rate = frame_rate
        
        # Component analyzers
        self.ball_analyzer = EnhancedBallMovementAnalyzer(frame_rate=frame_rate)
        self.pass_counter = EnhancedPassCounter(frame_rate=frame_rate)
        self.tackle_counter = EnhancedTackleCounter(frame_rate=frame_rate)
        
        # Statistics storage
        self.team_statistics: Dict[int, TeamStatistics] = {}
        self.player_statistics: Dict[int, PlayerStatistics] = {}
        
        # Raw data storage
        self.possession_data: List[PossessionEvent] = []
        self.pass_data: List[PassEvent] = []
        self.tackle_data: List[TackleEvent] = []
        self.interception_data: List[InterceptionEvent] = []
        
    def analyze_complete_match(self, tracks: Dict, player_assignments: List[int]) -> Dict:
        """
        Perform comprehensive analysis of a complete match.
        
        Args:
            tracks (Dict): Complete tracking data for the match
            player_assignments (List[int]): Player ball assignments per frame
            
        Returns:
            Dict: Comprehensive match statistics and analysis
        """
        print("🔍 Starting comprehensive match analysis...")
        
        # Analyze ball movement patterns
        print("📊 Analyzing ball movement patterns...")
        ball_analysis = self.ball_analyzer.analyze_ball_tracks(tracks["ball"])
        
        # Analyze passes
        print("⚽ Analyzing pass patterns...")
        pass_analysis = self.pass_counter.analyze_passes_from_tracks(tracks, player_assignments)
        
        # Analyze tackles and interceptions
        print("🛡️ Analyzing defensive actions...")
        tackle_analysis = self.tackle_counter.analyze_tackles_from_tracks(tracks, player_assignments)
        
        # Store raw data
        self.possession_data = pass_analysis.get("pass_events", [])
        self.pass_data = pass_analysis.get("pass_events", [])
        self.tackle_data = tackle_analysis.get("tackle_events", [])
        self.interception_data = tackle_analysis.get("interception_events", [])
        
        # Calculate comprehensive statistics
        print("📈 Calculating comprehensive statistics...")
        self._calculate_team_statistics(tracks, ball_analysis, pass_analysis, tackle_analysis)
        self._calculate_player_statistics(tracks, ball_analysis, pass_analysis, tackle_analysis)
        
        # Generate match summary
        match_summary = self._generate_match_summary(ball_analysis, pass_analysis, tackle_analysis)
        
        print("✅ Match analysis complete!")
        return match_summary
    
    def _calculate_team_statistics(self, tracks: Dict, ball_analysis: Dict, 
                                 pass_analysis: Dict, tackle_analysis: Dict):
        """Calculate comprehensive team statistics."""
        for team_id in [1, 2]:
            # Possession statistics
            possession_stats = pass_analysis.get("possession_statistics", {}).get(team_id, {})
            total_possession_time = possession_stats.get("total_possession_time", 0)
            possession_count = possession_stats.get("total_possessions", 0)
            
            # Calculate possession percentage
            total_frames = len(tracks.get("ball", []))
            possession_percentage = (total_possession_time / total_frames * 100) if total_frames > 0 else 0
            
            # Passing statistics
            team_pass_stats = pass_analysis.get("team_statistics", {}).get(team_id, {})
            total_passes = team_pass_stats.get("total_passes", 0)
            pass_accuracy = team_pass_stats.get("pass_accuracy", 0)
            
            # Defensive statistics
            team_tackle_stats = tackle_analysis.get("team_statistics", {}).get(team_id, {})
            total_tackles = team_tackle_stats.get("total_tackles", 0)
            total_interceptions = team_tackle_stats.get("total_interceptions", 0)
            
            # Create team statistics object
            self.team_statistics[team_id] = TeamStatistics(
                team_id=team_id,
                total_possession_time=total_possession_time,
                possession_percentage=possession_percentage,
                avg_possession_duration=possession_stats.get("avg_possession_duration", 0),
                possession_count=possession_count,
                total_passes=total_passes,
                successful_passes=int(total_passes * pass_accuracy) if pass_accuracy > 0 else 0,
                pass_accuracy=pass_accuracy,
                avg_pass_distance=team_pass_stats.get("avg_pass_distance", 0),
                short_passes=team_pass_stats.get("short_passes", 0),
                medium_passes=team_pass_stats.get("medium_passes", 0),
                long_passes=team_pass_stats.get("long_passes", 0),
                total_tackles=total_tackles,
                successful_tackles=int(total_tackles * team_tackle_stats.get("avg_tackle_success", 0)),
                tackle_success_rate=team_tackle_stats.get("avg_tackle_success", 0),
                total_interceptions=total_interceptions,
                defensive_actions=total_tackles + total_interceptions,
                ball_touches=self._calculate_team_ball_touches(team_id),
                avg_ball_control_duration=possession_stats.get("avg_possession_duration", 0),
                ball_control_quality=possession_stats.get("avg_possession_quality", 0),
                pass_completion_rate=pass_accuracy,
                defensive_efficiency=self._calculate_defensive_efficiency(team_id, tackle_analysis),
                ball_retention_rate=self._calculate_ball_retention_rate(team_id, pass_analysis)
            )
    
    def _calculate_player_statistics(self, tracks: Dict, ball_analysis: Dict, 
                                   pass_analysis: Dict, tackle_analysis: Dict):
        """Calculate comprehensive player statistics."""
        # Get all unique players from tracks
        all_players = set()
        for frame_players in tracks.get("players", []):
            all_players.update(frame_players.keys())
        
        for player_id in all_players:
            # Get player team
            player_team = self._get_player_team(player_id, tracks)
            if player_team is None:
                continue
            
            # Passing statistics
            player_pass_stats = pass_analysis.get("player_statistics", {}).get(player_id, {})
            passes_attempted = player_pass_stats.get("passes", 0)
            pass_accuracy = player_pass_stats.get("pass_accuracy", 0)
            
            # Defensive statistics
            player_tackle_stats = tackle_analysis.get("player_tackles", {}).get(player_id, {})
            player_interception_stats = tackle_analysis.get("player_interceptions", {}).get(player_id, {})
            
            tackles_attempted = player_tackle_stats.get("tackles", 0)
            tackle_success_rate = player_tackle_stats.get("tackle_success_rate", 0)
            interceptions = player_interception_stats.get("interceptions", 0)
            
            # Calculate movement statistics from tracks
            movement_stats = self._calculate_player_movement_stats(player_id, tracks)
            
            # Create player statistics object
            self.player_statistics[player_id] = PlayerStatistics(
                player_id=player_id,
                team_id=player_team,
                passes_attempted=passes_attempted,
                passes_completed=int(passes_attempted * pass_accuracy) if pass_accuracy > 0 else 0,
                pass_accuracy=pass_accuracy,
                avg_pass_distance=self._calculate_player_avg_pass_distance(player_id),
                key_passes=self._calculate_key_passes(player_id),
                tackles_attempted=tackles_attempted,
                tackles_successful=int(tackles_attempted * tackle_success_rate),
                tackle_success_rate=tackle_success_rate,
                interceptions=interceptions,
                defensive_actions=tackles_attempted + interceptions,
                ball_touches=self._calculate_player_ball_touches(player_id),
                possession_time=self._calculate_player_possession_time(player_id),
                avg_possession_duration=self._calculate_player_avg_possession_duration(player_id),
                ball_control_quality=self._calculate_player_ball_control_quality(player_id),
                distance_covered=movement_stats.get("distance_covered", 0),
                avg_speed=movement_stats.get("avg_speed", 0),
                max_speed=movement_stats.get("max_speed", 0)
            )
    
    def _get_player_team(self, player_id: int, tracks: Dict) -> Optional[int]:
        """Get the team ID for a player."""
        for frame_players in tracks.get("players", []):
            if player_id in frame_players and "team" in frame_players[player_id]:
                return frame_players[player_id]["team"]
        return None
    
    def _calculate_player_movement_stats(self, player_id: int, tracks: Dict) -> Dict:
        """Calculate movement statistics for a player."""
        distances = []
        speeds = []
        
        for frame_players in tracks.get("players", []):
            if player_id in frame_players:
                player_data = frame_players[player_id]
                if "distance" in player_data:
                    distances.append(player_data["distance"])
                if "speed" in player_data:
                    speeds.append(player_data["speed"])
        
        return {
            "distance_covered": sum(distances),
            "avg_speed": np.mean(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0
        }
    
    def _calculate_team_ball_touches(self, team_id: int) -> int:
        """Calculate total ball touches for a team."""
        touches = 0
        for possession in self.possession_data:
            if hasattr(possession, 'team_id') and possession.team_id == team_id:
                touches += getattr(possession, 'ball_touches', 1)
        return touches
    
    def _calculate_defensive_efficiency(self, team_id: int, tackle_analysis: Dict) -> float:
        """Calculate defensive efficiency for a team."""
        team_stats = tackle_analysis.get("team_statistics", {}).get(team_id, {})
        total_defensive_actions = team_stats.get("defensive_actions", 0)
        if total_defensive_actions == 0:
            return 0.0
        
        successful_actions = (
            team_stats.get("total_tackles", 0) * team_stats.get("avg_tackle_success", 0) +
            team_stats.get("total_interceptions", 0)
        )
        
        return successful_actions / total_defensive_actions if total_defensive_actions > 0 else 0.0
    
    def _calculate_ball_retention_rate(self, team_id: int, pass_analysis: Dict) -> float:
        """Calculate ball retention rate for a team."""
        team_stats = pass_analysis.get("team_statistics", {}).get(team_id, {})
        return team_stats.get("pass_accuracy", 0.0)
    
    def _calculate_player_ball_touches(self, player_id: int) -> int:
        """Calculate ball touches for a player."""
        touches = 0
        for possession in self.possession_data:
            if hasattr(possession, 'player_id') and possession.player_id == player_id:
                touches += getattr(possession, 'ball_touches', 1)
        return touches
    
    def _calculate_player_possession_time(self, player_id: int) -> float:
        """Calculate total possession time for a player."""
        total_time = 0
        for possession in self.possession_data:
            if hasattr(possession, 'player_id') and possession.player_id == player_id:
                total_time += getattr(possession, 'duration_frames', 0)
        return total_time / self.frame_rate  # Convert to seconds
    
    def _calculate_player_avg_possession_duration(self, player_id: int) -> float:
        """Calculate average possession duration for a player."""
        possessions = [p for p in self.possession_data 
                      if hasattr(p, 'player_id') and p.player_id == player_id]
        if not possessions:
            return 0.0
        
        total_duration = sum(getattr(p, 'duration_frames', 0) for p in possessions)
        return (total_duration / len(possessions)) / self.frame_rate
    
    def _calculate_player_ball_control_quality(self, player_id: int) -> float:
        """Calculate ball control quality for a player."""
        possessions = [p for p in self.possession_data 
                      if hasattr(p, 'player_id') and p.player_id == player_id]
        if not possessions:
            return 0.0
        
        qualities = [getattr(p, 'possession_quality', 0.5) for p in possessions]
        return np.mean(qualities)
    
    def _calculate_player_avg_pass_distance(self, player_id: int) -> float:
        """Calculate average pass distance for a player."""
        player_passes = [p for p in self.pass_data if p.passer_id == player_id]
        if not player_passes:
            return 0.0
        
        distances = [p.pass_distance for p in player_passes]
        return np.mean(distances)
    
    def _calculate_key_passes(self, player_id: int) -> int:
        """Calculate key passes for a player (simplified)."""
        # This is a simplified calculation - in a full implementation,
        # key passes would be determined by analyzing if the pass led to a shot or goal
        player_passes = [p for p in self.pass_data if p.passer_id == player_id]
        # Consider long passes and high-velocity passes as potential key passes
        key_passes = [p for p in player_passes 
                     if p.pass_distance > 60 or p.pass_velocity > 15]
        return len(key_passes)
    
    def _generate_match_summary(self, ball_analysis: Dict, pass_analysis: Dict, 
                              tackle_analysis: Dict) -> Dict:
        """Generate comprehensive match summary."""
        return {
            "match_overview": {
                "total_frames_analyzed": ball_analysis.get("total_frames", 0),
                "ball_tracking_quality": ball_analysis.get("quality_metrics", {}),
                "total_passes": sum(stats.total_passes for stats in self.team_statistics.values()),
                "total_tackles": sum(stats.total_tackles for stats in self.team_statistics.values()),
                "total_interceptions": sum(stats.total_interceptions for stats in self.team_statistics.values())
            },
            "team_statistics": {team_id: stats for team_id, stats in self.team_statistics.items()},
            "player_statistics": {player_id: stats for player_id, stats in self.player_statistics.items()},
            "ball_analysis": ball_analysis,
            "pass_analysis": pass_analysis,
            "tackle_analysis": tackle_analysis,
            "advanced_metrics": self._calculate_advanced_metrics()
        }
    
    def _calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced match metrics."""
        if not self.team_statistics:
            return {}
        
        team1_stats = self.team_statistics.get(1)
        team2_stats = self.team_statistics.get(2)
        
        if not team1_stats or not team2_stats:
            return {}
        
        return {
            "possession_dominance": abs(team1_stats.possession_percentage - team2_stats.possession_percentage),
            "passing_efficiency_difference": abs(team1_stats.pass_accuracy - team2_stats.pass_accuracy),
            "defensive_pressure": {
                1: team2_stats.defensive_actions,
                2: team1_stats.defensive_actions
            },
            "ball_control_quality": {
                1: team1_stats.ball_control_quality,
                2: team2_stats.ball_control_quality
            },
            "match_intensity": (
                sum(stats.defensive_actions for stats in self.team_statistics.values()) /
                max(1, ball_analysis.get("total_frames", 1)) * 100
            )
        }
    
    def export_comprehensive_statistics(self, output_dir: str = "data/output"):
        """Export all comprehensive statistics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export team statistics
        team_stats_path = os.path.join(output_dir, "comprehensive_team_stats.csv")
        with open(team_stats_path, 'w', newline='') as csvfile:
            fieldnames = [
                'team_id', 'possession_percentage', 'total_passes', 'pass_accuracy',
                'total_tackles', 'tackle_success_rate', 'total_interceptions',
                'defensive_efficiency', 'ball_retention_rate', 'avg_possession_duration'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for team_id, stats in self.team_statistics.items():
                writer.writerow({
                    'team_id': team_id,
                    'possession_percentage': round(stats.possession_percentage, 2),
                    'total_passes': stats.total_passes,
                    'pass_accuracy': round(stats.pass_accuracy, 3),
                    'total_tackles': stats.total_tackles,
                    'tackle_success_rate': round(stats.tackle_success_rate, 3),
                    'total_interceptions': stats.total_interceptions,
                    'defensive_efficiency': round(stats.defensive_efficiency, 3),
                    'ball_retention_rate': round(stats.ball_retention_rate, 3),
                    'avg_possession_duration': round(stats.avg_possession_duration, 2)
                })
        
        # Export player statistics
        player_stats_path = os.path.join(output_dir, "comprehensive_player_stats.csv")
        with open(player_stats_path, 'w', newline='') as csvfile:
            fieldnames = [
                'player_id', 'team_id', 'passes_attempted', 'pass_accuracy',
                'tackles_attempted', 'tackle_success_rate', 'interceptions',
                'ball_touches', 'possession_time', 'distance_covered', 'avg_speed'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for player_id, stats in self.player_statistics.items():
                writer.writerow({
                    'player_id': player_id,
                    'team_id': stats.team_id,
                    'passes_attempted': stats.passes_attempted,
                    'pass_accuracy': round(stats.pass_accuracy, 3),
                    'tackles_attempted': stats.tackles_attempted,
                    'tackle_success_rate': round(stats.tackle_success_rate, 3),
                    'interceptions': stats.interceptions,
                    'ball_touches': stats.ball_touches,
                    'possession_time': round(stats.possession_time, 2),
                    'distance_covered': round(stats.distance_covered, 2),
                    'avg_speed': round(stats.avg_speed, 2)
                })
        
        print(f"Comprehensive statistics exported to {output_dir}")
        
        # Also export individual component statistics
        self.pass_counter.export_enhanced_statistics_to_csv(output_dir)
        self.tackle_counter.export_enhanced_statistics_to_csv(output_dir)
