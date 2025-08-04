import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from src.utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    # For testing, try relative import
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils import get_center_of_bbox, get_foot_position, measure_distance


@dataclass
class ChallengeEvent:
    """Represents a detected challenge event with comprehensive data."""

    frame_num: int
    challenger_id: int
    challenged_player_id: int
    challenger_team: int
    challenged_team: int
    challenge_distance: float
    ball_distance_to_challenge: float
    challenge_type: str  # "ground", "aerial", "pressing", "sliding"
    success: bool  # True if challenger won the ball
    confidence_score: float
    ball_velocity_before: float
    ball_velocity_after: float
    challenge_duration_frames: int = 1


@dataclass
class PlayerChallengeStats:
    """Statistics for a player's challenge performance."""

    player_id: int
    team_id: int
    challenges_attempted: int = 0
    challenges_successful: int = 0
    challenges_failed: int = 0
    challenge_success_rate: float = 0.0
    avg_challenge_distance: float = 0.0
    ground_challenges: int = 0
    aerial_challenges: int = 0
    pressing_challenges: int = 0
    sliding_challenges: int = 0


class ChallengeDetector:
    """
    Advanced challenge detection system that identifies defensive actions
    where players attempt to win the ball from opponents.

    This detector identifies various types of challenges:
    - Ground challenges: Direct contests for the ball on the ground
    - Aerial challenges: Headers and aerial duels
    - Pressing challenges: Applying pressure to force errors
    - Sliding challenges: Slide tackles and similar actions
    """

    def __init__(self, frame_rate: float = 24.0):
        """
        Initialize the Challenge Detector.

        Args:
            frame_rate (float): Video frame rate for temporal analysis
        """
        # Challenge events tracking
        self.challenge_events: List[ChallengeEvent] = []
        self.active_challenges: Dict[Tuple[int, int], Dict] = (
            {}
        )  # (challenger, challenged) -> challenge_data

        # Statistics tracking
        self.team_challenges = {1: 0, 2: 0}
        self.team_successful_challenges = {1: 0, 2: 0}
        self.player_challenge_stats: Dict[int, PlayerChallengeStats] = {}

        # Detection parameters
        self.challenge_distance_threshold = (
            80  # pixels - maximum distance for challenge
        )
        self.ball_proximity_threshold = (
            120  # pixels - ball must be within this distance
        )
        self.min_challenge_duration = 1  # minimum frames for a challenge
        self.max_challenge_duration = 30  # maximum frames for a challenge
        self.confidence_threshold = 0.4  # minimum confidence for challenge validation

        # Movement analysis
        self.frame_rate = frame_rate
        self.prev_player_positions = {}
        self.player_velocities = {}

        # Possession tracking for success determination
        self.last_ball_possessor = -1
        self.last_team_possession = None
        self.possession_change_cooldown = 0

    def detect_challenge(
        self,
        current_frame_players: Dict,
        ball_possessor: int,
        ball_team: int,
        frame_num: int,
        ball_data: Dict = None,
    ) -> Optional[ChallengeEvent]:
        """
        Detect challenge events in the current frame.

        Args:
            current_frame_players (Dict): Dictionary of player data for current frame
            ball_possessor (int): ID of player currently possessing the ball
            ball_team (int): Team of player currently possessing the ball
            frame_num (int): Current frame number
            ball_data (Dict): Ball tracking data for current frame

        Returns:
            ChallengeEvent if challenge is detected, None otherwise
        """
        # Skip if in cooldown period
        if self.possession_change_cooldown > 0:
            self.possession_change_cooldown -= 1

        # Update player tracking
        self._update_player_tracking(current_frame_players, frame_num)

        # Check for challenge completion (possession changes) FIRST
        completed_challenge = self._check_challenge_completion(
            ball_possessor, ball_team, frame_num, ball_data
        )

        # Update active challenges
        self._update_active_challenges(current_frame_players, frame_num, ball_data)

        # Detect new challenges (only if no challenge was just completed)
        new_challenge = None
        if not completed_challenge:
            new_challenge = self._detect_new_challenges(
                current_frame_players, ball_possessor, ball_team, frame_num, ball_data
            )

        # Update possession tracking
        self.last_ball_possessor = ball_possessor
        self.last_team_possession = ball_team

        return completed_challenge or new_challenge

    def _update_player_tracking(self, current_frame_players: Dict, frame_num: int):
        """Update player position and velocity tracking."""
        for player_id, player_data in current_frame_players.items():
            current_pos = get_foot_position(player_data["bbox"])

            # Calculate velocity if we have previous position
            if player_id in self.prev_player_positions:
                prev_pos = self.prev_player_positions[player_id]
                distance = measure_distance(current_pos, prev_pos)
                velocity = distance * self.frame_rate / 1000.0  # Convert to m/s
                self.player_velocities[player_id] = velocity
            else:
                self.player_velocities[player_id] = 0.0

            self.prev_player_positions[player_id] = current_pos

    def _detect_new_challenges(
        self,
        current_frame_players: Dict,
        ball_possessor: int,
        ball_team: int,
        frame_num: int,
        ball_data: Dict = None,
    ) -> Optional[ChallengeEvent]:
        """Detect new challenge events based on player proximity and ball involvement."""
        if ball_possessor == -1 or ball_team is None:
            return None

        ball_position = ball_data.get("position") if ball_data else None
        if ball_position is None:
            return None

        # Look for players from opposing team near the ball possessor
        possessor_pos = get_foot_position(current_frame_players[ball_possessor]["bbox"])

        for player_id, player_data in current_frame_players.items():
            if player_id == ball_possessor:
                continue

            player_team = player_data.get("team")
            if player_team == ball_team or player_team is None:
                continue  # Same team or unknown team

            # Check if this player is challenging
            challenger_pos = get_foot_position(player_data["bbox"])
            challenge_distance = measure_distance(challenger_pos, possessor_pos)
            ball_distance = measure_distance(challenger_pos, ball_position)

            # Check challenge criteria
            if (
                challenge_distance <= self.challenge_distance_threshold
                and ball_distance <= self.ball_proximity_threshold
            ):

                # Check if this is a new challenge (not already active)
                challenge_key = (player_id, ball_possessor)
                if challenge_key not in self.active_challenges:

                    # Determine challenge type
                    challenge_type = self._classify_challenge_type(
                        challenger_pos,
                        possessor_pos,
                        challenge_distance,
                        self.player_velocities.get(player_id, 0.0),
                    )

                    # Calculate confidence score
                    confidence = self._calculate_challenge_confidence(
                        challenge_distance, ball_distance, challenge_type, ball_data
                    )

                    if confidence >= self.confidence_threshold:
                        # Start tracking this challenge
                        self.active_challenges[challenge_key] = {
                            "start_frame": frame_num,
                            "challenger_id": player_id,
                            "challenged_id": ball_possessor,
                            "challenger_team": player_team,
                            "challenged_team": ball_team,
                            "challenge_type": challenge_type,
                            "confidence": confidence,
                            "initial_distance": challenge_distance,
                            "ball_distance": ball_distance,
                        }

                        print(
                            f"🥊 Challenge started: Player {player_id} (Team {player_team}) "
                            f"challenging Player {ball_possessor} (Team {ball_team}) "
                            f"at frame {frame_num} - Type: {challenge_type}"
                        )

        return None

    def _check_challenge_completion(
        self,
        ball_possessor: int,
        ball_team: int,
        frame_num: int,
        ball_data: Dict = None,
    ) -> Optional[ChallengeEvent]:
        """Check if any active challenges have been completed due to possession changes."""
        if (
            self.last_ball_possessor == -1
            or self.last_team_possession is None
            or ball_possessor == self.last_ball_possessor
        ):
            return None

        completed_challenge = None
        challenges_to_remove = []

        # Check all active challenges for completion
        for challenge_key, challenge_data in self.active_challenges.items():
            challenger_id = challenge_data["challenger_id"]
            challenged_id = challenge_data["challenged_id"]

            # Check if the challenge involved the possession change
            if challenged_id == self.last_ball_possessor:
                # Determine if challenge was successful
                success = (
                    ball_possessor == challenger_id
                    or ball_team == challenge_data["challenger_team"]
                )

                # Create completed challenge event
                duration = frame_num - challenge_data["start_frame"]
                ball_velocity_before = 0.0
                ball_velocity_after = 0.0

                if ball_data:
                    ball_velocity_after = ball_data.get("velocity", 0.0)

                challenge_event = ChallengeEvent(
                    frame_num=frame_num,
                    challenger_id=challenger_id,
                    challenged_player_id=challenged_id,
                    challenger_team=challenge_data["challenger_team"],
                    challenged_team=challenge_data["challenged_team"],
                    challenge_distance=challenge_data["initial_distance"],
                    ball_distance_to_challenge=challenge_data["ball_distance"],
                    challenge_type=challenge_data["challenge_type"],
                    success=success,
                    confidence_score=challenge_data["confidence"],
                    ball_velocity_before=ball_velocity_before,
                    ball_velocity_after=ball_velocity_after,
                    challenge_duration_frames=duration,
                )

                # Record the challenge
                self._record_challenge_event(challenge_event)
                completed_challenge = challenge_event
                challenges_to_remove.append(challenge_key)

                print(
                    f"🥊 Challenge completed: Player {challenger_id} "
                    f"{'WON' if success else 'LOST'} challenge against Player {challenged_id} "
                    f"at frame {frame_num}"
                )

        # Remove completed challenges
        for key in challenges_to_remove:
            del self.active_challenges[key]

        # Set cooldown to prevent multiple detections
        if completed_challenge:
            self.possession_change_cooldown = 5

        return completed_challenge

    def _update_active_challenges(
        self, current_frame_players: Dict, frame_num: int, ball_data: Dict = None
    ):
        """Update active challenges and remove expired ones."""
        challenges_to_remove = []

        for challenge_key, challenge_data in self.active_challenges.items():
            duration = frame_num - challenge_data["start_frame"]

            # Remove challenges that have lasted too long
            if duration > self.max_challenge_duration:
                challenges_to_remove.append(challenge_key)
                continue

            # Check if players are still in proximity
            challenger_id = challenge_data["challenger_id"]
            challenged_id = challenge_data["challenged_id"]

            if (
                challenger_id not in current_frame_players
                or challenged_id not in current_frame_players
            ):
                challenges_to_remove.append(challenge_key)
                continue

            # Update challenge data if needed
            challenger_pos = get_foot_position(
                current_frame_players[challenger_id]["bbox"]
            )
            challenged_pos = get_foot_position(
                current_frame_players[challenged_id]["bbox"]
            )
            current_distance = measure_distance(challenger_pos, challenged_pos)

            # If players are too far apart, end the challenge
            if current_distance > self.challenge_distance_threshold * 1.5:
                challenges_to_remove.append(challenge_key)

        # Remove expired challenges
        for key in challenges_to_remove:
            del self.active_challenges[key]

    def _classify_challenge_type(
        self,
        challenger_pos: Tuple[int, int],
        challenged_pos: Tuple[int, int],
        distance: float,
        challenger_velocity: float,
    ) -> str:
        """Classify the type of challenge based on positions and movement."""
        # Simple classification based on distance and velocity
        if distance < 30:
            if challenger_velocity > 3.0:
                return "sliding"
            else:
                return "ground"
        elif distance < 60:
            if challenger_velocity > 2.0:
                return "pressing"
            else:
                return "ground"
        else:
            # Check if this might be an aerial challenge (simplified)
            if challenger_velocity > 1.5:
                return "aerial"
            else:
                return "pressing"

    def _calculate_challenge_confidence(
        self,
        challenge_distance: float,
        ball_distance: float,
        challenge_type: str,
        ball_data: Dict = None,
    ) -> float:
        """Calculate confidence score for a challenge detection."""
        confidence = 0.5  # Base confidence

        # Distance-based confidence (closer = higher confidence)
        distance_factor = max(
            0, 1.0 - (challenge_distance / self.challenge_distance_threshold)
        )
        confidence += distance_factor * 0.3

        # Ball proximity factor
        ball_factor = max(0, 1.0 - (ball_distance / self.ball_proximity_threshold))
        confidence += ball_factor * 0.2

        # Ball detection confidence
        if ball_data and "confidence" in ball_data:
            ball_conf = ball_data["confidence"]
            confidence += ball_conf * 0.1

        # Challenge type factor
        type_factors = {
            "ground": 0.1,
            "sliding": 0.15,
            "pressing": 0.05,
            "aerial": 0.08,
        }
        confidence += type_factors.get(challenge_type, 0.05)

        return min(1.0, confidence)

    def _record_challenge_event(self, challenge_event: ChallengeEvent):
        """Record a challenge event and update statistics."""
        self.challenge_events.append(challenge_event)

        # Update team statistics
        challenger_team = challenge_event.challenger_team
        self.team_challenges[challenger_team] = (
            self.team_challenges.get(challenger_team, 0) + 1
        )

        if challenge_event.success:
            self.team_successful_challenges[challenger_team] = (
                self.team_successful_challenges.get(challenger_team, 0) + 1
            )

        # Update player statistics
        challenger_id = challenge_event.challenger_id
        if challenger_id not in self.player_challenge_stats:
            self.player_challenge_stats[challenger_id] = PlayerChallengeStats(
                player_id=challenger_id, team_id=challenger_team
            )

        stats = self.player_challenge_stats[challenger_id]
        stats.challenges_attempted += 1

        if challenge_event.success:
            stats.challenges_successful += 1
        else:
            stats.challenges_failed += 1

        # Update success rate
        stats.challenge_success_rate = (
            stats.challenges_successful / stats.challenges_attempted * 100
        )

        # Update challenge type counts
        challenge_type = challenge_event.challenge_type
        if challenge_type == "ground":
            stats.ground_challenges += 1
        elif challenge_type == "aerial":
            stats.aerial_challenges += 1
        elif challenge_type == "pressing":
            stats.pressing_challenges += 1
        elif challenge_type == "sliding":
            stats.sliding_challenges += 1

        # Update average distance
        total_distance = stats.avg_challenge_distance * (stats.challenges_attempted - 1)
        stats.avg_challenge_distance = (
            total_distance + challenge_event.challenge_distance
        ) / stats.challenges_attempted

    def get_team_challenge_statistics(self) -> Dict:
        """Get comprehensive team challenge statistics."""
        team_stats = {}

        for team_id in [1, 2]:
            attempted = self.team_challenges.get(team_id, 0)
            successful = self.team_successful_challenges.get(team_id, 0)
            failed = attempted - successful
            success_rate = (successful / attempted * 100) if attempted > 0 else 0.0

            team_stats[team_id] = {
                "challenges_attempted": attempted,
                "challenges_successful": successful,
                "challenges_failed": failed,
                "challenge_success_rate": success_rate,
            }

        return team_stats

    def get_player_challenge_statistics(self) -> Dict[int, PlayerChallengeStats]:
        """Get comprehensive player challenge statistics."""
        return self.player_challenge_stats.copy()

    def get_challenge_events(self) -> List[ChallengeEvent]:
        """Get all recorded challenge events."""
        return self.challenge_events.copy()

    def export_challenge_statistics_to_csv(self, output_dir: str = "data/output"):
        """Export challenge statistics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Export challenge events
        events_path = os.path.join(output_dir, "challenge_events.csv")
        with open(events_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_num",
                "challenger_id",
                "challenged_player_id",
                "challenger_team",
                "challenged_team",
                "challenge_distance",
                "ball_distance_to_challenge",
                "challenge_type",
                "success",
                "confidence_score",
                "ball_velocity_before",
                "ball_velocity_after",
                "challenge_duration_frames",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in self.challenge_events:
                writer.writerow(
                    {
                        "frame_num": event.frame_num,
                        "challenger_id": event.challenger_id,
                        "challenged_player_id": event.challenged_player_id,
                        "challenger_team": event.challenger_team,
                        "challenged_team": event.challenged_team,
                        "challenge_distance": round(event.challenge_distance, 2),
                        "ball_distance_to_challenge": round(
                            event.ball_distance_to_challenge, 2
                        ),
                        "challenge_type": event.challenge_type,
                        "success": event.success,
                        "confidence_score": round(event.confidence_score, 3),
                        "ball_velocity_before": round(event.ball_velocity_before, 2),
                        "ball_velocity_after": round(event.ball_velocity_after, 2),
                        "challenge_duration_frames": event.challenge_duration_frames,
                    }
                )

        # Export player statistics
        player_stats_path = os.path.join(output_dir, "player_challenge_stats.csv")
        with open(player_stats_path, "w", newline="") as csvfile:
            fieldnames = [
                "player_id",
                "team_id",
                "challenges_attempted",
                "challenges_successful",
                "challenges_failed",
                "challenge_success_rate",
                "avg_challenge_distance",
                "ground_challenges",
                "aerial_challenges",
                "pressing_challenges",
                "sliding_challenges",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for player_id, stats in self.player_challenge_stats.items():
                writer.writerow(
                    {
                        "player_id": stats.player_id,
                        "team_id": stats.team_id,
                        "challenges_attempted": stats.challenges_attempted,
                        "challenges_successful": stats.challenges_successful,
                        "challenges_failed": stats.challenges_failed,
                        "challenge_success_rate": round(
                            stats.challenge_success_rate, 2
                        ),
                        "avg_challenge_distance": round(
                            stats.avg_challenge_distance, 2
                        ),
                        "ground_challenges": stats.ground_challenges,
                        "aerial_challenges": stats.aerial_challenges,
                        "pressing_challenges": stats.pressing_challenges,
                        "sliding_challenges": stats.sliding_challenges,
                    }
                )

        print(f"📊 Challenge statistics exported:")
        print(f"   Events: {events_path}")
        print(f"   Player stats: {player_stats_path}")

        return events_path, player_stats_path
