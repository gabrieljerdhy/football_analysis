import csv
import os

import numpy as np


class TackleCounter:
    def __init__(self):
        """
        Initialize the TackleCounter to track and count tackles and interceptions.
        """
        self.team_tackles = {1: 0, 2: 0}  # Track tackles by team
        self.player_tackles = {}  # Track tackles by player ID
        self.team_interceptions = {1: 0, 2: 0}  # Track interceptions by team
        self.player_interceptions = {}  # Track interceptions by player ID

        # Tracking variables for detecting tackles
        self.last_ball_possessor = -1
        self.last_team_possession = None
        self.possession_change_cooldown = 0
        self.min_possession_frames = 3  # Minimum frames to consider stable possession
        self.possession_frames = 0

        # Distance threshold for tackle detection
        self.tackle_distance_threshold = 50  # pixels

        # Store player positions from previous frame
        self.prev_player_positions = {}

    def detect_tackles_and_interceptions(
        self, current_frame_players, ball_possessor, ball_team, frame_num
    ):
        """
        Detect tackles and interceptions based on ball possession changes and player proximity.

        Args:
            current_frame_players (dict): Dictionary of player data for current frame
            ball_possessor (int): ID of player currently possessing the ball
            ball_team (int): Team of player currently possessing the ball
            frame_num (int): Current frame number
        """
        # Skip if in cooldown period
        if self.possession_change_cooldown > 0:
            self.possession_change_cooldown -= 1
            return

        # If no player has the ball, reset tracking
        if ball_possessor == -1 or ball_team is None:
            self.possession_frames = 0
            return

        # If same player has the ball, increment possession frames
        if ball_possessor == self.last_ball_possessor:
            self.possession_frames += 1
            return

        # Different player has the ball now

        # Check if previous possession was stable
        if (
            self.possession_frames >= self.min_possession_frames
            and self.last_ball_possessor != -1
        ):
            # Check if possession changed to different team (interception)
            if (
                ball_team != self.last_team_possession
                and self.last_team_possession is not None
            ):
                # This is an interception
                self.team_interceptions[ball_team] = (
                    self.team_interceptions.get(ball_team, 0) + 1
                )

                # Track interceptions by player
                if ball_possessor not in self.player_interceptions:
                    self.player_interceptions[ball_possessor] = {
                        "interceptions": 0,
                        "team": ball_team,
                    }
                self.player_interceptions[ball_possessor]["interceptions"] += 1

                print(
                    f"Interception by Team {ball_team}, Player {ball_possessor} at frame {frame_num}"
                )

                # Check if this was also a tackle (player proximity)
                if self._is_tackle(
                    current_frame_players, ball_possessor, self.last_ball_possessor
                ):
                    self.team_tackles[ball_team] = (
                        self.team_tackles.get(ball_team, 0) + 1
                    )

                    # Track tackles by player
                    if ball_possessor not in self.player_tackles:
                        self.player_tackles[ball_possessor] = {
                            "tackles": 0,
                            "team": ball_team,
                        }
                    self.player_tackles[ball_possessor]["tackles"] += 1

                    print(
                        f"Tackle by Team {ball_team}, Player {ball_possessor} at frame {frame_num}"
                    )

            # Set cooldown to prevent multiple detections from same event
            self.possession_change_cooldown = 10

        # Update tracking variables
        self.last_ball_possessor = ball_possessor
        self.last_team_possession = ball_team
        self.possession_frames = 1

        # Store current player positions for next frame
        self.prev_player_positions = {}
        for player_id, player_data in current_frame_players.items():
            if "position" in player_data:
                self.prev_player_positions[player_id] = player_data["position"]

    def _is_tackle(self, current_frame_players, current_possessor, previous_possessor):
        """
        Determine if a possession change was a tackle based on player proximity.

        Args:
            current_frame_players (dict): Dictionary of player data for current frame
            current_possessor (int): ID of player currently possessing the ball
            previous_possessor (int): ID of player previously possessing the ball

        Returns:
            bool: True if the possession change was a tackle, False otherwise
        """
        # Check if both players exist in the current frame
        if (
            current_possessor not in current_frame_players
            or previous_possessor not in current_frame_players
        ):
            return False

        # Check if both players have position data
        if (
            "position" not in current_frame_players[current_possessor]
            or "position" not in current_frame_players[previous_possessor]
        ):
            return False

        # Get player positions
        current_pos = current_frame_players[current_possessor]["position"]
        previous_pos = current_frame_players[previous_possessor]["position"]

        # Calculate distance between players
        distance = np.sqrt(
            (current_pos[0] - previous_pos[0]) ** 2
            + (current_pos[1] - previous_pos[1]) ** 2
        )

        # If players are close enough, consider it a tackle
        return distance < self.tackle_distance_threshold

    def export_tackle_stats_to_csv(self, output_path="tackle_stats.csv"):
        """
        Export tackle and interception statistics to a CSV file.

        Args:
            output_path (str): Path to save the CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Combine player tackles and interceptions data
        player_stats = {}

        # Add tackle data
        for player_id, data in self.player_tackles.items():
            if player_id not in player_stats:
                player_stats[player_id] = {
                    "team": data.get("team", "Unknown"),
                    "tackles": data.get("tackles", 0),
                    "interceptions": 0,
                }
            else:
                player_stats[player_id]["tackles"] = data.get("tackles", 0)
                player_stats[player_id]["team"] = data.get(
                    "team", player_stats[player_id]["team"]
                )

        # Add interception data
        for player_id, data in self.player_interceptions.items():
            if player_id not in player_stats:
                player_stats[player_id] = {
                    "team": data.get("team", "Unknown"),
                    "tackles": 0,
                    "interceptions": data.get("interceptions", 0),
                }
            else:
                player_stats[player_id]["interceptions"] = data.get("interceptions", 0)
                player_stats[player_id]["team"] = data.get(
                    "team", player_stats[player_id]["team"]
                )

        # Write to CSV
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "player_id",
                "jersey_number",
                "team",
                "tackles",
                "interceptions",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for player_id, data in player_stats.items():
                writer.writerow(
                    {
                        "player_id": player_id,
                        "jersey_number": player_id,  # Using player_id as jersey number for now
                        "team": data.get("team", "Unknown"),
                        "tackles": data.get("tackles", 0),
                        "interceptions": data.get("interceptions", 0),
                    }
                )

        print(f"Tackle and interception statistics exported to {output_path}")

        # Also export team statistics
        team_stats_path = output_path.replace(
            "tackle_stats.csv", "team_tackle_stats.csv"
        )
        with open(team_stats_path, "w", newline="") as csvfile:
            fieldnames = ["team", "tackles", "interceptions"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for team_id in [1, 2]:
                writer.writerow(
                    {
                        "team": team_id,
                        "tackles": self.team_tackles.get(team_id, 0),
                        "interceptions": self.team_interceptions.get(team_id, 0),
                    }
                )

        print(f"Team tackle statistics exported to {team_stats_path}")
        print(f"Team tackle statistics exported to {team_stats_path}")
