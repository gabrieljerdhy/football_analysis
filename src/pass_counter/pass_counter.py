import csv
import os

import cv2
import numpy as np


class PassCounter:
    def __init__(self, video_width=None, video_height=None):
        """
        Initialize the PassCounter to track and count passes between players of the same team.

        Args:
            video_width (int): Width of the video in pixels
            video_height (int): Height of the video in pixels
        """
        self.team_passes = {1: 0, 2: 0}  # Using numeric keys instead of strings
        self.player_passes = {}  # Track passes by player ID
        self.team_goals = {1: 0, 2: 0}  # Track goals by team
        self.player_goals = {}  # Track goals by player ID
        self.last_player_id = -1
        self.last_team = None
        self.pass_detected = False
        self.no_possession_frames = 0
        self.same_player_frames = 0
        self.min_frames_for_pass = 3  # Minimum frames to consider a stable possession

        # Video dimensions for goal area calculation
        self.video_width = video_width or 1920  # Default fallback
        self.video_height = video_height or 1080  # Default fallback

        # Goal detection parameters
        self.goal_detected = False
        self.goal_cooldown = 0
        self.goal_cooldown_frames = 300  # FIXED: Reduced to 300 frames (10 seconds at 30fps) for better goal detection

        # Ball trajectory for basic validation
        self.ball_trajectory = []
        self.max_trajectory_length = (
            10  # Keep last 10 positions for trajectory analysis
        )

        # Store pass and goal counts per frame to show incremental progress
        self.pass_counts_per_frame = []
        self.goal_counts_per_frame = []

    def set_video_dimensions(self, width, height):
        """
        Set the video dimensions for accurate goal area calculation.

        Args:
            width (int): Video width in pixels
            height (int): Video height in pixels
        """
        self.video_width = width
        self.video_height = height

    def count_passes(self, current_player_id, current_team, frame_num):
        """
        Count passes when ball possession changes between players of the same team.

        Args:
            current_player_id (int): ID of the player currently possessing the ball
            current_team (int or None): Team of the player currently possessing the ball
            frame_num (int): Current frame number

        Returns:
            int: Current pass count for the team, or 0 if no team is specified
        """
        # Handle None or -1 team values
        if current_team is None or current_team == -1:
            self.no_possession_frames += 1
            # Reset pass detection if ball is not possessed for too long
            if self.no_possession_frames > 5:
                self.pass_detected = False

            # Store current pass counts for this frame
            self.pass_counts_per_frame.append(
                {1: self.team_passes.get(1, 0), 2: self.team_passes.get(2, 0)}
            )
            return 0

        # Reset no possession counter
        self.no_possession_frames = 0

        # Convert to int if needed
        current_team = int(current_team)

        # If this is the first valid possession, just record it
        if self.last_player_id == -1 or self.last_team is None:
            self.last_player_id = current_player_id
            self.last_team = current_team
            self.same_player_frames = 1

            # Store current pass counts for this frame
            self.pass_counts_per_frame.append(
                {1: self.team_passes.get(1, 0), 2: self.team_passes.get(2, 0)}
            )
            return self.team_passes.get(current_team, 0)

        # If same player has the ball, increment counter
        if current_player_id == self.last_player_id:
            self.same_player_frames += 1

            # Store current pass counts for this frame
            self.pass_counts_per_frame.append(
                {1: self.team_passes.get(1, 0), 2: self.team_passes.get(2, 0)}
            )
            return self.team_passes.get(current_team, 0)

        # Different player has the ball
        if current_team == self.last_team:  # Same team
            # Only count as pass if previous player had stable possession
            if (
                self.same_player_frames >= self.min_frames_for_pass
                and not self.pass_detected
            ):
                self.team_passes[current_team] = (
                    self.team_passes.get(current_team, 0) + 1
                )

                # Track passes by player ID - new code for player pass tracking
                if self.last_player_id not in self.player_passes:
                    self.player_passes[self.last_player_id] = {
                        "passes": 0,
                        "team": current_team,
                    }
                self.player_passes[self.last_player_id]["passes"] += 1

                self.pass_detected = True
                print(
                    f"Pass detected for team {current_team}: {self.team_passes[current_team]} at frame {frame_num}"
                )
                print(
                    f"Player {self.last_player_id} made a pass (total: {self.player_passes[self.last_player_id]['passes']})"
                )
        else:
            # Different team has the ball now, reset pass detection
            self.pass_detected = False

        # Update last possession
        self.last_player_id = current_player_id
        self.last_team = current_team
        self.same_player_frames = 1

        # Store current pass counts for this frame
        self.pass_counts_per_frame.append(
            {1: self.team_passes.get(1, 0), 2: self.team_passes.get(2, 0)}
        )

        return self.team_passes.get(current_team, 0)

    def draw_pass_counts(self, frames):
        """
        Draw pass counts for each team on the video frames, showing incremental progress.

        Args:
            frames (list): List of video frames to draw on

        Returns:
            list: Updated video frames with pass counts drawn
        """
        # Ensure we have pass counts for each frame
        if len(self.pass_counts_per_frame) < len(frames):
            # Extend with the last known counts if needed
            last_counts = (
                self.pass_counts_per_frame[-1]
                if self.pass_counts_per_frame
                else {1: 0, 2: 0}
            )
            self.pass_counts_per_frame.extend(
                [last_counts] * (len(frames) - len(self.pass_counts_per_frame))
            )

        for i, frame in enumerate(frames):
            # Create a semi-transparent background for better readability
            overlay = frame.copy()

            # Position below ball control UI (which is at 1350, 850 to 1900, 970)
            # Draw background rectangle for team 1 and team 2
            cv2.rectangle(overlay, (1350, 980), (1900, 1080), (0, 0, 0), -1)

            # Apply transparency
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get the pass counts for this specific frame
            frame_counts = self.pass_counts_per_frame[
                min(i, len(self.pass_counts_per_frame) - 1)
            ]

            # Draw team 1 pass count with team color indicator
            cv2.rectangle(
                frame, (1360, 990), (1380, 1020), (0, 0, 255), -1
            )  # Red indicator for team 1
            cv2.putText(
                frame,
                f"Team 1 Passes: {frame_counts[1]}",
                (1400, 1010),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Draw team 2 pass count with team color indicator
            cv2.rectangle(
                frame, (1360, 1030), (1380, 1060), (255, 0, 0), -1
            )  # Blue indicator for team 2
            cv2.putText(
                frame,
                f"Team 2 Passes: {frame_counts[2]}",
                (1400, 1050),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        return frames

    def export_player_passes_to_csv(self, output_path="data/output/player_passes.csv"):
        """
        Export player pass data to a CSV file.

        Args:
            output_path (str): Path to save the CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w", newline="") as csvfile:
            fieldnames = ["player_id", "jersey_number", "team", "passes"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for player_id, data in self.player_passes.items():
                writer.writerow(
                    {
                        "player_id": player_id,
                        "jersey_number": player_id,  # Using player_id as jersey number for now
                        "team": data.get("team", "Unknown"),
                        "passes": data.get("passes", 0),
                    }
                )

        print(f"Player pass data exported to {output_path}")

    def export_player_stats_to_csv(self, output_path="data/output/player_stats.csv"):
        """
        Export player statistics (passes and goals) to a CSV file.

        Args:
            output_path (str): Path to save the CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Combine player passes and goals data
        player_stats = {}

        # Add pass data
        for player_id, data in self.player_passes.items():
            if player_id not in player_stats:
                player_stats[player_id] = {
                    "team": data.get("team", "Unknown"),
                    "passes": data.get("passes", 0),
                    "goals": 0,
                }
            else:
                player_stats[player_id]["passes"] = data.get("passes", 0)
                player_stats[player_id]["team"] = data.get(
                    "team", player_stats[player_id]["team"]
                )

        # Add goal data
        for player_id, data in self.player_goals.items():
            if player_id not in player_stats:
                player_stats[player_id] = {
                    "team": data.get("team", "Unknown"),
                    "passes": 0,
                    "goals": data.get("goals", 0),
                }
            else:
                player_stats[player_id]["goals"] = data.get("goals", 0)
                player_stats[player_id]["team"] = data.get(
                    "team", player_stats[player_id]["team"]
                )

        # Write to CSV
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = ["player_id", "jersey_number", "team", "passes", "goals"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for player_id, data in player_stats.items():
                writer.writerow(
                    {
                        "player_id": player_id,
                        "jersey_number": player_id,  # Using player_id as jersey number for now
                        "team": data.get("team", "Unknown"),
                        "passes": data.get("passes", 0),
                        "goals": data.get("goals", 0),
                    }
                )

        print(f"Player statistics exported to {output_path}")

        # Also export team statistics
        team_stats_path = output_path.replace("player_stats.csv", "team_stats.csv")
        with open(team_stats_path, "w", newline="") as csvfile:
            fieldnames = ["team", "passes", "goals"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for team_id in [1, 2]:
                writer.writerow(
                    {
                        "team": team_id,
                        "passes": self.team_passes.get(team_id, 0),
                        "goals": self.team_goals.get(team_id, 0),
                    }
                )

        print(f"Team statistics exported to {team_stats_path}")

    def detect_goal(self, ball_position, player_id, team, frame_num):
        """
        Detect if a goal has been scored based on ball position and movement.

        Args:
            ball_position (tuple): (x, y) position of the ball
            player_id (int): ID of the player who last touched the ball
            team (int): Team of the player
            frame_num (int): Current frame number

        Returns:
            bool: True if a goal was detected, False otherwise
        """
        # Skip if we're in goal cooldown period
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1
            return False

        if not ball_position:
            return False

        x, y = ball_position

        # Add ball position to trajectory for validation
        self._update_ball_trajectory(ball_position)

        # FIXED: Much more realistic goal areas based on actual goal post dimensions
        # Real football goal is approximately 7.32m wide × 2.44m high
        # On a typical football field view, goals are much smaller than previous implementation

        # Calculate balanced goal dimensions for better detection
        goal_width = int(
            self.video_width * 0.06
        )  # FIXED: 6% of width for realistic goal detection
        goal_height = int(
            self.video_height * 0.30
        )  # FIXED: 30% of height for proper goal coverage

        # Position goals in typical locations for football broadcasts
        goal_y_center = int(self.video_height * 0.5)  # Center vertically
        goal_y_start = goal_y_center - goal_height // 2
        goal_y_end = goal_y_center + goal_height // 2

        # Left goal (Team 2 scores) - very restrictive
        left_goal_area = {
            "x_min": 0,
            "x_max": goal_width,
            "y_min": goal_y_start,
            "y_max": goal_y_end,
        }

        # Right goal (Team 1 scores) - very restrictive
        right_goal_area = {
            "x_min": self.video_width - goal_width,
            "x_max": self.video_width,
            "y_min": goal_y_start,
            "y_max": goal_y_end,
        }

        # Check for goal with trajectory validation
        goal_detected = False

        if (
            left_goal_area["x_min"] <= x <= left_goal_area["x_max"]
            and left_goal_area["y_min"] <= y <= left_goal_area["y_max"]
            and not self.goal_detected
        ):
            # Validate trajectory - ball should be moving toward the goal
            if self._validate_goal_trajectory("left"):
                self.team_goals[2] = self.team_goals.get(2, 0) + 1
                self.goal_detected = True
                self.goal_cooldown = self.goal_cooldown_frames
                goal_detected = True

                # Record the player who scored
                if player_id != -1 and team == 2:
                    if player_id not in self.player_goals:
                        self.player_goals[player_id] = {"goals": 0, "team": team}
                    self.player_goals[player_id]["goals"] += 1

                print(
                    f"GOAL for Team 2 at frame {frame_num}! Total: {self.team_goals[2]}"
                )
                if player_id != -1 and team == 2:
                    print(f"Scored by Player {player_id}")

        elif (
            right_goal_area["x_min"] <= x <= right_goal_area["x_max"]
            and right_goal_area["y_min"] <= y <= right_goal_area["y_max"]
            and not self.goal_detected
        ):
            # Validate trajectory - ball should be moving toward the goal
            if self._validate_goal_trajectory("right"):
                self.team_goals[1] = self.team_goals.get(1, 0) + 1
                self.goal_detected = True
                self.goal_cooldown = self.goal_cooldown_frames
                goal_detected = True

                # Record the player who scored
                if player_id != -1 and team == 1:
                    if player_id not in self.player_goals:
                        self.player_goals[player_id] = {"goals": 0, "team": team}
                    self.player_goals[player_id]["goals"] += 1

                print(
                    f"GOAL for Team 1 at frame {frame_num}! Total: {self.team_goals[1]}"
                )
                if player_id != -1 and team == 1:
                    print(f"Scored by Player {player_id}")

        # Reset goal detection when ball is in middle of field (expanded middle area)
        middle_start = int(self.video_width * 0.25)  # 25% from left
        middle_end = int(self.video_width * 0.75)  # 75% from left
        if middle_start < x < middle_end and not self.goal_cooldown:
            self.goal_detected = False

        # Store current goal counts for this frame
        self.goal_counts_per_frame.append(
            {1: self.team_goals.get(1, 0), 2: self.team_goals.get(2, 0)}
        )

        return goal_detected

    def _update_ball_trajectory(self, ball_position):
        """
        Update ball trajectory for goal validation.

        Args:
            ball_position (tuple): (x, y) position of the ball
        """
        self.ball_trajectory.append(ball_position)

        # Keep only the last N positions
        if len(self.ball_trajectory) > self.max_trajectory_length:
            self.ball_trajectory.pop(0)

    def _validate_goal_trajectory(self, goal_side):
        """
        Validate if the ball trajectory indicates a valid goal.

        Args:
            goal_side (str): "left" or "right" goal

        Returns:
            bool: True if trajectory indicates a valid goal
        """
        if len(self.ball_trajectory) < 3:
            return False  # Need at least 3 positions for trajectory analysis

        # Get the last 3 positions
        recent_positions = self.ball_trajectory[-3:]

        # Calculate movement direction
        start_x = recent_positions[0][0]
        end_x = recent_positions[-1][0]
        movement_x = end_x - start_x

        # For left goal, ball should be moving left (negative x direction)
        if goal_side == "left":
            return (
                movement_x < -3
            )  # Ball must be moving left with minimum speed (reduced from -5)

        # For right goal, ball should be moving right (positive x direction)
        elif goal_side == "right":
            return (
                movement_x > 3
            )  # Ball must be moving right with minimum speed (reduced from 5)

        return False

    def draw_goal_counts(self, frames):
        """
        Draw goal counts for each team on the video frames.

        Args:
            frames (list): List of video frames to draw on

        Returns:
            list: Updated video frames with goal counts drawn
        """
        # Ensure we have goal counts for each frame
        if len(self.goal_counts_per_frame) < len(frames):
            # Extend with the last known counts if needed
            last_counts = (
                self.goal_counts_per_frame[-1]
                if self.goal_counts_per_frame
                else {1: 0, 2: 0}
            )
            self.goal_counts_per_frame.extend(
                [last_counts] * (len(frames) - len(self.goal_counts_per_frame))
            )

        for i, frame in enumerate(frames):
            # Create a semi-transparent background for better readability
            overlay = frame.copy()

            # Position below pass counts UI
            cv2.rectangle(overlay, (1350, 1090), (1900, 1170), (0, 0, 0), -1)

            # Apply transparency
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get the goal counts for this specific frame
            frame_counts = self.goal_counts_per_frame[
                min(i, len(self.goal_counts_per_frame) - 1)
            ]

            # Draw team 1 goal count with team color indicator
            cv2.rectangle(
                frame, (1360, 1100), (1380, 1130), (0, 0, 255), -1
            )  # Red indicator for team 1
            cv2.putText(
                frame,
                f"Team 1 Goals: {frame_counts[1]}",
                (1400, 1120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Draw team 2 goal count with team color indicator
            cv2.rectangle(
                frame, (1360, 1140), (1380, 1170), (255, 0, 0), -1
            )  # Blue indicator for team 2
            cv2.putText(
                frame,
                f"Team 2 Goals: {frame_counts[2]}",
                (1400, 1160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        return frames

    def add_manual_goal(self, team, player_id=None, frame_num=None):
        """
        Manually add a goal for a team.

        Args:
            team (int): Team number (1 or 2)
            player_id (int, optional): ID of the player who scored
            frame_num (int, optional): Frame number when the goal was scored
        """
        # Convert to int if needed
        team = int(team)

        # Update team goals
        self.team_goals[team] = self.team_goals.get(team, 0) + 1

        # Update player goals if player_id is provided
        if player_id is not None and player_id != -1:
            if player_id not in self.player_goals:
                self.player_goals[player_id] = {"goals": 0, "team": team}
            self.player_goals[player_id]["goals"] += 1

        # Print information
        frame_info = f" at frame {frame_num}" if frame_num is not None else ""
        player_info = (
            f" by Player {player_id}"
            if player_id is not None and player_id != -1
            else ""
        )
        print(
            f"Manual GOAL added for Team {team}{frame_info}! Total: {self.team_goals[team]}{player_info}"
        )

        # Update goal counts per frame if frame_num is provided
        if frame_num is not None:
            # Ensure we have enough entries in goal_counts_per_frame
            while len(self.goal_counts_per_frame) <= frame_num:
                last_counts = (
                    self.goal_counts_per_frame[-1]
                    if self.goal_counts_per_frame
                    else {1: 0, 2: 0}
                )
                self.goal_counts_per_frame.append(last_counts.copy())

            # Update the goal count for this frame and all subsequent frames
            for i in range(frame_num, len(self.goal_counts_per_frame)):
                self.goal_counts_per_frame[i][team] = self.team_goals[team]
