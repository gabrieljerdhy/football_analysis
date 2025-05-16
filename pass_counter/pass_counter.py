import cv2
import numpy as np


class PassCounter:
    def __init__(self):
        """
        Initialize the PassCounter to track and count passes between players of the same team.
        """
        self.team_passes = {1: 0, 2: 0}  # Using numeric keys instead of strings
        self.last_player_id = -1
        self.last_team = None
        self.pass_detected = False
        self.no_possession_frames = 0
        self.same_player_frames = 0
        self.min_frames_for_pass = 3  # Minimum frames to consider a stable possession

        # Store pass counts per frame to show incremental progress
        self.pass_counts_per_frame = []

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
                self.pass_detected = True
                print(
                    f"Pass detected for team {current_team}: {self.team_passes[current_team]} at frame {frame_num}"
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
