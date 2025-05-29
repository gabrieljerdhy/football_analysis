import os
import sys

import cv2
import numpy as np

sys.path.append("../")
from .field_keypoints_detector import FieldKeypointsDetector


class GoalDetector:
    """
    Enhanced goal detection using field keypoints and ball tracking.
    """

    def __init__(self, keypoints_detector=None, goal_cooldown_frames=60):
        """
        Initialize the goal detector.

        Args:
            keypoints_detector: FieldKeypointsDetector instance
            goal_cooldown_frames (int): Frames to wait before detecting another goal
        """
        self.keypoints_detector = keypoints_detector or FieldKeypointsDetector()
        self.goal_cooldown_frames = goal_cooldown_frames

        # Goal tracking state
        self.team_goals = {1: 0, 2: 0}
        self.player_goals = {}
        self.goal_events = []  # List of goal events with details

        # Detection state
        self.goal_detected = False
        self.goal_cooldown = 0
        self.ball_trajectory = []  # Track ball movement for goal validation
        self.max_trajectory_length = 10

        # Fallback goal areas (if keypoints not detected)
        # These will be updated based on actual video dimensions
        self.fallback_goal_areas = {
            "left": {"x_min": 0, "x_max": 150, "y_min": 300, "y_max": 780},
            "right": {"x_min": 1770, "x_max": 1920, "y_min": 300, "y_max": 780},
        }

        # Dynamic goal areas based on video dimensions
        self.video_dimensions = None

    def update_keypoints(self, frame):
        """
        Update field keypoints detection for the current frame.

        Args:
            frame: Current video frame
        """
        # Update video dimensions if not set
        if self.video_dimensions is None:
            height, width = frame.shape[:2]
            self.video_dimensions = (width, height)
            self._update_fallback_goal_areas(width, height)

        self.keypoints_detector.detect_keypoints(frame)

    def _update_fallback_goal_areas(self, width, height):
        """
        Update fallback goal areas based on video dimensions.

        Args:
            width (int): Video width
            height (int): Video height
        """
        # Calculate goal areas as percentage of video dimensions
        goal_width_percent = 0.08  # 8% of video width
        goal_height_percent = 0.45  # 45% of video height (centered)

        goal_width = int(width * goal_width_percent)
        goal_height = int(height * goal_height_percent)
        goal_y_start = int(height * 0.275)  # Start at 27.5% from top
        goal_y_end = goal_y_start + goal_height

        self.fallback_goal_areas = {
            "left": {
                "x_min": 0,
                "x_max": goal_width,
                "y_min": goal_y_start,
                "y_max": goal_y_end,
            },
            "right": {
                "x_min": width - goal_width,
                "x_max": width,
                "y_min": goal_y_start,
                "y_max": goal_y_end,
            },
        }

        print(f"📐 Updated goal areas for {width}x{height} video:")
        print(f"   Left goal: {self.fallback_goal_areas['left']}")
        print(f"   Right goal: {self.fallback_goal_areas['right']}")

    def detect_goal(self, ball_position, player_id, team, frame_num):
        """
        Detect if a goal has been scored using enhanced logic.

        Args:
            ball_position (tuple): (x, y) position of the ball
            player_id (int): ID of the player who last touched the ball
            team (int): Team of the player
            frame_num (int): Current frame number

        Returns:
            dict or None: Goal event details if goal detected, None otherwise
        """
        # Skip if we're in goal cooldown period
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1
            return None

        if not ball_position:
            return None

        # Add ball position to trajectory
        self._update_ball_trajectory(ball_position)

        # Check if ball is in goal area using keypoints
        goal_side = self._check_ball_in_goal(ball_position)

        if goal_side and not self.goal_detected:
            # Validate goal using trajectory analysis
            if self._validate_goal_trajectory(goal_side):
                goal_event = self._register_goal(
                    goal_side, player_id, team, frame_num, ball_position
                )
                return goal_event

        # Reset goal detection when ball is in middle of field
        elif self._is_ball_in_midfield(ball_position):
            self.goal_detected = False

        return None

    def _update_ball_trajectory(self, ball_position):
        """
        Update ball trajectory for goal validation.

        Args:
            ball_position (tuple): Current ball position
        """
        self.ball_trajectory.append(ball_position)

        # Keep only recent positions
        if len(self.ball_trajectory) > self.max_trajectory_length:
            self.ball_trajectory.pop(0)

    def _check_ball_in_goal(self, ball_position):
        """
        Check if ball is in goal area using keypoints or fallback areas.

        Args:
            ball_position (tuple): Ball position

        Returns:
            str or None: "left" or "right" if in goal, None otherwise
        """
        # Try using keypoints first
        goal_side = self.keypoints_detector.is_ball_in_goal_area(ball_position)

        if goal_side:
            return goal_side

        # Fallback to hardcoded areas
        x, y = ball_position

        # Check left goal
        left_area = self.fallback_goal_areas["left"]
        if (
            left_area["x_min"] <= x <= left_area["x_max"]
            and left_area["y_min"] <= y <= left_area["y_max"]
        ):
            return "left"

        # Check right goal
        right_area = self.fallback_goal_areas["right"]
        if (
            right_area["x_min"] <= x <= right_area["x_max"]
            and right_area["y_min"] <= y <= right_area["y_max"]
        ):
            return "right"

        return None

    def _validate_goal_trajectory(self, goal_side):
        """
        Validate goal using ball trajectory analysis.

        Args:
            goal_side (str): Which goal side ("left" or "right")

        Returns:
            bool: True if trajectory indicates a valid goal
        """
        if len(self.ball_trajectory) < 3:
            return True  # Not enough data, assume valid

        # Check if ball was moving towards the goal
        recent_positions = self.ball_trajectory[-3:]

        if goal_side == "left":
            # Ball should be moving left (decreasing x)
            x_movement = recent_positions[-1][0] - recent_positions[0][0]
            return x_movement < 0
        elif goal_side == "right":
            # Ball should be moving right (increasing x)
            x_movement = recent_positions[-1][0] - recent_positions[0][0]
            return x_movement > 0

        return True

    def _is_ball_in_midfield(self, ball_position):
        """
        Check if ball is in midfield area.

        Args:
            ball_position (tuple): Ball position

        Returns:
            bool: True if ball is in midfield
        """
        x, y = ball_position

        # Use dynamic midfield based on video dimensions
        if self.video_dimensions:
            width, height = self.video_dimensions
            midfield_start = int(width * 0.25)  # 25% from left
            midfield_end = int(width * 0.75)  # 75% from left
            return midfield_start < x < midfield_end
        else:
            # Fallback to hardcoded values
            return 400 < x < 880

    def _register_goal(self, goal_side, player_id, team, frame_num, ball_position):
        """
        Register a goal event.

        Args:
            goal_side (str): Which goal ("left" or "right")
            player_id (int): Player who scored
            team (int): Team that scored
            frame_num (int): Frame number
            ball_position (tuple): Ball position when goal was scored

        Returns:
            dict: Goal event details
        """
        # Determine scoring team based on goal side
        # Assuming team 1 attacks right goal, team 2 attacks left goal
        scoring_team = 1 if goal_side == "right" else 2

        # Update team goals
        self.team_goals[scoring_team] += 1

        # Update player goals
        if player_id != -1 and team == scoring_team:
            if player_id not in self.player_goals:
                self.player_goals[player_id] = {"goals": 0, "team": team}
            self.player_goals[player_id]["goals"] += 1

        # Create goal event
        goal_event = {
            "team": scoring_team,
            "player_id": player_id if team == scoring_team else -1,
            "frame_num": frame_num,
            "goal_side": goal_side,
            "ball_position": ball_position,
            "trajectory": self.ball_trajectory.copy(),
        }

        self.goal_events.append(goal_event)

        # Set cooldown
        self.goal_detected = True
        self.goal_cooldown = self.goal_cooldown_frames

        # Print goal notification
        print(
            f"🥅 GOAL for Team {scoring_team} at frame {frame_num}! Total: {self.team_goals[scoring_team]}"
        )
        if player_id != -1 and team == scoring_team:
            print(f"⚽ Scored by Player {player_id}")
        print(f"📍 Goal side: {goal_side}, Ball position: {ball_position}")

        return goal_event

    def get_goal_statistics(self):
        """
        Get comprehensive goal statistics.

        Returns:
            dict: Goal statistics including team and player goals
        """
        return {
            "team_goals": self.team_goals.copy(),
            "player_goals": self.player_goals.copy(),
            "goal_events": self.goal_events.copy(),
            "total_goals": sum(self.team_goals.values()),
        }

    def draw_goal_info(self, frame):
        """
        Draw goal information on the frame.

        Args:
            frame: Input frame

        Returns:
            frame: Frame with goal information drawn
        """
        frame_copy = frame.copy()

        # Draw goal areas
        frame_copy = self.keypoints_detector.draw_goal_areas(frame_copy)

        # Draw goal score
        score_text = f"Team 1: {self.team_goals[1]} - Team 2: {self.team_goals[2]}"
        cv2.putText(
            frame_copy,
            score_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            frame_copy,
            score_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
        )

        # Draw ball trajectory if available
        if len(self.ball_trajectory) > 1:
            for i in range(1, len(self.ball_trajectory)):
                cv2.line(
                    frame_copy,
                    self.ball_trajectory[i - 1],
                    self.ball_trajectory[i],
                    (0, 255, 255),
                    2,
                )

        return frame_copy

    def reset_detection_state(self):
        """
        Reset the goal detection state (useful for new video analysis).
        """
        self.team_goals = {1: 0, 2: 0}
        self.player_goals = {}
        self.goal_events = []
        self.goal_detected = False
        self.goal_cooldown = 0
        self.ball_trajectory = []
