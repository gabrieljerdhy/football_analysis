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

    def __init__(
        self, keypoints_detector=None, goal_cooldown_frames=300
    ):  # FIXED: Reduced to 300 frames (10 seconds at 30fps) to allow multiple goals
        """
        Initialize the goal detector.

        Args:
            keypoints_detector: FieldKeypointsDetector instance
            goal_cooldown_frames (int): Frames to wait before detecting another goal
        """
        self.keypoints_detector = keypoints_detector or FieldKeypointsDetector()
        self.goal_cooldown_frames = goal_cooldown_frames

        # Enhanced goal tracking state
        self.team_goals = {1: 0, 2: 0}  # Enhanced goal count (real-time detection)
        self.player_goals = {}
        self.goal_events = []  # List of goal events with details

        # Final goal counts (validated/corrected goal tally - primary metric)
        self.final_team_goals = {1: 0, 2: 0}
        self.final_player_goals = {}

        # Goal validation and analysis
        self.goal_validation_history = []  # Track validation decisions
        self.goal_confidence_scores = []  # Track confidence of each goal detection

        # Detection state
        self.goal_detected = False
        self.goal_cooldown = 0
        self.ball_trajectory = []  # Track ball movement for goal validation
        self.max_trajectory_length = 20  # Increased for better trajectory analysis

        # FIXED: Balanced parameters to allow goal detection while reducing false positives
        self.min_trajectory_for_goal = (
            6  # FIXED: Reduced from 12 to 6 for better sensitivity
        )
        self.goal_direction_threshold = (
            0.6  # FIXED: Reduced from 0.8 to 0.6 for more flexible movement detection
        )
        self.goal_speed_threshold = (
            8  # FIXED: Reduced from 15 to 8 for more realistic goal speeds
        )

        # Fallback goal areas (if keypoints not detected)
        # These will be updated based on actual video dimensions
        self.fallback_goal_areas = {
            "left": {"x_min": 0, "x_max": 150, "y_min": 300, "y_max": 780},
            "right": {"x_min": 1770, "x_max": 1920, "y_min": 300, "y_max": 780},
        }

        # Dynamic goal areas based on video dimensions
        self.video_dimensions = None

    def update_keypoints(self, frame, force_detection=False):
        """
        Update field keypoints detection for the current frame with optimization.

        Args:
            frame: Current video frame
            force_detection (bool): Force keypoint detection even if not at interval
        """
        # Update video dimensions if not set
        if self.video_dimensions is None:
            height, width = frame.shape[:2]
            self.video_dimensions = (width, height)
            self._update_fallback_goal_areas(width, height)
            # Force detection on first frame to establish baseline
            force_detection = True

        self.keypoints_detector.detect_keypoints(frame, force_detection=force_detection)

    def set_keypoint_optimization(self, detection_interval=5, stability_threshold=10):
        """
        Configure keypoint detection optimization parameters.

        Args:
            detection_interval (int): Frames between keypoint detections
            stability_threshold (int): Frames needed to consider keypoints stable
        """
        self.keypoints_detector.set_optimization_parameters(
            detection_interval=detection_interval,
            stability_threshold=stability_threshold,
        )

    def _update_fallback_goal_areas(self, width, height):
        """
        Update fallback goal areas based on video dimensions.

        Args:
            width (int): Video width
            height (int): Video height
        """
        # BALANCED: Realistic goal areas for better detection
        goal_width_percent = (
            0.08  # FIXED: 8% of video width for realistic goal detection
        )
        goal_height_percent = (
            0.35  # FIXED: 35% of video height for proper goal coverage
        )

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

    def detect_goal(
        self,
        ball_position,
        player_id,
        team,
        frame_num,
        ball_confidence=None,
        ball_source=None,
    ):
        """
        Detect if a goal has been scored using enhanced logic with ball detection confidence.

        Args:
            ball_position (tuple): (x, y) position of the ball
            player_id (int): ID of the player who last touched the ball
            team (int): Team of the player
            frame_num (int): Current frame number
            ball_confidence (float): Confidence score of ball detection (optional)
            ball_source (str): Source of ball detection (e.g., 'specialized', 'general', 'interpolated')

        Returns:
            dict or None: Goal event details if goal detected, None otherwise
        """
        # Skip if we're in goal cooldown period
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1
            return None

        if not ball_position:
            return None

        # Enhanced ball position tracking with confidence information
        ball_info = {
            "position": ball_position,
            "confidence": ball_confidence or 0.5,
            "source": ball_source or "unknown",
            "frame_num": frame_num,
        }

        # Add ball position to trajectory with enhanced information
        self._update_ball_trajectory_enhanced(ball_info)

        # Check if ball is in goal area using keypoints
        goal_side = self._check_ball_in_goal(ball_position)

        if goal_side and not self.goal_detected:
            # Enhanced trajectory validation considering ball detection quality
            if self._validate_goal_trajectory_enhanced(
                goal_side, ball_confidence, ball_source
            ):
                goal_event = self._register_goal(
                    goal_side, player_id, team, frame_num, ball_position
                )
                # Add enhanced ball tracking information to goal event
                goal_event["ball_detection_quality"] = {
                    "confidence": ball_confidence,
                    "source": ball_source,
                    "trajectory_confidence": self._calculate_trajectory_confidence(),
                }
                return goal_event

        # Reset goal detection when ball is in middle of field
        elif self._is_ball_in_midfield(ball_position):
            self.goal_detected = False

        return None

    def _update_ball_trajectory_enhanced(self, ball_info):
        """
        Update ball trajectory with enhanced information including confidence and source.

        Args:
            ball_info (dict): Dictionary containing position, confidence, source, and frame_num
        """
        self.ball_trajectory.append(ball_info)

        # Keep trajectory length manageable
        if len(self.ball_trajectory) > self.max_trajectory_length:
            self.ball_trajectory = self.ball_trajectory[-self.max_trajectory_length :]

    def _update_ball_trajectory(self, ball_position):
        """
        Update ball trajectory for goal validation (backward compatibility).

        Args:
            ball_position (tuple): Current ball position
        """
        self.ball_trajectory.append(ball_position)

        # Keep only recent positions
        if len(self.ball_trajectory) > self.max_trajectory_length:
            self.ball_trajectory.pop(0)

    def _check_ball_in_goal(self, ball_position):
        """
        Enhanced check if ball is in goal area using keypoints, penalty areas, and fallback areas.

        Args:
            ball_position (tuple): Ball position

        Returns:
            str or None: "left" or "right" if in goal, None otherwise
        """
        # Get comprehensive field context
        field_context = self.keypoints_detector.get_field_context(ball_position)

        # Primary check: Direct goal area detection
        goal_side = self.keypoints_detector.is_ball_in_goal_area(ball_position)
        if goal_side:
            return goal_side

        # Secondary check: Enhanced goal validation using penalty area context
        penalty_side = field_context["in_penalty_area"]
        if penalty_side:
            # If ball is in penalty area, check if it's close enough to goal line to be considered a goal
            goal_areas = self.keypoints_detector.get_goal_areas()
            penalty_areas = self.keypoints_detector.get_penalty_areas()

            if goal_areas[penalty_side] and penalty_areas[penalty_side]:
                # Calculate if ball is in the goal-side portion of the penalty area
                if penalty_side == "left":
                    # For left goal, ball should be at the leftmost part of penalty area
                    penalty_goal_threshold = (
                        penalty_areas[penalty_side]["x_min"]
                        + (
                            penalty_areas[penalty_side]["x_max"]
                            - penalty_areas[penalty_side]["x_min"]
                        )
                        * 0.2
                    )
                    if ball_position[0] <= penalty_goal_threshold:
                        # Additional check: ball should be within goal height range
                        goal_y_min = goal_areas[penalty_side]["y_min"]
                        goal_y_max = goal_areas[penalty_side]["y_max"]
                        if goal_y_min <= ball_position[1] <= goal_y_max:
                            return penalty_side
                else:  # right goal
                    # For right goal, ball should be at the rightmost part of penalty area
                    penalty_goal_threshold = (
                        penalty_areas[penalty_side]["x_max"]
                        - (
                            penalty_areas[penalty_side]["x_max"]
                            - penalty_areas[penalty_side]["x_min"]
                        )
                        * 0.2
                    )
                    if ball_position[0] >= penalty_goal_threshold:
                        # Additional check: ball should be within goal height range
                        goal_y_min = goal_areas[penalty_side]["y_min"]
                        goal_y_max = goal_areas[penalty_side]["y_max"]
                        if goal_y_min <= ball_position[1] <= goal_y_max:
                            return penalty_side

        # Tertiary check: Fallback to hardcoded areas with enhanced logic
        x, y = ball_position

        # Check left goal with enhanced validation
        left_area = self.fallback_goal_areas["left"]
        if (
            left_area["x_min"] <= x <= left_area["x_max"]
            and left_area["y_min"] <= y <= left_area["y_max"]
        ):
            # SIMPLIFIED: Allow goal detection if ball is in goal area
            if len(self.ball_trajectory) >= 2:
                # Check movement over last 2 frames for direction
                current_x = self.ball_trajectory[-1]["position"][0]
                previous_x = self.ball_trajectory[-2]["position"][0]
                movement_x = current_x - previous_x
                # Ball moving toward goal OR stationary in goal area
                if movement_x <= 1:  # Moving left or stationary (very lenient)
                    return "left"
            else:
                # If not enough trajectory data, allow goal if ball is in area
                return "left"

        # Check right goal with enhanced validation
        right_area = self.fallback_goal_areas["right"]
        if (
            right_area["x_min"] <= x <= right_area["x_max"]
            and right_area["y_min"] <= y <= right_area["y_max"]
        ):
            # SIMPLIFIED: Allow goal detection if ball is in goal area
            if len(self.ball_trajectory) >= 2:
                # Check movement over last 2 frames for direction
                current_x = self.ball_trajectory[-1]["position"][0]
                previous_x = self.ball_trajectory[-2]["position"][0]
                movement_x = current_x - previous_x
                # Ball moving toward goal OR stationary in goal area
                if movement_x >= -1:  # Moving right or stationary (very lenient)
                    return "right"
            else:
                # If not enough trajectory data, allow goal if ball is in area
                return "right"

        return None

    def _validate_goal_trajectory(self, goal_side):
        """
        Advanced goal validation using comprehensive ball trajectory analysis and field context.

        Args:
            goal_side (str): Which goal side ("left" or "right")

        Returns:
            bool: True if trajectory indicates a valid goal
        """
        if len(self.ball_trajectory) < self.min_trajectory_for_goal:
            return False  # IMPROVED: Not enough data, require more evidence for goal

        # Get field context for enhanced validation
        current_ball_info = self.ball_trajectory[-1]
        current_ball_position = current_ball_info["position"]  # Extract (x, y) tuple
        field_context = self.keypoints_detector.get_field_context(current_ball_position)

        # Calculate trajectory metrics with enhanced field-aware scoring
        trajectory_length = len(self.ball_trajectory)
        confidence_score = 0.0

        # 1. Direction consistency check (enhanced with field context)
        direction_score = self._calculate_enhanced_direction_consistency(
            goal_side, field_context
        )
        confidence_score += direction_score * 0.3  # 30% weight

        # 2. Speed analysis (enhanced)
        speed_score = self._calculate_enhanced_speed_consistency(field_context)
        confidence_score += speed_score * 0.25  # 25% weight

        # 3. Trajectory smoothness
        smoothness_score = self._calculate_trajectory_smoothness()
        confidence_score += smoothness_score * 0.15  # 15% weight

        # 4. Enhanced goal approach angle using field keypoints
        angle_score = self._calculate_enhanced_goal_approach_angle(
            goal_side, field_context
        )
        confidence_score += angle_score * 0.15  # 15% weight

        # 5. NEW: Field context validation
        field_context_score = self._calculate_field_context_score(
            goal_side, field_context
        )
        confidence_score += field_context_score * 0.15  # 15% weight

        # Store confidence score for analysis
        self.goal_confidence_scores.append(confidence_score)

        # LENIENT: Very permissive validation threshold for goal detection
        base_threshold = 0.3  # FIXED: Much lower threshold for better goal detection
        field_confidence_bonus = (
            field_context["field_confidence"] * 0.2
        )  # FIXED: Higher bonus for field context
        validation_threshold = max(
            0.2, base_threshold - field_confidence_bonus
        )  # FIXED: Very low minimum threshold

        # Goal is valid if confidence score is above dynamic threshold
        is_valid = confidence_score >= validation_threshold

        # Store enhanced validation decision
        validation_record = {
            "goal_side": goal_side,
            "confidence_score": confidence_score,
            "direction_score": direction_score,
            "speed_score": speed_score,
            "smoothness_score": smoothness_score,
            "angle_score": angle_score,
            "field_context_score": field_context_score,
            "field_confidence": field_context["field_confidence"],
            "validation_threshold": validation_threshold,
            "trajectory_length": trajectory_length,
            "is_valid": is_valid,
            "field_context": field_context,
        }
        self.goal_validation_history.append(validation_record)

        return is_valid

    def _validate_goal_trajectory_enhanced(
        self, goal_side, ball_confidence=None, ball_source=None
    ):
        """
        Enhanced goal validation that considers ball detection quality and confidence.

        Args:
            goal_side (str): Which goal side ("left" or "right")
            ball_confidence (float): Confidence of ball detection
            ball_source (str): Source of ball detection

        Returns:
            bool: True if trajectory indicates a valid goal
        """
        # Use standard validation as base
        base_validation = self._validate_goal_trajectory(goal_side)

        if not base_validation:
            return False

        # Enhanced validation considering ball detection quality
        confidence_modifier = 1.0

        # Adjust confidence based on ball detection source
        if ball_source == "specialized":
            confidence_modifier += 0.1  # Boost for specialized ball model
        elif ball_source == "interpolated_high_conf":
            confidence_modifier += 0.05  # Small boost for high-confidence interpolation
        elif ball_source in ["interpolated_standard", "interpolated"]:
            confidence_modifier -= 0.1  # Penalty for standard interpolation

        # Adjust confidence based on ball detection confidence
        if ball_confidence is not None:
            if ball_confidence >= 0.8:
                confidence_modifier += 0.1  # High confidence detection
            elif ball_confidence >= 0.6:
                confidence_modifier += 0.05  # Medium confidence detection
            elif ball_confidence < 0.3:
                confidence_modifier -= 0.15  # Low confidence detection

        # Calculate enhanced trajectory confidence
        trajectory_confidence = self._calculate_trajectory_confidence()

        # Enhanced validation requires higher standards for interpolated data
        if ball_source and "interpolated" in ball_source:
            # Require higher trajectory confidence for interpolated data
            enhanced_threshold = 0.8
        else:
            enhanced_threshold = 0.7

        # Apply confidence modifier to trajectory confidence
        final_confidence = trajectory_confidence * confidence_modifier

        return final_confidence >= enhanced_threshold

    def _calculate_trajectory_confidence(self):
        """
        Calculate overall confidence of the ball trajectory based on detection quality.

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not self.ball_trajectory:
            return 0.0

        # Analyze trajectory for enhanced ball tracking information
        total_confidence = 0.0
        specialized_count = 0
        interpolated_count = 0

        for ball_info in self.ball_trajectory:
            if isinstance(ball_info, dict) and "confidence" in ball_info:
                # Enhanced trajectory with confidence information
                confidence = ball_info.get("confidence", 0.5)
                source = ball_info.get("source", "unknown")

                # Weight confidence based on source
                if source == "specialized":
                    confidence *= 1.2  # Boost specialized detections
                    specialized_count += 1
                elif "interpolated" in source:
                    confidence *= 0.8  # Reduce interpolated confidence
                    interpolated_count += 1

                total_confidence += min(1.0, confidence)
            else:
                # Legacy trajectory format
                total_confidence += 0.5  # Default confidence

        avg_confidence = total_confidence / len(self.ball_trajectory)

        # Bonus for having more specialized detections
        specialized_ratio = specialized_count / len(self.ball_trajectory)
        specialized_bonus = specialized_ratio * 0.1

        # Penalty for having too many interpolated detections
        interpolated_ratio = interpolated_count / len(self.ball_trajectory)
        interpolated_penalty = interpolated_ratio * 0.1

        final_confidence = avg_confidence + specialized_bonus - interpolated_penalty

        return max(0.0, min(1.0, final_confidence))

    def _calculate_enhanced_direction_consistency(self, goal_side, field_context):
        """
        Enhanced direction consistency calculation using field context.
        """
        base_score = self._calculate_direction_consistency(goal_side)

        # Bonus for being in penalty area moving towards goal
        penalty_bonus = 0.0
        if field_context["in_penalty_area"] == goal_side:
            penalty_bonus = 0.2

        # Bonus based on distance to goal (closer = higher bonus)
        distance_bonus = 0.0
        if goal_side in field_context["distances_to_goals"]:
            distance = field_context["distances_to_goals"][goal_side]
            # Normalize distance bonus (closer distances get higher bonus)
            max_distance = 200  # Assume max relevant distance
            distance_bonus = max(0.0, (max_distance - distance) / max_distance) * 0.1

        return min(1.0, base_score + penalty_bonus + distance_bonus)

    def _calculate_enhanced_speed_consistency(self, field_context):
        """
        Enhanced speed consistency calculation with field context.
        """
        base_score = self._calculate_speed_consistency()

        # Adjust expectations based on field position
        field_bonus = 0.0
        if field_context["in_goal_area"]:
            # In goal area, allow for slower speeds (ball might be settling)
            field_bonus = 0.1
        elif field_context["in_penalty_area"]:
            # In penalty area, expect moderate speeds
            field_bonus = 0.05

        return min(1.0, base_score + field_bonus)

    def _calculate_enhanced_goal_approach_angle(self, goal_side, field_context):
        """
        Enhanced goal approach angle calculation using field keypoints.
        """
        base_score = self._calculate_goal_approach_angle(goal_side)

        # Enhanced scoring using actual goal post positions if available
        goal_areas = self.keypoints_detector.get_goal_areas()
        if goal_areas[goal_side] and "keypoints_used" in goal_areas[goal_side]:
            # Use actual goal post positions for more accurate angle calculation
            goal_center_x = (
                goal_areas[goal_side]["x_min"] + goal_areas[goal_side]["x_max"]
            ) / 2
            goal_center_y = (
                goal_areas[goal_side]["y_min"] + goal_areas[goal_side]["y_max"]
            ) / 2

            if len(self.ball_trajectory) >= 3:
                # Calculate approach vector towards actual goal center
                start_pos = self.ball_trajectory[0]["position"]
                end_pos = self.ball_trajectory[-1]["position"]

                # Vector from start to goal center
                ideal_vector = (
                    goal_center_x - start_pos[0],
                    goal_center_y - start_pos[1],
                )
                # Actual trajectory vector
                actual_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

                # Calculate angle similarity
                ideal_magnitude = (ideal_vector[0] ** 2 + ideal_vector[1] ** 2) ** 0.5
                actual_magnitude = (
                    actual_vector[0] ** 2 + actual_vector[1] ** 2
                ) ** 0.5

                if ideal_magnitude > 0 and actual_magnitude > 0:
                    dot_product = (
                        ideal_vector[0] * actual_vector[0]
                        + ideal_vector[1] * actual_vector[1]
                    )
                    angle_similarity = dot_product / (
                        ideal_magnitude * actual_magnitude
                    )
                    # Convert to positive score (cosine similarity)
                    enhanced_score = max(0.0, angle_similarity)

                    # Weight between base score and enhanced score based on keypoint confidence
                    keypoint_confidence = goal_areas[goal_side].get("confidence", 0.5)
                    final_score = (
                        base_score * (1 - keypoint_confidence)
                        + enhanced_score * keypoint_confidence
                    )
                    return final_score

        return base_score

    def _calculate_field_context_score(self, goal_side, field_context):
        """
        Calculate validation score based on field context.
        """
        score = 0.0

        # 1. Goal area context
        if field_context["in_goal_area"] == goal_side:
            score += 0.4  # High confidence if ball is in correct goal area

        # 2. Penalty area context
        if field_context["in_penalty_area"] == goal_side:
            score += 0.2  # Medium confidence if ball is in correct penalty area

        # 3. Distance-based scoring
        if goal_side in field_context["distances_to_goals"]:
            distance = field_context["distances_to_goals"][goal_side]
            # Closer to goal = higher score
            max_distance = 300  # Maximum relevant distance
            distance_score = max(0.0, (max_distance - distance) / max_distance) * 0.3
            score += distance_score

        # 4. Field detection confidence bonus
        confidence_bonus = field_context["field_confidence"] * 0.1
        score += confidence_bonus

        return min(1.0, score)

    def _calculate_direction_consistency(self, goal_side):
        """Calculate how consistently the ball moves toward the goal."""
        if len(self.ball_trajectory) < 3:
            return 1.0

        movements = []
        for i in range(1, len(self.ball_trajectory)):
            current_pos = self.ball_trajectory[i]["position"]
            prev_pos = self.ball_trajectory[i - 1]["position"]
            x_movement = current_pos[0] - prev_pos[0]
            movements.append(x_movement)

        if goal_side == "left":
            # Ball should be moving left (negative x movement)
            correct_movements = sum(1 for m in movements if m < 0)
        else:
            # Ball should be moving right (positive x movement)
            correct_movements = sum(1 for m in movements if m > 0)

        return correct_movements / len(movements) if movements else 0.0

    def _calculate_speed_consistency(self):
        """Calculate ball speed consistency."""
        if len(self.ball_trajectory) < 3:
            return 1.0

        speeds = []
        for i in range(1, len(self.ball_trajectory)):
            current_pos = self.ball_trajectory[i]["position"]
            prev_pos = self.ball_trajectory[i - 1]["position"]
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            speed = np.sqrt(dx * dx + dy * dy)
            speeds.append(speed)

        if not speeds:
            return 0.0

        avg_speed = np.mean(speeds)
        # Penalize very slow movement (likely tracking errors)
        if avg_speed < self.goal_speed_threshold:
            return 0.3

        # Reward consistent speed
        speed_variance = np.var(speeds)
        consistency = max(0.0, 1.0 - speed_variance / (avg_speed * avg_speed + 1))
        return consistency

    def _calculate_trajectory_smoothness(self):
        """Calculate trajectory smoothness (penalize erratic movement)."""
        if len(self.ball_trajectory) < 4:
            return 1.0

        direction_changes = 0
        for i in range(2, len(self.ball_trajectory)):
            # Calculate direction vectors
            pos_i_2 = self.ball_trajectory[i - 2]["position"]
            pos_i_1 = self.ball_trajectory[i - 1]["position"]
            pos_i = self.ball_trajectory[i]["position"]

            v1 = (
                pos_i_1[0] - pos_i_2[0],
                pos_i_1[1] - pos_i_2[1],
            )
            v2 = (
                pos_i[0] - pos_i_1[0],
                pos_i[1] - pos_i_1[1],
            )

            # Check for significant direction change
            if (v1[0] * v2[0] + v1[1] * v2[1]) < 0:  # Dot product negative
                direction_changes += 1

        max_changes = len(self.ball_trajectory) - 2
        smoothness = 1.0 - (direction_changes / max_changes) if max_changes > 0 else 1.0
        return max(0.0, smoothness)

    def _calculate_goal_approach_angle(self, goal_side):
        """Calculate how well the ball approaches the goal."""
        if len(self.ball_trajectory) < 3:
            return 1.0

        # Get start and end positions
        start_pos = self.ball_trajectory[0]["position"]
        end_pos = self.ball_trajectory[-1]["position"]

        # Calculate approach vector
        approach_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

        # Define ideal approach vectors for each goal
        if goal_side == "left":
            ideal_vector = (-1, 0)  # Moving left
        else:
            ideal_vector = (1, 0)  # Moving right

        # Calculate angle similarity (dot product normalized)
        magnitude = np.sqrt(approach_vector[0] ** 2 + approach_vector[1] ** 2)
        if magnitude == 0:
            return 0.0

        normalized_approach = (
            approach_vector[0] / magnitude,
            approach_vector[1] / magnitude,
        )
        similarity = abs(
            normalized_approach[0] * ideal_vector[0]
            + normalized_approach[1] * ideal_vector[1]
        )

        return similarity

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

        # Get confidence score from latest validation
        confidence_score = (
            self.goal_confidence_scores[-1] if self.goal_confidence_scores else 0.5
        )

        # Get comprehensive field context for enhanced goal event analysis
        field_context = self.keypoints_detector.get_field_context(ball_position)
        goal_areas = self.keypoints_detector.get_goal_areas()
        penalty_areas = self.keypoints_detector.get_penalty_areas()

        # Enhanced goal event analysis
        goal_analysis = self._analyze_goal_event(
            goal_side, ball_position, field_context, goal_areas, penalty_areas
        )

        # Create enhanced goal event with comprehensive data
        goal_event = {
            "team": scoring_team,
            "player_id": player_id if team == scoring_team else -1,
            "frame_num": frame_num,
            "goal_side": goal_side,
            "ball_position": ball_position,
            "trajectory": self.ball_trajectory.copy(),
            "confidence_score": confidence_score,
            "validation_data": (
                self.goal_validation_history[-1]
                if self.goal_validation_history
                else None
            ),
            "timestamp": frame_num / 30.0 if frame_num else 0,  # Assuming 30 FPS
            "field_context": field_context,
            "goal_analysis": goal_analysis,
        }

        self.goal_events.append(goal_event)

        # Update final goal counts (these are the primary metrics)
        self.final_team_goals[scoring_team] += 1
        if player_id != -1 and team == scoring_team:
            if player_id not in self.final_player_goals:
                self.final_player_goals[player_id] = {"goals": 0, "team": team}
            self.final_player_goals[player_id]["goals"] += 1

        # Set cooldown
        self.goal_detected = True
        self.goal_cooldown = self.goal_cooldown_frames

        # Enhanced goal notification with confidence
        print(
            f"🥅 ENHANCED GOAL for Team {scoring_team} at frame {frame_num}! "
            f"Enhanced: {self.team_goals[scoring_team]}, Final: {self.final_team_goals[scoring_team]}"
        )
        if player_id != -1 and team == scoring_team:
            print(f"⚽ Scored by Player {player_id}")
        print(f"📍 Goal side: {goal_side}, Ball position: {ball_position}")
        print(f"🎯 Confidence: {confidence_score:.2f}")

        return goal_event

    def _analyze_goal_event(
        self, goal_side, ball_position, field_context, goal_areas, penalty_areas
    ):
        """
        Perform comprehensive analysis of a goal event using field context and geometry.

        Args:
            goal_side (str): Which goal side ("left" or "right")
            ball_position (tuple): Ball position when goal was scored
            field_context (dict): Field context information
            goal_areas (dict): Goal area boundaries
            penalty_areas (dict): Penalty area boundaries

        Returns:
            dict: Comprehensive goal event analysis
        """
        analysis = {
            "goal_type": "unknown",
            "approach_angle": 0.0,
            "distance_from_goal_center": 0.0,
            "penalty_area_entry": False,
            "goal_post_proximity": {},
            "trajectory_quality": "unknown",
            "field_geometry_score": 0.0,
        }

        # 1. Determine goal type based on field context
        if field_context["in_goal_area"] == goal_side:
            analysis["goal_type"] = "close_range"
        elif field_context["in_penalty_area"] == goal_side:
            analysis["goal_type"] = "penalty_area"
        else:
            analysis["goal_type"] = "long_range"

        # 2. Calculate approach angle using actual goal geometry
        if goal_areas[goal_side] and len(self.ball_trajectory) >= 3:
            goal_center_x = (
                goal_areas[goal_side]["x_min"] + goal_areas[goal_side]["x_max"]
            ) / 2
            goal_center_y = (
                goal_areas[goal_side]["y_min"] + goal_areas[goal_side]["y_max"]
            ) / 2

            # Calculate approach vector from trajectory start to goal center
            start_pos = self.ball_trajectory[0]["position"]
            approach_vector = (
                goal_center_x - start_pos[0],
                goal_center_y - start_pos[1],
            )

            # Calculate actual trajectory vector
            end_pos = self.ball_trajectory[-1]["position"]
            trajectory_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

            # Calculate angle between ideal approach and actual trajectory
            approach_magnitude = (
                approach_vector[0] ** 2 + approach_vector[1] ** 2
            ) ** 0.5
            trajectory_magnitude = (
                trajectory_vector[0] ** 2 + trajectory_vector[1] ** 2
            ) ** 0.5

            if approach_magnitude > 0 and trajectory_magnitude > 0:
                dot_product = (
                    approach_vector[0] * trajectory_vector[0]
                    + approach_vector[1] * trajectory_vector[1]
                )
                cos_angle = dot_product / (approach_magnitude * trajectory_magnitude)
                angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                analysis["approach_angle"] = np.degrees(angle_radians)

            # Calculate distance from goal center
            analysis["distance_from_goal_center"] = (
                (ball_position[0] - goal_center_x) ** 2
                + (ball_position[1] - goal_center_y) ** 2
            ) ** 0.5

        # 3. Check penalty area entry during trajectory
        if penalty_areas[goal_side] and len(self.ball_trajectory) >= 2:
            for pos in self.ball_trajectory:
                if (
                    penalty_areas[goal_side]["x_min"]
                    <= pos[0]
                    <= penalty_areas[goal_side]["x_max"]
                    and penalty_areas[goal_side]["y_min"]
                    <= pos[1]
                    <= penalty_areas[goal_side]["y_max"]
                ):
                    analysis["penalty_area_entry"] = True
                    break

        # 4. Calculate goal post proximity if keypoints are available
        if goal_areas[goal_side] and "keypoints_used" in goal_areas[goal_side]:
            detected_keypoints = self.keypoints_detector.detected_keypoints
            goal_keypoints = (
                self.keypoints_detector.left_goal_keypoints
                if goal_side == "left"
                else self.keypoints_detector.right_goal_keypoints
            )

            for keypoint in goal_keypoints:
                if keypoint in detected_keypoints:
                    post_pos = detected_keypoints[keypoint]["position"]
                    distance = (
                        (ball_position[0] - post_pos[0]) ** 2
                        + (ball_position[1] - post_pos[1]) ** 2
                    ) ** 0.5
                    analysis["goal_post_proximity"][keypoint] = {
                        "distance": distance,
                        "confidence": detected_keypoints[keypoint]["confidence"],
                    }

        # 5. Assess trajectory quality
        if len(self.ball_trajectory) >= 5:
            # Calculate trajectory smoothness and consistency
            direction_changes = 0
            for i in range(2, len(self.ball_trajectory)):
                pos_i_2 = self.ball_trajectory[i - 2]["position"]
                pos_i_1 = self.ball_trajectory[i - 1]["position"]
                pos_i = self.ball_trajectory[i]["position"]

                v1 = (
                    pos_i_1[0] - pos_i_2[0],
                    pos_i_1[1] - pos_i_2[1],
                )
                v2 = (
                    pos_i[0] - pos_i_1[0],
                    pos_i[1] - pos_i_1[1],
                )

                if (v1[0] * v2[0] + v1[1] * v2[1]) < 0:
                    direction_changes += 1

            smoothness = 1.0 - (
                direction_changes / max(1, len(self.ball_trajectory) - 2)
            )
            if smoothness > 0.8:
                analysis["trajectory_quality"] = "smooth"
            elif smoothness > 0.6:
                analysis["trajectory_quality"] = "moderate"
            else:
                analysis["trajectory_quality"] = "erratic"

        # 6. Calculate field geometry score
        geometry_score = 0.0
        if field_context["field_confidence"] > 0.7:
            geometry_score += 0.3
        if analysis["goal_type"] in ["close_range", "penalty_area"]:
            geometry_score += 0.3
        if analysis["approach_angle"] < 45:  # Good approach angle
            geometry_score += 0.2
        if analysis["penalty_area_entry"]:
            geometry_score += 0.2

        analysis["field_geometry_score"] = geometry_score

        return analysis

    def get_goal_statistics(self):
        """
        Get comprehensive goal statistics with enhanced and final counts.

        Returns:
            dict: Enhanced goal statistics including both enhanced and final counts
        """
        # Calculate average confidence score
        avg_confidence = (
            np.mean(self.goal_confidence_scores) if self.goal_confidence_scores else 0.0
        )

        return {
            # Enhanced detection results (real-time)
            "team_goals": self.team_goals.copy(),
            "player_goals": self.player_goals.copy(),
            # Final validated counts (primary metric)
            "final_team_goals": self.final_team_goals.copy(),
            "final_player_goals": self.final_player_goals.copy(),
            # Comprehensive event data
            "goal_events": self.goal_events.copy(),
            "validation_history": self.goal_validation_history.copy(),
            # Summary statistics
            "total_goals": sum(self.team_goals.values()),
            "final_total_goals": sum(self.final_team_goals.values()),
            "average_confidence": avg_confidence,
            "detection_accuracy": len(
                [v for v in self.goal_validation_history if v["is_valid"]]
            )
            / max(1, len(self.goal_validation_history)),
        }

    def set_final_goal_counts(self, team_goals, player_goals):
        """
        Set final goal counts (used when manual goals override detection).

        Args:
            team_goals (dict): Final team goal counts
            player_goals (dict): Final player goal counts
        """
        self.final_team_goals = team_goals.copy()
        self.final_player_goals = player_goals.copy()

        print(f"📊 Final goal counts updated:")
        print(
            f"   Team 1: {self.final_team_goals[1]}, Team 2: {self.final_team_goals[2]}"
        )
        print(f"   Player goals: {len(self.final_player_goals)} players scored")

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
        self.final_team_goals = {1: 0, 2: 0}
        self.final_player_goals = {}
        self.goal_events = []
        self.goal_validation_history = []
        self.goal_confidence_scores = []
        self.goal_detected = False
        self.goal_cooldown = 0
        self.ball_trajectory = []
