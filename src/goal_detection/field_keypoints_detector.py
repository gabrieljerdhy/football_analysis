import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append("../")


class FieldKeypointsDetector:
    """
    Detects field keypoints using YOLO model and provides goal area detection capabilities.
    """

    def __init__(self, model_path="models/best_keypoint.pt", confidence_threshold=0.7):
        """
        Initialize the field keypoints detector with optimization features.

        Args:
            model_path (str): Path to the YOLO field keypoints model
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Optimization parameters
        self.detection_interval = 5  # Detect keypoints every N frames
        self.frame_count = 0
        self.stable_keypoints_threshold = 10  # Frames to consider keypoints stable
        self.stable_keypoints_count = 0
        self.last_detection_frame = -1

        # Keypoint stability tracking
        self.keypoint_stability = {}
        self.stable_areas_cache = {
            "goal_areas": {"left": None, "right": None},
            "penalty_areas": {"left": None, "right": None},
            "six_yard_areas": {"left": None, "right": None},
            "goal_lines": {"left": None, "right": None},
        }

        # Enhanced football field keypoints mapping
        # These represent the typical keypoints that should be detected
        self.keypoint_classes = {
            0: "goal_left_post_top",
            1: "goal_left_post_bottom",
            2: "goal_right_post_top",
            3: "goal_right_post_bottom",
            4: "penalty_area_left_top",
            5: "penalty_area_left_bottom",
            6: "penalty_area_right_top",
            7: "penalty_area_right_bottom",
            8: "center_circle_top",
            9: "center_circle_bottom",
            10: "center_circle_left",
            11: "center_circle_right",
            12: "halfway_line_top",
            13: "halfway_line_bottom",
            # Additional keypoints for enhanced detection
            14: "goal_line_left",
            15: "goal_line_right",
            16: "penalty_area_left_left",
            17: "penalty_area_left_right",
            18: "penalty_area_right_left",
            19: "penalty_area_right_right",
            20: "six_yard_box_left_top",
            21: "six_yard_box_left_bottom",
            22: "six_yard_box_right_top",
            23: "six_yard_box_right_bottom",
        }

        # Enhanced goal-related keypoints for each goal
        self.left_goal_keypoints = [
            "goal_left_post_top",
            "goal_left_post_bottom",
            "goal_line_left",
        ]
        self.right_goal_keypoints = [
            "goal_right_post_top",
            "goal_right_post_bottom",
            "goal_line_right",
        ]

        # Penalty area keypoints for enhanced goal validation
        self.left_penalty_keypoints = [
            "penalty_area_left_top",
            "penalty_area_left_bottom",
            "penalty_area_left_left",
            "penalty_area_left_right",
        ]
        self.right_penalty_keypoints = [
            "penalty_area_right_top",
            "penalty_area_right_bottom",
            "penalty_area_right_left",
            "penalty_area_right_right",
        ]

        # Six-yard box keypoints for precise goal area detection
        self.left_six_yard_keypoints = [
            "six_yard_box_left_top",
            "six_yard_box_left_bottom",
        ]
        self.right_six_yard_keypoints = [
            "six_yard_box_right_top",
            "six_yard_box_right_bottom",
        ]

        # Cache for detected keypoints and enhanced areas
        self.detected_keypoints = {}
        self.goal_areas = {"left": None, "right": None}
        self.penalty_areas = {"left": None, "right": None}
        self.six_yard_areas = {"left": None, "right": None}
        self.goal_lines = {"left": None, "right": None}

    def detect_keypoints(self, frame, force_detection=False):
        """
        Optimized field keypoints detection with caching and interval-based processing.

        Args:
            frame: Input video frame
            force_detection (bool): Force detection even if not at interval

        Returns:
            dict: Dictionary of detected keypoints with their positions
        """
        self.frame_count += 1

        # Check if we should perform detection based on optimization strategy
        should_detect = (
            force_detection
            or self.frame_count % self.detection_interval == 0
            or not self._has_stable_keypoints()
        )

        if should_detect:
            detected_keypoints = self._perform_keypoint_detection(frame)
            self._update_keypoint_stability(detected_keypoints)
            self.last_detection_frame = self.frame_count
        else:
            # Use cached keypoints for performance
            detected_keypoints = self.detected_keypoints.copy()

        return detected_keypoints

    def _perform_keypoint_detection(self, frame):
        """
        Perform actual keypoint detection on the frame.
        """
        results = self.model(frame, conf=self.confidence_threshold)

        detected_keypoints = {}

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for i, (box, conf, class_id) in enumerate(
                zip(boxes, confidences, class_ids)
            ):
                if conf >= self.confidence_threshold:
                    class_id = int(class_id)
                    if class_id in self.keypoint_classes:
                        keypoint_name = self.keypoint_classes[class_id]

                        # Get center point of the bounding box
                        x_center = int((box[0] + box[2]) / 2)
                        y_center = int((box[1] + box[3]) / 2)

                        detected_keypoints[keypoint_name] = {
                            "position": (x_center, y_center),
                            "bbox": box,
                            "confidence": conf,
                        }

        self.detected_keypoints = detected_keypoints
        self._update_enhanced_areas()
        return detected_keypoints

    def _has_stable_keypoints(self):
        """
        Check if keypoints have been stable for enough frames.
        """
        return self.stable_keypoints_count >= self.stable_keypoints_threshold

    def _update_keypoint_stability(self, detected_keypoints):
        """
        Update keypoint stability tracking.
        """
        # Check if current keypoints are similar to previous ones
        if self._are_keypoints_stable(detected_keypoints):
            self.stable_keypoints_count += 1
        else:
            self.stable_keypoints_count = 0

        # Update stability cache if keypoints are stable
        if self._has_stable_keypoints():
            self._update_stable_cache()

    def _are_keypoints_stable(self, current_keypoints):
        """
        Check if current keypoints are stable compared to previous detection.
        """
        if not self.detected_keypoints:
            return False

        # Compare positions of key goal-related keypoints
        stability_threshold = 20  # pixels
        key_keypoints = [
            "goal_left_post_top",
            "goal_left_post_bottom",
            "goal_right_post_top",
            "goal_right_post_bottom",
        ]

        stable_count = 0
        total_count = 0

        for keypoint in key_keypoints:
            if keypoint in current_keypoints and keypoint in self.detected_keypoints:
                current_pos = current_keypoints[keypoint]["position"]
                previous_pos = self.detected_keypoints[keypoint]["position"]

                distance = (
                    (current_pos[0] - previous_pos[0]) ** 2
                    + (current_pos[1] - previous_pos[1]) ** 2
                ) ** 0.5

                if distance <= stability_threshold:
                    stable_count += 1
                total_count += 1

        # Consider stable if at least 75% of key keypoints are stable
        return total_count > 0 and (stable_count / total_count) >= 0.75

    def _update_stable_cache(self):
        """
        Update the stable areas cache with current areas.
        """
        self.stable_areas_cache = {
            "goal_areas": self.goal_areas.copy(),
            "penalty_areas": self.penalty_areas.copy(),
            "six_yard_areas": self.six_yard_areas.copy(),
            "goal_lines": self.goal_lines.copy(),
        }

    def _update_enhanced_areas(self):
        """
        Update goal areas, penalty areas, and other field boundaries based on detected keypoints.
        """
        self._update_goal_areas()
        self._update_penalty_areas()
        self._update_six_yard_areas()
        self._update_goal_lines()

    def _update_goal_areas(self):
        """
        Update goal area boundaries based on detected keypoints with enhanced logic.
        """
        # Left goal area - prioritize six-yard box, then goal posts
        left_goal_points = []

        # First try six-yard box keypoints for more precise goal area
        for keypoint in self.left_six_yard_keypoints:
            if keypoint in self.detected_keypoints:
                left_goal_points.append(self.detected_keypoints[keypoint]["position"])

        # If no six-yard box, use goal posts
        if len(left_goal_points) < 2:
            left_goal_points = []
            for keypoint in self.left_goal_keypoints:
                if keypoint in self.detected_keypoints:
                    left_goal_points.append(
                        self.detected_keypoints[keypoint]["position"]
                    )

        if len(left_goal_points) >= 2:
            # Calculate goal area boundaries with enhanced logic
            x_coords = [p[0] for p in left_goal_points]
            y_coords = [p[1] for p in left_goal_points]

            # Use smaller margin for more precise detection
            margin_x = 30  # Reduced margin for better accuracy
            margin_y = 10

            self.goal_areas["left"] = {
                "x_min": min(x_coords) - margin_x,
                "x_max": max(x_coords) + margin_x,
                "y_min": min(y_coords) - margin_y,
                "y_max": max(y_coords) + margin_y,
                "keypoints_used": [
                    kp
                    for kp in (
                        self.left_six_yard_keypoints
                        if len(
                            [
                                kp
                                for kp in self.left_six_yard_keypoints
                                if kp in self.detected_keypoints
                            ]
                        )
                        >= 2
                        else self.left_goal_keypoints
                    )
                    if kp in self.detected_keypoints
                ],
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"]
                    for kp in self.goal_areas["left"]["keypoints_used"]
                    if kp in self.detected_keypoints
                )
                / max(
                    1,
                    len(
                        [
                            kp
                            for kp in (
                                self.left_six_yard_keypoints
                                if len(
                                    [
                                        kp
                                        for kp in self.left_six_yard_keypoints
                                        if kp in self.detected_keypoints
                                    ]
                                )
                                >= 2
                                else self.left_goal_keypoints
                            )
                            if kp in self.detected_keypoints
                        ]
                    ),
                ),
            }

        # Right goal area - same logic
        right_goal_points = []

        # First try six-yard box keypoints
        for keypoint in self.right_six_yard_keypoints:
            if keypoint in self.detected_keypoints:
                right_goal_points.append(self.detected_keypoints[keypoint]["position"])

        # If no six-yard box, use goal posts
        if len(right_goal_points) < 2:
            right_goal_points = []
            for keypoint in self.right_goal_keypoints:
                if keypoint in self.detected_keypoints:
                    right_goal_points.append(
                        self.detected_keypoints[keypoint]["position"]
                    )

        if len(right_goal_points) >= 2:
            x_coords = [p[0] for p in right_goal_points]
            y_coords = [p[1] for p in right_goal_points]

            margin_x = 30
            margin_y = 10

            keypoints_used = [
                kp
                for kp in (
                    self.right_six_yard_keypoints
                    if len(
                        [
                            kp
                            for kp in self.right_six_yard_keypoints
                            if kp in self.detected_keypoints
                        ]
                    )
                    >= 2
                    else self.right_goal_keypoints
                )
                if kp in self.detected_keypoints
            ]

            self.goal_areas["right"] = {
                "x_min": min(x_coords) - margin_x,
                "x_max": max(x_coords) + margin_x,
                "y_min": min(y_coords) - margin_y,
                "y_max": max(y_coords) + margin_y,
                "keypoints_used": keypoints_used,
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"] for kp in keypoints_used
                )
                / max(1, len(keypoints_used)),
            }

    def _update_penalty_areas(self):
        """
        Update penalty area boundaries based on detected keypoints.
        """
        # Left penalty area
        left_penalty_points = []
        for keypoint in self.left_penalty_keypoints:
            if keypoint in self.detected_keypoints:
                left_penalty_points.append(
                    self.detected_keypoints[keypoint]["position"]
                )

        if len(left_penalty_points) >= 2:
            x_coords = [p[0] for p in left_penalty_points]
            y_coords = [p[1] for p in left_penalty_points]

            self.penalty_areas["left"] = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "keypoints_used": [
                    kp
                    for kp in self.left_penalty_keypoints
                    if kp in self.detected_keypoints
                ],
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"]
                    for kp in self.left_penalty_keypoints
                    if kp in self.detected_keypoints
                )
                / max(
                    1,
                    len(
                        [
                            kp
                            for kp in self.left_penalty_keypoints
                            if kp in self.detected_keypoints
                        ]
                    ),
                ),
            }

        # Right penalty area
        right_penalty_points = []
        for keypoint in self.right_penalty_keypoints:
            if keypoint in self.detected_keypoints:
                right_penalty_points.append(
                    self.detected_keypoints[keypoint]["position"]
                )

        if len(right_penalty_points) >= 2:
            x_coords = [p[0] for p in right_penalty_points]
            y_coords = [p[1] for p in right_penalty_points]

            keypoints_used = [
                kp
                for kp in self.right_penalty_keypoints
                if kp in self.detected_keypoints
            ]

            self.penalty_areas["right"] = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "keypoints_used": keypoints_used,
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"] for kp in keypoints_used
                )
                / max(1, len(keypoints_used)),
            }

    def _update_six_yard_areas(self):
        """
        Update six-yard box boundaries based on detected keypoints.
        """
        # Left six-yard area
        left_six_yard_points = []
        for keypoint in self.left_six_yard_keypoints:
            if keypoint in self.detected_keypoints:
                left_six_yard_points.append(
                    self.detected_keypoints[keypoint]["position"]
                )

        if len(left_six_yard_points) >= 2:
            x_coords = [p[0] for p in left_six_yard_points]
            y_coords = [p[1] for p in left_six_yard_points]

            keypoints_used = [
                kp
                for kp in self.left_six_yard_keypoints
                if kp in self.detected_keypoints
            ]

            self.six_yard_areas["left"] = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "keypoints_used": keypoints_used,
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"] for kp in keypoints_used
                )
                / max(1, len(keypoints_used)),
            }

        # Right six-yard area
        right_six_yard_points = []
        for keypoint in self.right_six_yard_keypoints:
            if keypoint in self.detected_keypoints:
                right_six_yard_points.append(
                    self.detected_keypoints[keypoint]["position"]
                )

        if len(right_six_yard_points) >= 2:
            x_coords = [p[0] for p in right_six_yard_points]
            y_coords = [p[1] for p in right_six_yard_points]

            keypoints_used = [
                kp
                for kp in self.right_six_yard_keypoints
                if kp in self.detected_keypoints
            ]

            self.six_yard_areas["right"] = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "keypoints_used": keypoints_used,
                "confidence": sum(
                    self.detected_keypoints[kp]["confidence"] for kp in keypoints_used
                )
                / max(1, len(keypoints_used)),
            }

    def _update_goal_lines(self):
        """
        Update goal line positions based on detected keypoints.
        """
        # Left goal line
        if "goal_line_left" in self.detected_keypoints:
            self.goal_lines["left"] = {
                "position": self.detected_keypoints["goal_line_left"]["position"],
                "confidence": self.detected_keypoints["goal_line_left"]["confidence"],
            }

        # Right goal line
        if "goal_line_right" in self.detected_keypoints:
            self.goal_lines["right"] = {
                "position": self.detected_keypoints["goal_line_right"]["position"],
                "confidence": self.detected_keypoints["goal_line_right"]["confidence"],
            }

    def get_goal_areas(self, use_cache=True):
        """
        Get the current goal area boundaries with optional caching.

        Args:
            use_cache (bool): Whether to use cached stable areas if available

        Returns:
            dict: Goal areas for left and right goals
        """
        if use_cache and self._has_stable_keypoints():
            return self.stable_areas_cache["goal_areas"]
        return self.goal_areas

    def get_penalty_areas(self, use_cache=True):
        """
        Get the current penalty area boundaries with optional caching.

        Args:
            use_cache (bool): Whether to use cached stable areas if available

        Returns:
            dict: Penalty areas for left and right sides
        """
        if use_cache and self._has_stable_keypoints():
            return self.stable_areas_cache["penalty_areas"]
        return self.penalty_areas

    def get_six_yard_areas(self, use_cache=True):
        """
        Get the current six-yard box boundaries with optional caching.

        Args:
            use_cache (bool): Whether to use cached stable areas if available

        Returns:
            dict: Six-yard areas for left and right sides
        """
        if use_cache and self._has_stable_keypoints():
            return self.stable_areas_cache["six_yard_areas"]
        return self.six_yard_areas

    def get_goal_lines(self, use_cache=True):
        """
        Get the current goal line positions with optional caching.

        Args:
            use_cache (bool): Whether to use cached stable areas if available

        Returns:
            dict: Goal line positions for left and right sides
        """
        if use_cache and self._has_stable_keypoints():
            return self.stable_areas_cache["goal_lines"]
        return self.goal_lines

    def set_optimization_parameters(
        self, detection_interval=None, stability_threshold=None
    ):
        """
        Set optimization parameters for keypoint detection.

        Args:
            detection_interval (int): Frames between keypoint detections
            stability_threshold (int): Frames needed to consider keypoints stable
        """
        if detection_interval is not None:
            self.detection_interval = detection_interval
        if stability_threshold is not None:
            self.stable_keypoints_threshold = stability_threshold

    def get_optimization_stats(self):
        """
        Get statistics about optimization performance.

        Returns:
            dict: Optimization statistics
        """
        detection_rate = 0.0
        if self.frame_count > 0:
            actual_detections = (self.frame_count // self.detection_interval) + 1
            detection_rate = actual_detections / self.frame_count

        return {
            "frame_count": self.frame_count,
            "detection_interval": self.detection_interval,
            "stable_keypoints_count": self.stable_keypoints_count,
            "has_stable_keypoints": self._has_stable_keypoints(),
            "last_detection_frame": self.last_detection_frame,
            "detection_rate": detection_rate,
            "frames_since_last_detection": self.frame_count - self.last_detection_frame,
        }

    def is_ball_in_goal_area(self, ball_position, goal_side="both"):
        """
        Enhanced check if ball is in a goal area using multiple validation methods.

        Args:
            ball_position (tuple): (x, y) position of the ball
            goal_side (str): "left", "right", or "both"

        Returns:
            str or None: Which goal the ball is in, or None if not in any goal
        """
        x, y = ball_position

        if goal_side in ["left", "both"] and self.goal_areas["left"]:
            left_area = self.goal_areas["left"]
            if (
                left_area["x_min"] <= x <= left_area["x_max"]
                and left_area["y_min"] <= y <= left_area["y_max"]
            ):
                return "left"

        if goal_side in ["right", "both"] and self.goal_areas["right"]:
            right_area = self.goal_areas["right"]
            if (
                right_area["x_min"] <= x <= right_area["x_max"]
                and right_area["y_min"] <= y <= right_area["y_max"]
            ):
                return "right"

        return None

    def is_ball_in_penalty_area(self, ball_position, goal_side="both"):
        """
        Check if ball is in a penalty area.

        Args:
            ball_position (tuple): (x, y) position of the ball
            goal_side (str): "left", "right", or "both"

        Returns:
            str or None: Which penalty area the ball is in, or None if not in any
        """
        x, y = ball_position

        if goal_side in ["left", "both"] and self.penalty_areas["left"]:
            left_area = self.penalty_areas["left"]
            if (
                left_area["x_min"] <= x <= left_area["x_max"]
                and left_area["y_min"] <= y <= left_area["y_max"]
            ):
                return "left"

        if goal_side in ["right", "both"] and self.penalty_areas["right"]:
            right_area = self.penalty_areas["right"]
            if (
                right_area["x_min"] <= x <= right_area["x_max"]
                and right_area["y_min"] <= y <= right_area["y_max"]
            ):
                return "right"

        return None

    def get_field_context(self, ball_position):
        """
        Get comprehensive field context for a ball position.

        Args:
            ball_position (tuple): (x, y) position of the ball

        Returns:
            dict: Field context information including areas and distances
        """
        context = {
            "in_goal_area": self.is_ball_in_goal_area(ball_position),
            "in_penalty_area": self.is_ball_in_penalty_area(ball_position),
            "distances_to_goals": {},
            "distances_to_penalty_areas": {},
            "field_confidence": 0.0,
        }

        # Calculate distances to goal areas
        for side in ["left", "right"]:
            if self.goal_areas[side]:
                goal_center_x = (
                    self.goal_areas[side]["x_min"] + self.goal_areas[side]["x_max"]
                ) / 2
                goal_center_y = (
                    self.goal_areas[side]["y_min"] + self.goal_areas[side]["y_max"]
                ) / 2
                distance = (
                    (ball_position[0] - goal_center_x) ** 2
                    + (ball_position[1] - goal_center_y) ** 2
                ) ** 0.5
                context["distances_to_goals"][side] = distance

            if self.penalty_areas[side]:
                penalty_center_x = (
                    self.penalty_areas[side]["x_min"]
                    + self.penalty_areas[side]["x_max"]
                ) / 2
                penalty_center_y = (
                    self.penalty_areas[side]["y_min"]
                    + self.penalty_areas[side]["y_max"]
                ) / 2
                distance = (
                    (ball_position[0] - penalty_center_x) ** 2
                    + (ball_position[1] - penalty_center_y) ** 2
                ) ** 0.5
                context["distances_to_penalty_areas"][side] = distance

        # Calculate overall field detection confidence
        total_confidence = 0.0
        confidence_count = 0

        for side in ["left", "right"]:
            if self.goal_areas[side] and "confidence" in self.goal_areas[side]:
                total_confidence += self.goal_areas[side]["confidence"]
                confidence_count += 1
            if self.penalty_areas[side] and "confidence" in self.penalty_areas[side]:
                total_confidence += self.penalty_areas[side]["confidence"]
                confidence_count += 1

        context["field_confidence"] = total_confidence / max(1, confidence_count)

        return context

    def draw_keypoints(self, frame):
        """
        Draw detected keypoints on the frame.

        Args:
            frame: Input frame to draw on

        Returns:
            frame: Frame with keypoints drawn
        """
        frame_copy = frame.copy()

        for keypoint_name, data in self.detected_keypoints.items():
            position = data["position"]
            confidence = data["confidence"]

            # Draw keypoint
            cv2.circle(frame_copy, position, 5, (0, 255, 0), -1)

            # Draw label
            label = f"{keypoint_name}: {confidence:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (position[0] + 10, position[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return frame_copy

    def draw_goal_areas(self, frame):
        """
        Draw goal areas on the frame.

        Args:
            frame: Input frame to draw on

        Returns:
            frame: Frame with goal areas drawn
        """
        frame_copy = frame.copy()

        # Draw left goal area
        if self.goal_areas["left"]:
            left_area = self.goal_areas["left"]
            cv2.rectangle(
                frame_copy,
                (left_area["x_min"], left_area["y_min"]),
                (left_area["x_max"], left_area["y_max"]),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame_copy,
                "Left Goal",
                (left_area["x_min"], left_area["y_min"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        # Draw right goal area
        if self.goal_areas["right"]:
            right_area = self.goal_areas["right"]
            cv2.rectangle(
                frame_copy,
                (right_area["x_min"], right_area["y_min"]),
                (right_area["x_max"], right_area["y_max"]),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame_copy,
                "Right Goal",
                (right_area["x_min"], right_area["y_min"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame_copy
