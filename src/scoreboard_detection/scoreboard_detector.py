"""
Scoreboard Detector

This module implements computer vision techniques to detect scoreboard regions
in football/soccer video frames.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ScoreboardDetector:
    """
    Detects scoreboard regions in video frames using multiple computer vision techniques.

    The detector uses a combination of:
    1. Text region detection using MSER (Maximally Stable Extremal Regions)
    2. Contour analysis for rectangular regions
    3. Color analysis for typical scoreboard backgrounds
    4. Template matching for common scoreboard layouts
    """

    def __init__(
        self,
        min_scoreboard_width: int = 200,
        min_scoreboard_height: int = 50,
        max_scoreboard_width: int = 800,
        max_scoreboard_height: int = 200,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the scoreboard detector.

        Args:
            min_scoreboard_width: Minimum width for scoreboard detection
            min_scoreboard_height: Minimum height for scoreboard detection
            max_scoreboard_width: Maximum width for scoreboard detection
            max_scoreboard_height: Maximum height for scoreboard detection
            confidence_threshold: Minimum confidence for detection
        """
        self.min_width = min_scoreboard_width
        self.min_height = min_scoreboard_height
        self.max_width = max_scoreboard_width
        self.max_height = max_scoreboard_height
        self.confidence_threshold = confidence_threshold

        # Initialize MSER detector for text regions
        self.mser = cv2.MSER_create(100, 14400, 5)  # min_area  # max_area  # delta

        # Common scoreboard colors (BGR format)
        self.scoreboard_colors = [
            ([0, 0, 0], [50, 50, 50]),  # Black/dark backgrounds
            ([200, 200, 200], [255, 255, 255]),  # White/light backgrounds
            ([0, 50, 100], [50, 150, 255]),  # Red backgrounds
            ([100, 100, 0], [255, 255, 100]),  # Yellow backgrounds
            ([0, 100, 0], [100, 255, 100]),  # Green backgrounds
        ]

        # Detection history for temporal consistency
        self.detection_history = []
        self.max_history_length = 10

        self.logger = logging.getLogger(__name__)

    def detect_scoreboard_regions(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect potential scoreboard regions in a frame.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detected regions with confidence scores
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        # Focus on upper portion of frame where scoreboards typically appear
        upper_region = frame[: height // 3, :]

        # Detect using multiple methods
        text_regions = self._detect_text_regions(upper_region)
        contour_regions = self._detect_rectangular_regions(upper_region)
        color_regions = self._detect_color_regions(upper_region)

        # Combine and filter detections
        all_regions = text_regions + contour_regions + color_regions
        filtered_regions = self._filter_and_merge_regions(
            all_regions, upper_region.shape[:2]
        )

        # Add temporal consistency
        consistent_regions = self._apply_temporal_filtering(filtered_regions)

        return consistent_regions

    def _detect_text_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect regions containing text using MSER."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect MSER regions
        regions, _ = self.mser.detectRegions(gray)

        text_regions = []
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

            # Filter by size
            if (
                self.min_width <= w <= self.max_width
                and self.min_height <= h <= self.max_height
            ):

                confidence = self._calculate_text_confidence(gray[y : y + h, x : x + w])

                if confidence > self.confidence_threshold:
                    text_regions.append(
                        {"bbox": (x, y, w, h), "confidence": confidence, "type": "text"}
                    )

        return text_regions

    def _detect_rectangular_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect rectangular regions that could be scoreboards."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangular_regions = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size and aspect ratio
                if (
                    self.min_width <= w <= self.max_width
                    and self.min_height <= h <= self.max_height
                    and 2 <= w / h <= 8
                ):  # Typical scoreboard aspect ratio

                    confidence = self._calculate_rectangle_confidence(contour, (w, h))

                    if confidence > self.confidence_threshold:
                        rectangular_regions.append(
                            {
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                                "type": "rectangle",
                            }
                        )

        return rectangular_regions

    def _detect_color_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect regions with typical scoreboard colors."""
        color_regions = []

        for lower_color, upper_color in self.scoreboard_colors:
            # Create mask for this color range
            mask = cv2.inRange(frame, np.array(lower_color), np.array(upper_color))

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size
                if (
                    self.min_width <= w <= self.max_width
                    and self.min_height <= h <= self.max_height
                ):

                    confidence = self._calculate_color_confidence(
                        mask[y : y + h, x : x + w]
                    )

                    if confidence > self.confidence_threshold:
                        color_regions.append(
                            {
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                                "type": "color",
                            }
                        )

        return color_regions

    def _calculate_text_confidence(self, region: np.ndarray) -> float:
        """Calculate confidence score for text regions."""
        if region.size == 0:
            return 0.0

        # Check for text-like characteristics
        # 1. Edge density
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 2. Contrast
        contrast = np.std(region) / 255.0

        # 3. Horizontal structure (text typically has horizontal patterns)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / horizontal_lines.size

        # Combine scores
        confidence = edge_density * 0.4 + contrast * 0.3 + horizontal_score * 0.3
        return min(confidence, 1.0)

    def _calculate_rectangle_confidence(
        self, contour: np.ndarray, size: Tuple[int, int]
    ) -> float:
        """Calculate confidence score for rectangular regions."""
        w, h = size

        # 1. Rectangularity (how close to a perfect rectangle)
        area = cv2.contourArea(contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0

        # 2. Aspect ratio score (scoreboards are typically wider than tall)
        aspect_ratio = w / h
        aspect_score = 1.0 if 2 <= aspect_ratio <= 8 else 0.5

        # 3. Size score (prefer medium-sized regions)
        size_score = min(w / self.max_width, h / self.max_height)

        confidence = rectangularity * 0.5 + aspect_score * 0.3 + size_score * 0.2
        return min(confidence, 1.0)

    def _calculate_color_confidence(self, mask: np.ndarray) -> float:
        """Calculate confidence score for color regions."""
        if mask.size == 0:
            return 0.0

        # Calculate fill ratio
        fill_ratio = np.sum(mask > 0) / mask.size

        # Prefer regions that are well-filled but not completely filled
        if 0.3 <= fill_ratio <= 0.8:
            return fill_ratio
        else:
            return fill_ratio * 0.5

    def _filter_and_merge_regions(
        self, regions: List[Dict], frame_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Filter overlapping regions and merge similar ones."""
        if not regions:
            return []

        # Sort by confidence
        regions.sort(key=lambda x: x["confidence"], reverse=True)

        filtered_regions = []
        for region in regions:
            # Check for overlap with existing regions
            overlaps = False
            for existing in filtered_regions:
                if self._calculate_overlap(region["bbox"], existing["bbox"]) > 0.5:
                    overlaps = True
                    break

            if not overlaps:
                filtered_regions.append(region)

        return filtered_regions

    def _calculate_overlap(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _apply_temporal_filtering(self, current_regions: List[Dict]) -> List[Dict]:
        """Apply temporal filtering for consistent detection."""
        # Add current detections to history
        self.detection_history.append(current_regions)

        # Keep only recent history
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)

        # For now, just return current regions
        # In a more sophisticated implementation, we could track regions across frames
        return current_regions
