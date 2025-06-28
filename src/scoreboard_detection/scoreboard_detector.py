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
        min_scoreboard_width: int = 100,  # Reduced minimum width
        min_scoreboard_height: int = 30,  # Reduced minimum height
        max_scoreboard_width: int = 1200,  # Increased maximum width
        max_scoreboard_height: int = 300,  # Increased maximum height
        confidence_threshold: float = 0.4,  # Lowered threshold for better detection
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

        # Initialize multiple MSER detectors with different parameters for better coverage
        self.mser_detectors = [
            cv2.MSER_create(50, 10000, 3),  # More sensitive detector
            cv2.MSER_create(100, 14400, 5),  # Original detector
            cv2.MSER_create(200, 20000, 8),  # Less sensitive detector
        ]

        # Expanded scoreboard colors (BGR format) with more variations
        self.scoreboard_colors = [
            ([0, 0, 0], [80, 80, 80]),  # Black/dark backgrounds (expanded range)
            (
                [180, 180, 180],
                [255, 255, 255],
            ),  # White/light backgrounds (expanded range)
            ([0, 30, 80], [80, 180, 255]),  # Red/orange backgrounds (expanded)
            ([80, 80, 0], [255, 255, 120]),  # Yellow backgrounds (expanded)
            ([0, 80, 0], [120, 255, 120]),  # Green backgrounds (expanded)
            ([80, 0, 0], [255, 120, 120]),  # Blue backgrounds
            ([40, 40, 40], [120, 120, 120]),  # Gray backgrounds
            ([0, 0, 100], [100, 100, 255]),  # Red variations
            ([100, 0, 0], [255, 100, 100]),  # Blue variations
        ]

        # Detection history for temporal consistency
        self.detection_history = []
        self.max_history_length = 15  # Increased history length

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

        # Search in multiple regions where scoreboards might appear
        search_regions = [
            ("upper", frame[: height // 2, :]),  # Upper half (most common)
            ("top_strip", frame[: height // 4, :]),  # Top quarter
            ("center_strip", frame[height // 4 : 3 * height // 4, :]),  # Center strip
            ("full", frame),  # Full frame as fallback
        ]

        all_text_regions = []
        all_contour_regions = []
        all_color_regions = []

        for region_name, region in search_regions:
            if region.size == 0:
                continue

            # Detect using multiple methods in each region
            text_regions = self._detect_text_regions(region)
            contour_regions = self._detect_rectangular_regions(region)
            color_regions = self._detect_color_regions(region)

            # Adjust coordinates back to full frame
            if region_name == "center_strip":
                offset_y = height // 4
                for regions_list in [text_regions, contour_regions, color_regions]:
                    for region_dict in regions_list:
                        x, y, w, h = region_dict["bbox"]
                        region_dict["bbox"] = (x, y + offset_y, w, h)

            all_text_regions.extend(text_regions)
            all_contour_regions.extend(contour_regions)
            all_color_regions.extend(color_regions)

        # Combine and filter detections
        all_regions = all_text_regions + all_contour_regions + all_color_regions
        filtered_regions = self._filter_and_merge_regions(all_regions, frame.shape[:2])

        # Add temporal consistency
        consistent_regions = self._apply_temporal_filtering(filtered_regions)

        return consistent_regions

    def _detect_text_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect regions containing text using multiple MSER detectors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)

        text_regions = []
        all_regions = []

        # Use multiple MSER detectors for better coverage
        for mser_detector in self.mser_detectors:
            try:
                # Detect MSER regions on both original and equalized images
                for img in [gray, equalized]:
                    regions, _ = mser_detector.detectRegions(img)
                    all_regions.extend(regions)
            except Exception as e:
                self.logger.debug(f"MSER detection failed: {e}")
                continue

        # Process all detected regions
        for region in all_regions:
            try:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

                # Filter by size with more flexible constraints
                if (
                    self.min_width <= w <= self.max_width
                    and self.min_height <= h <= self.max_height
                    and w > h  # Scoreboards are typically wider than tall
                    and 1.5 <= w / h <= 15  # Reasonable aspect ratio range
                ):
                    # Extract region for confidence calculation
                    region_img = gray[y : y + h, x : x + w]
                    confidence = self._calculate_text_confidence(region_img)

                    if confidence > self.confidence_threshold:
                        text_regions.append(
                            {
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                                "type": "text",
                            }
                        )
            except Exception as e:
                self.logger.debug(f"Region processing failed: {e}")
                continue

        return text_regions

    def _detect_rectangular_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect rectangular regions that could be scoreboards."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        rectangular_regions = []

        # Try multiple edge detection parameters
        edge_params = [
            (30, 100),  # Lower thresholds for subtle edges
            (50, 150),  # Original parameters
            (80, 200),  # Higher thresholds for strong edges
        ]

        for low_thresh, high_thresh in edge_params:
            # Edge detection
            edges = cv2.Canny(blurred, low_thresh, high_thresh, apertureSize=3)

            # Apply morphological operations to connect broken edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Skip very small contours
                if cv2.contourArea(contour) < 500:
                    continue

                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's roughly rectangular (4 corners) or has reasonable shape
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Filter by size and aspect ratio with more flexible constraints
                    if (
                        self.min_width <= w <= self.max_width
                        and self.min_height <= h <= self.max_height
                        and w > h  # Scoreboards are wider than tall
                        and 1.5 <= w / h <= 12  # More flexible aspect ratio
                    ):
                        confidence = self._calculate_rectangle_confidence(
                            contour, (w, h)
                        )

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
