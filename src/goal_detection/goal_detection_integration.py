#!/usr/bin/env python3
"""
Goal Detection Integration Module

This module provides integration between the comprehensive goal detection system
and the main video processing pipeline. It handles:
1. Integration with existing tracker and detection systems
2. Frame-by-frame processing coordination
3. CSV output generation
4. Performance optimization for real-time processing
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .comprehensive_goal_detector import ComprehensiveGoalDetector
from .field_keypoints_detector import FieldKeypointsDetector
from .goal_csv_generator import GoalCSVGenerator


class GoalDetectionIntegrator:
    """
    Integrates comprehensive goal detection with the main video processing pipeline.

    This class serves as the main interface between the goal detection system
    and the existing video analysis pipeline.
    """

    def __init__(
        self,
        field_keypoints_model_path: str = "data/models/best_field_keypoint.pt",
        output_dir: str = "data/output",
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the goal detection integrator.

        Args:
            field_keypoints_model_path: Path to field keypoints detection model
            output_dir: Directory for CSV output files
            device: Device for model inference
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.field_keypoints_detector = FieldKeypointsDetector(
            field_keypoints_model_path, device=device
        )
        self.csv_generator = GoalCSVGenerator(output_dir)
        self.comprehensive_detector = ComprehensiveGoalDetector(
            self.field_keypoints_detector, self.csv_generator, config
        )

        # State tracking
        self.video_name = None
        self.processing_active = False

    def start_video_processing(self, video_name: str):
        """Start processing a new video."""
        self.video_name = video_name
        self.processing_active = True
        self.comprehensive_detector.reset()
        self.logger.info(f"Started goal detection for video: {video_name}")

    def process_frame_with_detections(
        self,
        frame_num: int,
        frame: Any,
        ball_detections: List[Dict[str, Any]],
        player_detections: List[Dict[str, Any]],
        tracks: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Process a frame with existing detection results.

        Args:
            frame_num: Current frame number
            frame: Video frame for keypoint detection
            ball_detections: Ball detection results
            player_detections: Player detection results with tracking IDs
            tracks: Optional tracking data

        Returns:
            True if goal detected in this frame
        """
        if not self.processing_active:
            return False

        # Update field keypoints detection (only if frame is provided)
        if frame is not None:
            self.field_keypoints_detector.detect_keypoints(frame)

        # Extract ball information
        ball_position = None
        ball_confidence = 0.0

        if ball_detections:
            # Use the highest confidence ball detection
            best_ball = max(ball_detections, key=lambda x: x.get("confidence", 0))
            if "bbox" in best_ball:
                bbox = best_ball["bbox"]
                ball_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            elif "position" in best_ball:
                ball_position = best_ball["position"]
            ball_confidence = best_ball.get("confidence", 0.0)

        # Process player detections to extract required information
        processed_players = []
        for player in player_detections:
            player_info = {
                "player_id": player.get("track_id", player.get("player_id")),
                "team": player.get("team"),
                "bbox": player.get("bbox"),
                "position": player.get("position"),
                "confidence": player.get("confidence", 1.0),
            }

            # Extract position from bbox if not provided
            if not player_info["position"] and player_info["bbox"]:
                bbox = player_info["bbox"]
                player_info["position"] = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                )

            processed_players.append(player_info)

        # Process frame through comprehensive detector
        goal_detected = self.comprehensive_detector.process_frame(
            frame_num=frame_num,
            ball_position=ball_position,
            ball_confidence=ball_confidence,
            player_detections=processed_players,
            frame_data=frame,
        )

        return goal_detected

    def process_frame_with_tracks(
        self, frame_num: int, frame: Any, tracks: Dict[str, Any]
    ) -> bool:
        """
        Process a frame using tracking data from the main pipeline.

        Args:
            frame_num: Current frame number
            frame: Video frame
            tracks: Tracking data containing ball and player information

        Returns:
            True if goal detected in this frame
        """
        if not self.processing_active:
            return False

        # Extract ball detections from tracks
        ball_detections = []
        if "ball" in tracks and frame_num < len(tracks["ball"]):
            ball_track = tracks["ball"][frame_num]
            if ball_track:
                ball_detections = [{"bbox": ball_track, "confidence": 0.8}]

        # Extract player detections from tracks
        player_detections = []
        if "players" in tracks and frame_num < len(tracks["players"]):
            player_tracks = tracks["players"][frame_num]
            for track_id, bbox in player_tracks.items():
                if bbox:
                    player_detections.append(
                        {"track_id": track_id, "bbox": bbox, "confidence": 0.8}
                    )

        return self.process_frame_with_detections(
            frame_num, frame, ball_detections, player_detections, tracks
        )

    def finalize_video_processing(self) -> Tuple[str, str]:
        """
        Finalize video processing and generate CSV outputs.

        Returns:
            Tuple of (frame_csv_path, scoreboard_csv_path)
        """
        if not self.processing_active or not self.video_name:
            raise ValueError("No active video processing session")

        # Generate CSV outputs
        frame_csv, scoreboard_csv = self.comprehensive_detector.generate_csv_outputs(
            self.video_name
        )

        # Log statistics
        stats = self.comprehensive_detector.get_statistics()
        self.logger.info(f"Goal detection completed for {self.video_name}")
        self.logger.info(f"Total goals detected: {stats['total_goals_detected']}")
        self.logger.info(f"Frames processed: {stats['frames_processed']}")
        self.logger.info(f"Generated CSV files: {frame_csv}, {scoreboard_csv}")

        self.processing_active = False
        return frame_csv, scoreboard_csv

    def get_statistics(self) -> Dict[str, Any]:
        """Get current goal detection statistics."""
        return self.comprehensive_detector.get_statistics()

    def get_goal_events(self) -> List[Dict[str, Any]]:
        """Get list of detected goal events."""
        return self.csv_generator.goal_events

    def is_processing_active(self) -> bool:
        """Check if video processing is active."""
        return self.processing_active


def create_goal_detection_integrator(
    field_keypoints_model_path: str = "data/models/best_field_keypoint.pt",
    output_dir: str = "data/output",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> GoalDetectionIntegrator:
    """
    Factory function to create a goal detection integrator.

    Args:
        field_keypoints_model_path: Path to field keypoints model
        output_dir: Output directory for CSV files
        device: Device for model inference
        config: Configuration parameters

    Returns:
        Configured GoalDetectionIntegrator instance
    """
    return GoalDetectionIntegrator(
        field_keypoints_model_path=field_keypoints_model_path,
        output_dir=output_dir,
        device=device,
        config=config,
    )


def integrate_with_existing_pipeline(
    integrator: GoalDetectionIntegrator,
    video_path: str,
    tracks: Dict[str, Any],
    video_frames: Any,
    batch_size: int = 50,
) -> Tuple[str, str]:
    """
    Integrate goal detection with existing video processing pipeline.

    Args:
        integrator: Goal detection integrator instance
        video_path: Path to video file
        tracks: Tracking data from main pipeline
        video_frames: Video frame iterator
        batch_size: Batch size for processing

    Returns:
        Tuple of (frame_csv_path, scoreboard_csv_path)
    """
    video_name = Path(video_path).stem
    integrator.start_video_processing(video_name)

    # Process frames in batches
    frame_count = 0
    total_frames = len(tracks.get("players", []))

    for frame_num in range(total_frames):
        # Get frame data
        frame = None
        if hasattr(video_frames, "__getitem__"):
            try:
                frame = video_frames[frame_num]
            except (IndexError, KeyError):
                frame = None

        # Process frame
        goal_detected = integrator.process_frame_with_tracks(frame_num, frame, tracks)

        if goal_detected:
            logging.info(f"Goal detected at frame {frame_num}")

        frame_count += 1

        # Log progress periodically
        if frame_count % batch_size == 0:
            logging.info(
                f"Processed {frame_count}/{total_frames} frames for goal detection"
            )

    # Finalize and generate CSV outputs
    return integrator.finalize_video_processing()


# Configuration presets
DEFAULT_CONFIG = {
    "goal_cooldown_frames": 90,
    "kick_detection_distance": 50,
    "kick_detection_frames": 5,
    "sequence_validation_frames": 15,
    "min_ball_confidence": 0.3,
    "min_goal_confidence": 0.5,
    "player_ball_max_distance": 100,
    "ball_speed_threshold": 10,
    "goal_area_buffer": 20,
}

HIGH_ACCURACY_CONFIG = {
    **DEFAULT_CONFIG,
    "min_ball_confidence": 0.5,
    "min_goal_confidence": 0.7,
    "sequence_validation_frames": 20,
    "kick_detection_frames": 8,
}

FAST_PROCESSING_CONFIG = {
    **DEFAULT_CONFIG,
    "min_ball_confidence": 0.2,
    "min_goal_confidence": 0.4,
    "sequence_validation_frames": 10,
    "kick_detection_frames": 3,
}
