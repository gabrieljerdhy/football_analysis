import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

sys.path.append("../")
from src.jersey_number_detector import JerseyNumberDetector
from src.utils import get_bbox_width, get_center_of_bbox, get_foot_position


class Tracker:
    def __init__(self, model_path, enable_jersey_detection=True):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Initialize jersey number detector
        self.enable_jersey_detection = enable_jersey_detection
        if enable_jersey_detection:
            try:
                self.jersey_detector = JerseyNumberDetector(
                    confidence_threshold=0.4,
                    consensus_frames=3,
                    valid_number_range=(1, 99),
                )
                print("✅ Jersey number detection enabled")
            except Exception as e:
                print(f"⚠️ Jersey number detection disabled due to error: {e}")
                self.enable_jersey_detection = False
                self.jersey_detector = None
        else:
            self.jersey_detector = None

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        # Calculate positions for interpolated ball positions
        for ball_track in ball_positions:
            if 1 in ball_track and ball_track[1]:
                bbox = ball_track[1]["bbox"]
                position = get_center_of_bbox(bbox)
                ball_track[1]["position"] = position

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def detect_frames_memory_efficient(self, frames, progress_callback=None):
        """Memory-efficient frame detection with smaller batches and cleanup."""
        batch_size = min(10, len(frames))  # Smaller batches for memory efficiency
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            detections_batch = self.model.predict(batch_frames, conf=0.1)
            detections += detections_batch

            # Progress callback
            if progress_callback:
                progress_callback(i + len(batch_frames), len(frames))

            # Force cleanup after each batch
            import gc

            gc.collect()

        return detections

    def get_object_tracks_memory_efficient(
        self, frames, read_from_stub=False, stub_path=None, progress_callback=None
    ):
        """Memory-efficient version of get_object_tracks."""

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        # Use memory-efficient detection
        detections = self.detect_frames_memory_efficient(frames, progress_callback)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def add_jersey_numbers_to_tracks(self, tracks, video_frames, frame_sampling=3):
        """
        Add jersey numbers to player tracks using OCR detection.

        Args:
            tracks: Player tracking data
            video_frames: List of video frames
            frame_sampling: Process every N frames for performance (default: 3)
        """
        if not self.enable_jersey_detection or self.jersey_detector is None:
            # Add placeholder jersey numbers (track_id as jersey number)
            for frame_num, player_track in enumerate(tracks["players"]):
                for track_id, track_info in player_track.items():
                    track_info["jersey_number"] = track_id
            return

        print("🔍 Detecting jersey numbers...")
        processed_frames = 0

        # Process frames with sampling for performance
        for frame_num in range(0, len(tracks["players"]), frame_sampling):
            if frame_num >= len(video_frames):
                break

            frame = video_frames[frame_num]
            player_track = tracks["players"][frame_num]

            for track_id, track_info in player_track.items():
                bbox = track_info["bbox"]

                # Detect jersey number
                jersey_number = self.jersey_detector.detect_jersey_number(
                    frame, bbox, track_id
                )

                # Store detected number or use track_id as fallback
                if jersey_number is not None:
                    track_info["jersey_number"] = jersey_number
                else:
                    # Check if we have a cached number for this player
                    cached_number = self.jersey_detector.get_jersey_number_for_player(
                        track_id
                    )
                    track_info["jersey_number"] = (
                        cached_number if cached_number is not None else track_id
                    )

            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"  Processed {processed_frames} frames...")

        # Propagate detected jersey numbers to all frames
        self._propagate_jersey_numbers(tracks)

        # Print enhanced detection statistics
        if self.jersey_detector:
            stats = self.jersey_detector.get_detection_stats()
            validation_report = self.jersey_detector.get_validation_report()

            print(f"📊 Jersey detection stats:")
            print(f"  OCR calls: {stats['ocr_calls']}")
            print(f"  Cache hits: {stats['cache_hits']}")
            print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"  Confirmed players: {stats['confirmed_players']}")
            print(f"  Valid detection rate: {stats['valid_detection_rate']:.2%}")

            # Validation statistics
            val_stats = stats["validation_stats"]
            print(f"📋 Validation breakdown:")
            print(f"  Total OCR results: {val_stats['total_ocr_results']}")
            print(f"  Valid numbers: {val_stats['valid_numbers']}")
            print(f"  Invalid range: {val_stats['invalid_range']}")
            print(f"  Invalid format: {val_stats['invalid_format']}")
            print(f"  Multi-digit filtered: {val_stats['filtered_multi_digit']}")

            if stats["total_invalid_detections"] > 0:
                print(
                    f"⚠️ Total invalid detections: {stats['total_invalid_detections']}"
                )
                invalid_summary = validation_report["invalid_detections_summary"]
                for reason, count in invalid_summary.items():
                    if count > 0:
                        print(f"  {reason}: {count}")

    def _propagate_jersey_numbers(self, tracks):
        """
        Propagate detected jersey numbers to all frames for each player.
        """
        if not self.jersey_detector:
            return

        # Get all confirmed jersey numbers
        confirmed_numbers = {}
        for track_id in self.jersey_detector.player_jersey_cache:
            confirmed_numbers[track_id] = self.jersey_detector.player_jersey_cache[
                track_id
            ]

        # Apply to all frames
        for frame_num, player_track in enumerate(tracks["players"]):
            for track_id, track_info in player_track.items():
                if track_id in confirmed_numbers:
                    track_info["jersey_number"] = confirmed_numbers[track_id]
                elif "jersey_number" not in track_info:
                    track_info["jersey_number"] = track_id  # Fallback to track_id

    def draw_ellipse(self, frame, bbox, color, track_id=None, jersey_number=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Convert color to proper format for OpenCV (ensure it's a tuple of integers)
        if isinstance(color, (list, tuple)):
            color = tuple(int(c) for c in color)
        elif hasattr(color, "__iter__"):  # numpy array or similar
            color = tuple(int(c) for c in color)
        else:
            color = (0, 0, 255)  # Default red color if conversion fails

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        # Use jersey number if available, otherwise use track_id
        display_number = jersey_number if jersey_number is not None else track_id

        if display_number is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if display_number > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{display_number}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Convert color to proper format for OpenCV (ensure it's a tuple of integers)
        if isinstance(color, (list, tuple)):
            color = tuple(int(c) for c in color)
        elif hasattr(color, "__iter__"):  # numpy array or similar
            color = tuple(int(c) for c in color)
        else:
            color = (0, 0, 255)  # Default red color if conversion fails

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1*100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2*100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                jersey_number = player.get("jersey_number", track_id)
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id, jersey_number
                )

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
