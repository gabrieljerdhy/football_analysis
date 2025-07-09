import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

sys.path.append("../")
try:
    from src.config import DEFAULT_BALL_DETECTION_CONFIG, BallDetectionConfig
    from src.jersey_number_detector import JerseyNumberDetector
    from src.utils import get_bbox_width, get_center_of_bbox, get_foot_position
except ImportError:
    # Handle relative imports when running from different contexts
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    from config import DEFAULT_BALL_DETECTION_CONFIG, BallDetectionConfig
    from jersey_number_detector import JerseyNumberDetector
    from utils import get_bbox_width, get_center_of_bbox, get_foot_position


class Tracker:
    def __init__(
        self,
        model_path=None,
        enable_jersey_detection=True,
        ball_model_path=None,
        enable_enhanced_ball_detection=True,
        config=None,
        device=None,
    ):
        """
        Initialize the Tracker with enhanced ball detection capabilities.

        Args:
            model_path (str): Path to the main detection model (players, referees, ball)
            enable_jersey_detection (bool): Enable jersey number detection
            ball_model_path (str): Path to specialized ball detection model (optional)
            enable_enhanced_ball_detection (bool): Enable enhanced ball detection using specialized model
            config (BallDetectionConfig): Configuration object for ball detection
            device (str | torch.device, optional): Device to run models on (auto, cpu, cuda, cuda:0, etc.)
        """
        # Use configuration if provided, otherwise use defaults
        if config is None:
            config = DEFAULT_BALL_DETECTION_CONFIG

        self.config = config

        # Override config with explicit parameters if provided
        if model_path is not None:
            self.config.general_model_path = model_path
        if ball_model_path is not None:
            self.config.ball_model_path = ball_model_path
        if enable_jersey_detection is not None:
            self.config.enable_jersey_detection = enable_jersey_detection
        if enable_enhanced_ball_detection is not None:
            self.config.enable_enhanced_ball_detection = enable_enhanced_ball_detection

        # Configure device for GPU acceleration
        from src.utils import get_optimal_device

        self.device = get_optimal_device(device, verbose=False)

        # Performance optimization settings
        self.use_half_precision = self.device.type == "cuda"
        self.optimized_batch_size = 64 if self.device.type == "cuda" else 16
        self.ball_batch_size = 32 if self.device.type == "cuda" else 8
        self.imgsz = 640  # Optimized image size for inference

        # Initialize main model with device configuration
        self.model = YOLO(self.config.general_model_path)
        # Move model to specified device
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Optimize model for inference
        if self.use_half_precision and self.device.type == "cuda":
            try:
                self.model.model.half()
                print(f"✅ Half precision enabled for main model on {self.device}")
            except Exception as e:
                print(f"⚠️ Half precision failed for main model: {e}")
                self.use_half_precision = False

        self.tracker = sv.ByteTrack()

        # Enhanced ball detection configuration
        self.enable_enhanced_ball_detection = self.config.enable_enhanced_ball_detection
        self.ball_model = None
        self.ball_model_path = self.config.ball_model_path

        # Initialize specialized ball detection model
        if self.enable_enhanced_ball_detection:
            if os.path.exists(self.ball_model_path):
                try:
                    self.ball_model = YOLO(self.ball_model_path)
                    # Move ball model to specified device
                    if hasattr(self.ball_model, "to"):
                        self.ball_model.to(self.device)

                    # Optimize ball model for inference
                    if self.use_half_precision and self.device.type == "cuda":
                        try:
                            self.ball_model.model.half()
                            print(
                                f"✅ Half precision enabled for ball model on {self.device}"
                            )
                        except Exception as e:
                            print(f"⚠️ Half precision failed for ball model: {e}")

                    print(
                        f"✅ Enhanced ball detection enabled with model: {self.ball_model_path} on {self.device}"
                    )
                except Exception as e:
                    print(f"⚠️ Enhanced ball detection disabled due to error: {e}")
                    self.enable_enhanced_ball_detection = False
                    self.ball_model = None
            else:
                print(
                    f"⚠️ Enhanced ball detection disabled: {self.ball_model_path} not found"
                )
                self.enable_enhanced_ball_detection = False

        # Ball detection fusion parameters from config
        self.ball_confidence_threshold = self.config.ball_confidence_threshold
        self.ball_fusion_confidence_threshold = (
            self.config.ball_fusion_confidence_threshold
        )
        self.ball_temporal_consistency_frames = (
            self.config.ball_temporal_consistency_frames
        )
        self.ball_detection_history = (
            []
        )  # Store recent ball detections for consistency checks

        # Initialize jersey number detector
        self.enable_jersey_detection = self.config.enable_jersey_detection
        if self.enable_jersey_detection:
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

        # Warm up models for optimal performance
        self._warmup_models()

    def _warmup_models(self):
        """Warm up models with dummy data for optimal performance."""
        if self.device.type == "cuda":
            try:
                import torch

                # Create dummy frame for warmup
                dummy_frame = torch.zeros(
                    (3, self.imgsz, self.imgsz), dtype=torch.uint8
                ).numpy()
                dummy_frame = np.transpose(
                    dummy_frame, (1, 2, 0)
                )  # Convert to HWC format

                print(f"🔥 Warming up models on {self.device}...")

                # Warmup main model
                _ = self.model.predict(
                    dummy_frame, verbose=False, device=self.device, imgsz=self.imgsz
                )

                # Warmup ball model if available
                if self.ball_model is not None:
                    _ = self.ball_model.predict(
                        dummy_frame, verbose=False, device=self.device, imgsz=self.imgsz
                    )

                print("✅ Model warmup completed")
            except Exception as e:
                print(f"⚠️ Model warmup failed: {e}")

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
        """
        Enhanced ball position interpolation that considers detection confidence and source.
        """
        # Extract ball data with enhanced information
        ball_data = []
        confidences = []
        sources = []

        for x in ball_positions:
            ball_info = x.get(1, {})
            bbox = ball_info.get("bbox", [])
            confidence = ball_info.get("confidence", 0.0)
            source = ball_info.get("source", "unknown")

            ball_data.append(bbox)
            confidences.append(confidence)
            sources.append(source)

        # Create DataFrame with enhanced information
        df_ball_positions = pd.DataFrame(ball_data, columns=["x1", "y1", "x2", "y2"])
        df_confidences = pd.Series(confidences)
        df_sources = pd.Series(sources)

        # Enhanced interpolation strategy
        # 1. First, try to interpolate only high-confidence detections
        high_conf_mask = df_confidences >= self.ball_fusion_confidence_threshold
        if high_conf_mask.sum() > 2:  # Need at least 2 points for interpolation
            # Create a copy for high-confidence interpolation
            df_high_conf = df_ball_positions.copy()
            df_high_conf[~high_conf_mask] = np.nan
            df_high_conf_interpolated = df_high_conf.interpolate(
                method="linear", limit_direction="both"
            )

            # Use high-confidence interpolation where available
            for i in range(len(df_ball_positions)):
                if (
                    pd.isna(df_ball_positions.iloc[i]).any()
                    and not pd.isna(df_high_conf_interpolated.iloc[i]).any()
                ):
                    df_ball_positions.iloc[i] = df_high_conf_interpolated.iloc[i]
                    df_confidences.iloc[i] = 0.4  # Mark as interpolated
                    df_sources.iloc[i] = "interpolated_high_conf"

        # 2. Standard interpolation for remaining gaps
        df_ball_positions = df_ball_positions.interpolate(method="linear")
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()

        # Mark remaining interpolated values
        for i in range(len(df_confidences)):
            if (
                df_confidences.iloc[i] == 0.0
                and not pd.isna(df_ball_positions.iloc[i]).any()
            ):
                df_confidences.iloc[i] = (
                    0.2  # Low confidence for standard interpolation
                )
                df_sources.iloc[i] = "interpolated_standard"

        # Reconstruct ball positions with enhanced information
        enhanced_ball_positions = []
        for i, (bbox_data, conf, source) in enumerate(
            zip(df_ball_positions.to_numpy(), df_confidences, df_sources)
        ):
            if not pd.isna(bbox_data).any():
                ball_track = {
                    1: {
                        "bbox": bbox_data.tolist(),
                        "confidence": conf,
                        "source": source,
                    }
                }

                # Calculate position
                bbox = ball_track[1]["bbox"]
                position = get_center_of_bbox(bbox)
                ball_track[1]["position"] = position

                enhanced_ball_positions.append(ball_track)
            else:
                enhanced_ball_positions.append({})

        return enhanced_ball_positions

    def detect_ball_enhanced(self, frames, frame_indices=None):
        """
        Enhanced ball detection using specialized ball model with fusion logic.

        Args:
            frames: List of video frames
            frame_indices: Optional list of frame indices for tracking history

        Returns:
            List of enhanced ball detections per frame
        """
        if not self.enable_enhanced_ball_detection or self.ball_model is None:
            return None

        if len(frames) == 0:
            return []

        batch_size = min(self.ball_batch_size, len(frames))
        ball_detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            # Use higher confidence for specialized ball model with optimized parameters
            detections_batch = self.ball_model.predict(
                batch_frames,
                conf=self.ball_confidence_threshold,
                device=self.device,
                imgsz=self.imgsz,
                half=self.use_half_precision,
                verbose=False,
            )
            ball_detections += detections_batch

        return ball_detections

    def fuse_ball_detections(self, general_detection, ball_detection, frame_num):
        """
        Fuse ball detections from general and specialized models.

        Args:
            general_detection: Detection from general model
            ball_detection: Detection from specialized ball model
            frame_num: Current frame number

        Returns:
            Best ball detection with confidence score
        """
        best_ball_bbox = None
        best_confidence = 0.0
        detection_source = "none"

        # Extract ball detections from general model
        general_ball_detections = []
        if general_detection.boxes is not None:
            boxes = general_detection.boxes.xyxy.cpu().numpy()
            confidences = general_detection.boxes.conf.cpu().numpy()
            class_ids = general_detection.boxes.cls.cpu().numpy()

            cls_names = general_detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            if "ball" in cls_names_inv:
                ball_class_id = cls_names_inv["ball"]
                for i, (box, conf, class_id) in enumerate(
                    zip(boxes, confidences, class_ids)
                ):
                    if (
                        class_id == ball_class_id
                        and conf >= self.ball_confidence_threshold
                    ):
                        general_ball_detections.append(
                            {
                                "bbox": box.tolist(),
                                "confidence": conf,
                                "source": "general",
                            }
                        )

        # Extract ball detections from specialized model
        specialized_ball_detections = []
        if ball_detection is not None and ball_detection.boxes is not None:
            boxes = ball_detection.boxes.xyxy.cpu().numpy()
            confidences = ball_detection.boxes.conf.cpu().numpy()

            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                if conf >= self.ball_confidence_threshold:
                    specialized_ball_detections.append(
                        {
                            "bbox": box.tolist(),
                            "confidence": conf,
                            "source": "specialized",
                        }
                    )

        # Select best detection based on confidence and temporal consistency
        all_detections = general_ball_detections + specialized_ball_detections

        if all_detections:
            # Prefer specialized model if confidence is high enough
            specialized_detections = [
                d for d in all_detections if d["source"] == "specialized"
            ]
            if specialized_detections:
                # Use highest confidence specialized detection
                best_detection = max(
                    specialized_detections, key=lambda x: x["confidence"]
                )
                if (
                    best_detection["confidence"]
                    >= self.ball_fusion_confidence_threshold
                ):
                    best_ball_bbox = best_detection["bbox"]
                    best_confidence = best_detection["confidence"]
                    detection_source = "specialized"

            # Fall back to general model if specialized model confidence is low
            if best_ball_bbox is None and general_ball_detections:
                best_detection = max(
                    general_ball_detections, key=lambda x: x["confidence"]
                )
                best_ball_bbox = best_detection["bbox"]
                best_confidence = best_detection["confidence"]
                detection_source = "general"

            # If still no detection from general, use best specialized
            if best_ball_bbox is None and specialized_detections:
                best_detection = max(
                    specialized_detections, key=lambda x: x["confidence"]
                )
                best_ball_bbox = best_detection["bbox"]
                best_confidence = best_detection["confidence"]
                detection_source = "specialized_fallback"

        # Update detection history for temporal consistency
        if best_ball_bbox is not None:
            detection_info = {
                "frame_num": frame_num,
                "bbox": best_ball_bbox,
                "confidence": best_confidence,
                "source": detection_source,
            }
            self.ball_detection_history.append(detection_info)

            # Keep only recent history
            if (
                len(self.ball_detection_history)
                > self.ball_temporal_consistency_frames * 2
            ):
                self.ball_detection_history = self.ball_detection_history[
                    -self.ball_temporal_consistency_frames * 2 :
                ]

        return (
            {
                "bbox": best_ball_bbox,
                "confidence": best_confidence,
                "source": detection_source,
            }
            if best_ball_bbox is not None
            else None
        )

    def detect_frames(self, frames):
        batch_size = min(self.optimized_batch_size, len(frames))
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i : i + batch_size],
                conf=0.1,
                device=self.device,
                imgsz=self.imgsz,
                half=self.use_half_precision,
                verbose=False,
            )
            detections += detections_batch
        return detections

    def detect_frames_memory_efficient(self, frames, progress_callback=None):
        """Memory-efficient frame detection with optimized batches and cleanup."""
        if len(frames) == 0:
            return []

        # Use optimized batch sizes for better GPU utilization
        batch_size = min(self.optimized_batch_size, len(frames))
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            detections_batch = self.model.predict(
                batch_frames,
                conf=0.1,
                device=self.device,
                imgsz=self.imgsz,
                half=self.use_half_precision,
                verbose=False,
            )
            detections += detections_batch

            # Progress callback
            if progress_callback:
                progress_callback(i + len(batch_frames), len(frames))

        # Single cleanup at the end instead of after each batch for better performance
        import gc

        gc.collect()
        if self.device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

        return detections

    def get_object_tracks_memory_efficient(
        self, frames, read_from_stub=False, stub_path=None, progress_callback=None
    ):
        """Memory-efficient version of get_object_tracks with enhanced ball detection."""

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        # Use memory-efficient detection
        detections = self.detect_frames_memory_efficient(frames, progress_callback)

        # Get enhanced ball detections if enabled
        ball_detections = None
        if self.enable_enhanced_ball_detection:
            ball_detections = self.detect_ball_enhanced(frames)

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

            # Enhanced ball detection with fusion logic
            ball_detection_result = None
            if ball_detections and frame_num < len(ball_detections):
                ball_detection_result = self.fuse_ball_detections(
                    detection, ball_detections[frame_num], frame_num
                )

            # Use enhanced ball detection if available, otherwise fall back to original
            if ball_detection_result and ball_detection_result["bbox"] is not None:
                tracks["ball"][frame_num][1] = {
                    "bbox": ball_detection_result["bbox"],
                    "confidence": ball_detection_result["confidence"],
                    "source": ball_detection_result["source"],
                }
            else:
                # Fall back to original ball detection from general model
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
        """Enhanced object tracking with improved ball detection."""

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # Get enhanced ball detections if enabled
        ball_detections = None
        if self.enable_enhanced_ball_detection:
            ball_detections = self.detect_ball_enhanced(frames)

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

            # Enhanced ball detection with fusion logic
            ball_detection_result = None
            if ball_detections and frame_num < len(ball_detections):
                ball_detection_result = self.fuse_ball_detections(
                    detection, ball_detections[frame_num], frame_num
                )

            # Use enhanced ball detection if available, otherwise fall back to original
            if ball_detection_result and ball_detection_result["bbox"] is not None:
                tracks["ball"][frame_num][1] = {
                    "bbox": ball_detection_result["bbox"],
                    "confidence": ball_detection_result["confidence"],
                    "source": ball_detection_result["source"],
                }
            else:
                # Fall back to original ball detection from general model
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

    def _propagate_jersey_numbers(self, tracks):
        """
        Propagate detected jersey numbers to all frames for each player.

        Args:
            tracks: Player tracking data to update with jersey numbers
        """
        if not self.enable_jersey_detection or self.jersey_detector is None:
            return

        # Get all confirmed jersey numbers from cache
        confirmed_jerseys = self.jersey_detector.player_jersey_cache

        print(
            f"🔄 Propagating {len(confirmed_jerseys)} confirmed jersey numbers to all frames..."
        )

        # Apply confirmed jersey numbers to all frames
        for frame_num, player_track in enumerate(tracks["players"]):
            for track_id, track_info in player_track.items():
                if track_id in confirmed_jerseys:
                    track_info["jersey_number"] = confirmed_jerseys[track_id]
                else:
                    # Use track_id as fallback for unconfirmed players
                    track_info["jersey_number"] = track_id

        print(f"✅ Jersey numbers propagated to {len(tracks['players']):,} frames")
