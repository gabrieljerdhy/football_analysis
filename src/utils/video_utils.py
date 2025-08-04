import gc
import os
from typing import Generator, List, Optional, Tuple

import cv2
import psutil

from .multi_storage_utils import MultiStorageHandler, is_object_storage_uri
from .s3_video_utils import S3VideoHandler, is_s3_uri


class VideoFrameIterator:
    """Memory-efficient video frame iterator for processing large videos."""

    def __init__(
        self,
        video_path: str,
        batch_size: int = 50,
        s3_handler: Optional[S3VideoHandler] = None,
        storage_handler: Optional[MultiStorageHandler] = None,
        **storage_auth_kwargs,
    ):
        self.original_video_path = video_path
        self.batch_size = batch_size
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.width = 0
        self.height = 0
        self.s3_handler = s3_handler  # Keep for backward compatibility
        self.storage_handler = storage_handler
        self.storage_auth_kwargs = storage_auth_kwargs
        self.local_video_path = None  # Will store local path for remote videos

    def __enter__(self):
        # Handle object storage URIs by downloading to local path first
        if is_object_storage_uri(self.original_video_path):
            print(f"🌐 Detected object storage URI: {self.original_video_path}")

            # Use multi-storage handler if available, otherwise fall back to S3 handler
            if self.storage_handler is not None:
                self.local_video_path = self.storage_handler.download_video(
                    self.original_video_path
                )
            elif is_s3_uri(self.original_video_path):
                # Backward compatibility with S3 handler
                if self.s3_handler is None:
                    # Extract S3-specific kwargs for backward compatibility
                    s3_kwargs = {
                        k: v
                        for k, v in self.storage_auth_kwargs.items()
                        if k.startswith("aws_") or k in ["region_name", "profile_name"]
                    }
                    self.s3_handler = S3VideoHandler(**s3_kwargs)

                self.local_video_path = self.s3_handler.get_local_path(
                    self.original_video_path
                )
            else:
                # Create multi-storage handler for non-S3 providers
                self.storage_handler = MultiStorageHandler(**self.storage_auth_kwargs)
                self.local_video_path = self.storage_handler.download_video(
                    self.original_video_path
                )

            if self.local_video_path is None:
                raise ValueError(
                    f"Could not download video: {self.original_video_path}"
                )

            video_path = self.local_video_path
        else:
            video_path = self.original_video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(
            f"📹 Video Info: {self.total_frames} frames, {self.fps:.1f} FPS, {self.width}x{self.height}"
        )
        duration_minutes = (self.total_frames / self.fps) / 60
        print(f"⏱️  Duration: {duration_minutes:.1f} minutes")

        if is_object_storage_uri(self.original_video_path):
            print(f"🌐 Object Storage Source: {self.original_video_path}")
            print(f"📁 Local Cache: {self.local_video_path}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self) -> List[cv2.Mat]:
        """Get next batch of frames."""
        if self.current_frame >= self.total_frames:
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            ret, frame = self.cap.read()
            if not ret:
                break
            batch.append(frame)
            self.current_frame += 1

        if not batch:
            raise StopIteration

        return batch

    def get_progress(self) -> float:
        """Get processing progress as percentage."""
        return (
            (self.current_frame / self.total_frames) * 100
            if self.total_frames > 0
            else 0
        )


def read_video(
    video_path,
    s3_handler: Optional[S3VideoHandler] = None,
    storage_handler: Optional[MultiStorageHandler] = None,
    **storage_auth_kwargs,
):
    """Legacy function - loads all frames into memory. Use VideoFrameIterator for large videos."""
    print(
        "⚠️  WARNING: Loading all frames into memory. For large videos, use process_video_efficiently() instead."
    )

    # Handle object storage URIs
    if is_object_storage_uri(video_path):
        print(f"🌐 Detected object storage URI: {video_path}")

        # Use multi-storage handler if available, otherwise fall back to S3 handler
        if storage_handler is not None:
            local_path = storage_handler.download_video(video_path)
        elif is_s3_uri(video_path) and s3_handler is not None:
            # Backward compatibility with S3 handler
            local_path = s3_handler.get_local_path(video_path)
        else:
            # Create appropriate handler
            if is_s3_uri(video_path):
                s3_kwargs = {
                    k: v
                    for k, v in storage_auth_kwargs.items()
                    if k.startswith("aws_") or k in ["region_name", "profile_name"]
                }
                s3_handler = S3VideoHandler(**s3_kwargs)
                local_path = s3_handler.get_local_path(video_path)
            else:
                storage_handler = MultiStorageHandler(**storage_auth_kwargs)
                local_path = storage_handler.download_video(video_path)

        if local_path is None:
            raise ValueError(f"Could not download video: {video_path}")

        video_path = local_path
        print(f"📁 Using local cache: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_info(
    video_path: str,
    s3_handler: Optional[S3VideoHandler] = None,
    storage_handler: Optional[MultiStorageHandler] = None,
    **storage_auth_kwargs,
) -> dict:
    """Get video information without loading frames."""

    # Handle object storage URIs
    if is_object_storage_uri(video_path):
        print(f"🌐 Getting info for object storage video: {video_path}")

        # Use multi-storage handler if available, otherwise fall back to S3 handler
        if storage_handler is not None:
            local_path = storage_handler.download_video(video_path)
        elif is_s3_uri(video_path) and s3_handler is not None:
            # Backward compatibility with S3 handler
            local_path = s3_handler.get_local_path(video_path)
        else:
            # Create appropriate handler
            if is_s3_uri(video_path):
                s3_kwargs = {
                    k: v
                    for k, v in storage_auth_kwargs.items()
                    if k.startswith("aws_") or k in ["region_name", "profile_name"]
                }
                s3_handler = S3VideoHandler(**s3_kwargs)
                local_path = s3_handler.get_local_path(video_path)
            else:
                storage_handler = MultiStorageHandler(**storage_auth_kwargs)
                local_path = storage_handler.download_video(video_path)

        if local_path is None:
            raise ValueError(f"Could not download video: {video_path}")

        video_path = local_path
        print(f"📁 Using local cache: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": 0,
    }

    if info["fps"] > 0:
        info["duration_seconds"] = info["total_frames"] / info["fps"]

    cap.release()
    return info


def monitor_memory_usage():
    """Monitor current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024**3)
    return memory_gb


def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()


def save_video_streaming(
    frame_iterator,
    output_video_path: str,
    fps: float = 24.0,
    frame_processor=None,
    total_frames: int = None,
):
    """
    Save video using streaming approach to avoid memory issues.

    Args:
        frame_iterator: Iterator that yields frames
        output_video_path: Path to save the output video
        fps: Frames per second for output video
        frame_processor: Optional function to process each frame
        total_frames: Total number of frames for progress tracking
    """
    print(f"💾 Saving video to: {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    frames_written = 0

    try:
        for batch_frames in frame_iterator:
            for frame in batch_frames:
                # Process frame if processor is provided
                if frame_processor:
                    frame = frame_processor(frame)

                # Initialize video writer with first frame dimensions
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(
                        output_video_path, fourcc, fps, (width, height)
                    )
                    print(f"📝 Initialized video writer: {width}x{height} at {fps} FPS")

                out.write(frame)
                frames_written += 1

                # Progress update
                if frames_written % 100 == 0:
                    if total_frames:
                        progress = (frames_written / total_frames) * 100
                        print(
                            f"💾 Saved {frames_written}/{total_frames} frames ({progress:.1f}%)"
                        )
                    else:
                        print(f"💾 Saved {frames_written} frames")

            # Clean up memory after each batch
            cleanup_memory()

    finally:
        if out:
            out.release()
            print(f"✅ Video saved successfully: {frames_written} frames written")


def save_video(ouput_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
