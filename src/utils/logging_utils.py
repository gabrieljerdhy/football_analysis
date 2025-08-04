"""
Comprehensive logging system for football analysis application.

This module provides detailed logging capabilities for tracking analysis runs,
performance metrics, and debugging information.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AnalysisLogger:
    """
    Comprehensive logger for football analysis pipeline.

    Features:
    - Unique log files per analysis run
    - Structured logging with timestamps
    - Performance metrics tracking
    - Error and warning capture
    - Human-readable format
    """

    def __init__(
        self,
        log_file_path: str,
        match_name: Optional[str] = None,
        log_level: int = logging.INFO,
        enable_console: bool = True,
        enable_performance_tracking: bool = True,
    ):
        """
        Initialize the analysis logger.

        Args:
            log_file_path: Path to the log file
            match_name: Optional match name for context
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_console: Whether to also log to console
            enable_performance_tracking: Whether to track performance metrics
        """
        self.log_file_path = log_file_path
        self.match_name = match_name or "Unknown Match"
        self.enable_performance_tracking = enable_performance_tracking

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(f"football_analysis_{id(self)}")
        self.logger.setLevel(log_level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler (optional)
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Performance tracking
        self.start_time = time.time()
        self.stage_times = {}
        self.current_stage = None
        self.performance_metrics = {
            "total_frames_processed": 0,
            "processing_speed_fps": 0.0,
            "memory_usage_peak_gb": 0.0,
            "stages_completed": [],
            "errors_encountered": 0,
            "warnings_encountered": 0,
        }

        # Analysis context
        self.analysis_context = {
            "input_video_path": None,
            "output_video_path": None,
            "video_metadata": {},
            "configuration": {},
            "team_statistics": {},
            "final_results": {},
        }

        # Log analysis start
        self._log_analysis_start()

    def _log_analysis_start(self):
        """Log the start of analysis with header information."""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 FOOTBALL ANALYSIS STARTED")
        self.logger.info(
            f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(f"🏟️  Match: {self.match_name}")
        self.logger.info(f"📝 Log File: {self.log_file_path}")
        self.logger.info("=" * 80)

    def set_input_video(self, video_path: str, metadata: Optional[Dict] = None):
        """Set input video information."""
        self.analysis_context["input_video_path"] = video_path
        if metadata:
            self.analysis_context["video_metadata"] = metadata

        self.logger.info(f"📹 Input Video: {video_path}")
        if metadata:
            self.logger.info(f"📊 Video Metadata:")
            for key, value in metadata.items():
                self.logger.info(f"   • {key}: {value}")

    def set_configuration(self, config: Dict[str, Any]):
        """Set analysis configuration."""
        self.analysis_context["configuration"] = config
        self.logger.info(f"⚙️  Analysis Configuration:")
        for key, value in config.items():
            self.logger.info(f"   • {key}: {value}")

    def start_stage(self, stage_name: str, description: Optional[str] = None):
        """Start a new processing stage."""
        if self.current_stage and self.enable_performance_tracking:
            self._end_current_stage()

        self.current_stage = stage_name
        if self.enable_performance_tracking:
            self.stage_times[stage_name] = {"start": time.time()}

        log_msg = f"🔄 Starting Stage: {stage_name}"
        if description:
            log_msg += f" - {description}"
        self.logger.info(log_msg)

    def end_stage(
        self, stage_name: Optional[str] = None, results: Optional[Dict] = None
    ):
        """End the current processing stage."""
        if stage_name and stage_name != self.current_stage:
            self.logger.warning(
                f"⚠️  Stage mismatch: expected {self.current_stage}, got {stage_name}"
            )

        if self.current_stage and self.enable_performance_tracking:
            self._end_current_stage()

        if results:
            self.logger.info(f"📊 Stage Results:")
            for key, value in results.items():
                self.logger.info(f"   • {key}: {value}")

        self.logger.info(f"✅ Completed Stage: {self.current_stage or stage_name}")
        self.current_stage = None

    def _end_current_stage(self):
        """Internal method to end current stage timing."""
        if self.current_stage in self.stage_times:
            self.stage_times[self.current_stage]["end"] = time.time()
            duration = (
                self.stage_times[self.current_stage]["end"]
                - self.stage_times[self.current_stage]["start"]
            )
            self.stage_times[self.current_stage]["duration"] = duration
            self.performance_metrics["stages_completed"].append(
                {"name": self.current_stage, "duration_seconds": duration}
            )
            self.logger.info(f"⏱️  Stage Duration: {duration:.2f} seconds")

    def log_detection_result(
        self,
        detection_type: str,
        frame_num: int,
        confidence: float,
        details: Optional[Dict] = None,
    ):
        """Log detection results (goals, passes, tackles, etc.)."""
        log_msg = f"🎯 {detection_type} detected at frame {frame_num} (confidence: {confidence:.3f})"
        if details:
            log_msg += f" - {details}"
        self.logger.info(log_msg)

    def log_scoreboard_detection(
        self, frame_num: int, score: Optional[str], confidence: float, success: bool
    ):
        """Log scoreboard detection results."""
        status = "✅" if success else "❌"
        score_text = score if score else "N/A"
        self.logger.info(
            f"{status} Scoreboard at frame {frame_num}: {score_text} (confidence: {confidence:.3f})"
        )

    def log_team_statistics(self, team_stats: Dict[str, Any]):
        """Log team statistics."""
        self.analysis_context["team_statistics"] = team_stats
        self.logger.info(f"📈 Team Statistics:")
        for team, stats in team_stats.items():
            self.logger.info(f"   Team {team}:")
            for stat_name, value in stats.items():
                self.logger.info(f"     • {stat_name}: {value}")

    def log_performance_metric(self, metric_name: str, value: Union[int, float, str]):
        """Log a performance metric."""
        self.performance_metrics[metric_name] = value
        self.logger.info(f"📊 Performance: {metric_name} = {value}")

    def log_frame_processing(
        self,
        frame_num: int,
        total_frames: int,
        processing_speed: Optional[float] = None,
    ):
        """Log frame processing progress."""
        progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0

        if frame_num % 1000 == 0 or frame_num == total_frames:  # Log every 1000 frames
            log_msg = (
                f"🎬 Processing frame {frame_num}/{total_frames} ({progress:.1f}%)"
            )
            if processing_speed:
                log_msg += f" - {processing_speed:.1f} FPS"
            self.logger.info(log_msg)

        self.performance_metrics["total_frames_processed"] = frame_num
        if processing_speed:
            self.performance_metrics["processing_speed_fps"] = processing_speed

    def log_memory_usage(self, memory_gb: float, stage: Optional[str] = None):
        """Log memory usage."""
        if memory_gb > self.performance_metrics["memory_usage_peak_gb"]:
            self.performance_metrics["memory_usage_peak_gb"] = memory_gb

        stage_text = f" ({stage})" if stage else ""
        self.logger.info(f"💾 Memory Usage: {memory_gb:.2f} GB{stage_text}")

    def log_output_files(self, output_files: Dict[str, str]):
        """Log generated output files."""
        self.logger.info(f"📁 Output Files Generated:")
        for file_type, file_path in output_files.items():
            self.logger.info(f"   • {file_type}: {file_path}")

        # Store in context
        self.analysis_context["output_files"] = output_files

    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log an error."""
        self.performance_metrics["errors_encountered"] += 1
        if exception:
            self.logger.error(f"❌ ERROR: {error_msg} - {str(exception)}")
        else:
            self.logger.error(f"❌ ERROR: {error_msg}")

    def log_warning(self, warning_msg: str):
        """Log a warning."""
        self.performance_metrics["warnings_encountered"] += 1
        self.logger.warning(f"⚠️  WARNING: {warning_msg}")

    def finalize_analysis(self, final_results: Optional[Dict] = None):
        """Finalize the analysis and log summary."""
        if self.current_stage and self.enable_performance_tracking:
            self._end_current_stage()

        total_duration = time.time() - self.start_time

        if final_results:
            self.analysis_context["final_results"] = final_results

        # Log final summary
        self.logger.info("=" * 80)
        self.logger.info(f"🏁 FOOTBALL ANALYSIS COMPLETED")
        self.logger.info(
            f"⏱️  Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)"
        )

        # Performance summary
        if self.enable_performance_tracking:
            self.logger.info(f"📊 Performance Summary:")
            self.logger.info(
                f"   • Frames Processed: {self.performance_metrics['total_frames_processed']:,}"
            )
            self.logger.info(
                f"   • Processing Speed: {self.performance_metrics['processing_speed_fps']:.1f} FPS"
            )
            self.logger.info(
                f"   • Peak Memory Usage: {self.performance_metrics['memory_usage_peak_gb']:.2f} GB"
            )
            self.logger.info(
                f"   • Stages Completed: {len(self.performance_metrics['stages_completed'])}"
            )
            self.logger.info(
                f"   • Errors: {self.performance_metrics['errors_encountered']}"
            )
            self.logger.info(
                f"   • Warnings: {self.performance_metrics['warnings_encountered']}"
            )

        # Final results
        if final_results:
            self.logger.info(f"🏆 Final Results:")
            for key, value in final_results.items():
                self.logger.info(f"   • {key}: {value}")

        self.logger.info("=" * 80)

        # Save analysis summary as JSON
        self._save_analysis_summary(total_duration)

    def _save_analysis_summary(self, total_duration: float):
        """Save analysis summary as JSON file."""
        summary_path = self.log_file_path.replace(".log", "_summary.json")

        summary = {
            "analysis_info": {
                "match_name": self.match_name,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": total_duration,
                "log_file": self.log_file_path,
            },
            "performance_metrics": self.performance_metrics,
            "analysis_context": self.analysis_context,
            "stage_timings": self.stage_times,
        }

        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"📋 Analysis summary saved to: {summary_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis summary: {e}")


def create_analysis_logger(
    input_video_path: str,
    output_dir: str = "data/output",
    match_name: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_console: bool = True,
) -> AnalysisLogger:
    """
    Create an analysis logger with automatic file naming.

    Args:
        input_video_path: Path to input video file
        output_dir: Directory to store log files
        match_name: Optional match name (extracted from video if not provided)
        log_level: Logging level
        enable_console: Whether to also log to console

    Returns:
        Configured AnalysisLogger instance
    """
    # Extract match name from video path if not provided
    if not match_name:
        match_name = Path(input_video_path).stem

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create log file path
    log_filename = f"{match_name}_analysis_log_{timestamp}.log"
    log_file_path = os.path.join(output_dir, log_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    return AnalysisLogger(
        log_file_path=log_file_path,
        match_name=match_name,
        log_level=log_level,
        enable_console=enable_console,
    )


# Global logger instance for easy access
_global_logger: Optional[AnalysisLogger] = None


def get_analysis_logger() -> Optional[AnalysisLogger]:
    """Get the global analysis logger instance."""
    return _global_logger


def set_global_logger(logger: AnalysisLogger):
    """Set the global analysis logger instance."""
    global _global_logger
    _global_logger = logger


def log_detection_event(
    detection_type: str,
    frame_num: int,
    confidence: float,
    details: Optional[Dict] = None,
):
    """
    Convenience function to log detection events using the global logger.

    Args:
        detection_type: Type of detection (goal, pass, tackle, etc.)
        frame_num: Frame number where detection occurred
        confidence: Detection confidence score
        details: Optional additional details
    """
    logger = get_analysis_logger()
    if logger:
        logger.log_detection_result(detection_type, frame_num, confidence, details)


def log_scoreboard_event(
    frame_num: int, score: Optional[str], confidence: float, success: bool
):
    """
    Convenience function to log scoreboard detection events using the global logger.

    Args:
        frame_num: Frame number where detection occurred
        score: Detected score string (if any)
        confidence: Detection confidence score
        success: Whether detection was successful
    """
    logger = get_analysis_logger()
    if logger:
        logger.log_scoreboard_detection(frame_num, score, confidence, success)


def log_performance_metric(metric_name: str, value: Union[int, float, str]):
    """
    Convenience function to log performance metrics using the global logger.

    Args:
        metric_name: Name of the metric
        value: Metric value
    """
    logger = get_analysis_logger()
    if logger:
        logger.log_performance_metric(metric_name, value)


def log_memory_usage(memory_gb: float, stage: Optional[str] = None):
    """
    Convenience function to log memory usage using the global logger.

    Args:
        memory_gb: Memory usage in GB
        stage: Optional stage name
    """
    logger = get_analysis_logger()
    if logger:
        logger.log_memory_usage(memory_gb, stage)
