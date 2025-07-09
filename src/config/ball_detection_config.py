"""
Enhanced Ball Detection Configuration

This module provides configuration options for the enhanced ball detection system
that uses both general and specialized ball detection models.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BallDetectionConfig:
    """Configuration for enhanced ball detection system."""

    # Model paths
    general_model_path: str = "data/models/best_player_detect.pt"
    ball_model_path: str = "data/models/best_ball_latest.pt"

    # Enhanced ball detection settings
    enable_enhanced_ball_detection: bool = True
    enable_jersey_detection: bool = True

    # Confidence thresholds
    ball_confidence_threshold: float = 0.3
    ball_fusion_confidence_threshold: float = 0.5

    # Temporal consistency settings
    ball_temporal_consistency_frames: int = 3

    # Interpolation settings
    enable_enhanced_interpolation: bool = True
    high_confidence_interpolation_threshold: float = 0.5

    # Performance settings
    ball_detection_batch_size: int = 10

    # YOLO optimization settings
    use_half_precision: bool = True  # Enable FP16 for GPU inference
    optimized_batch_size: int = 64  # Larger batch size for main model
    ball_batch_size: int = 32  # Batch size for ball detection model
    imgsz: int = 640  # Optimized image size for inference
    enable_model_warmup: bool = True  # Warm up models on initialization

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate_config()

    def validate_config(self):
        """Validate the configuration settings."""
        # Check if model files exist
        if not os.path.exists(self.general_model_path):
            raise FileNotFoundError(
                f"General model not found: {self.general_model_path}"
            )

        if self.enable_enhanced_ball_detection and not os.path.exists(
            self.ball_model_path
        ):
            print(
                f"⚠️ Ball model not found: {self.ball_model_path}. Enhanced ball detection will be disabled."
            )
            self.enable_enhanced_ball_detection = False

        # Validate thresholds
        if not 0.0 <= self.ball_confidence_threshold <= 1.0:
            raise ValueError("ball_confidence_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.ball_fusion_confidence_threshold <= 1.0:
            raise ValueError(
                "ball_fusion_confidence_threshold must be between 0.0 and 1.0"
            )

        if self.ball_temporal_consistency_frames < 1:
            raise ValueError("ball_temporal_consistency_frames must be at least 1")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BallDetectionConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "general_model_path": self.general_model_path,
            "ball_model_path": self.ball_model_path,
            "enable_enhanced_ball_detection": self.enable_enhanced_ball_detection,
            "enable_jersey_detection": self.enable_jersey_detection,
            "ball_confidence_threshold": self.ball_confidence_threshold,
            "ball_fusion_confidence_threshold": self.ball_fusion_confidence_threshold,
            "ball_temporal_consistency_frames": self.ball_temporal_consistency_frames,
            "enable_enhanced_interpolation": self.enable_enhanced_interpolation,
            "high_confidence_interpolation_threshold": self.high_confidence_interpolation_threshold,
            "ball_detection_batch_size": self.ball_detection_batch_size,
        }


# Default configuration
DEFAULT_BALL_DETECTION_CONFIG = BallDetectionConfig()


def get_ball_detection_config(config_path: Optional[str] = None) -> BallDetectionConfig:
    """
    Get ball detection configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        BallDetectionConfig instance
    """
    if config_path and os.path.exists(config_path):
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return BallDetectionConfig.from_dict(config_dict)

    return DEFAULT_BALL_DETECTION_CONFIG


def save_ball_detection_config(config: BallDetectionConfig, config_path: str):
    """
    Save ball detection configuration to file.

    Args:
        config: BallDetectionConfig instance
        config_path: Path to save configuration file
    """
    import json

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


# Configuration presets for different scenarios
PERFORMANCE_OPTIMIZED_CONFIG = BallDetectionConfig(
    ball_confidence_threshold=0.4,
    ball_fusion_confidence_threshold=0.6,
    ball_temporal_consistency_frames=2,
    ball_detection_batch_size=15,
)

ACCURACY_OPTIMIZED_CONFIG = BallDetectionConfig(
    ball_confidence_threshold=0.2,
    ball_fusion_confidence_threshold=0.4,
    ball_temporal_consistency_frames=5,
    ball_detection_batch_size=8,
)

BALANCED_CONFIG = BallDetectionConfig(
    ball_confidence_threshold=0.3,
    ball_fusion_confidence_threshold=0.5,
    ball_temporal_consistency_frames=3,
    ball_detection_batch_size=10,
)
