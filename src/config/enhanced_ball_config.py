"""
Enhanced Ball Detection Configuration for Improved Goal Detection

This module provides optimized configuration for ball detection to improve
goal detection accuracy while maintaining performance.
"""

from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class EnhancedBallDetectionConfig:
    """Enhanced configuration for ball detection optimized for goal detection."""
    
    # Model paths - using the best available models
    general_model_path: str = "data/models/best_player_detect.pt"
    ball_model_path: str = "data/models/best_ball_latest.pt"
    field_keypoint_model_path: str = "data/models/best_field_keypoint.pt"
    
    # Enhanced ball detection settings
    enable_enhanced_ball_detection: bool = True
    enable_multi_model_fusion: bool = True
    enable_temporal_consistency: bool = True
    
    # Optimized confidence thresholds for goal detection
    ball_confidence_threshold: float = 0.25  # Lowered for better sensitivity
    ball_fusion_confidence_threshold: float = 0.4  # Lowered for better detection
    high_confidence_threshold: float = 0.7  # For high-quality detections
    
    # Temporal consistency settings
    ball_temporal_consistency_frames: int = 5  # Increased for better tracking
    temporal_smoothing_window: int = 3
    
    # Enhanced interpolation settings
    enable_enhanced_interpolation: bool = True
    interpolation_max_gap: int = 8  # Maximum frames to interpolate
    interpolation_confidence_threshold: float = 0.4
    
    # Goal detection specific settings
    goal_area_ball_confidence_boost: float = 0.2  # Boost confidence in goal areas
    near_goal_detection_sensitivity: float = 1.5  # Increase sensitivity near goals
    
    # Performance optimization
    ball_detection_batch_size: int = 12
    use_half_precision: bool = True
    optimized_batch_size: int = 64
    ball_batch_size: int = 32
    imgsz: int = 640
    enable_model_warmup: bool = True
    
    # Advanced detection settings
    enable_trajectory_prediction: bool = True
    trajectory_prediction_frames: int = 5
    enable_occlusion_handling: bool = True
    occlusion_recovery_frames: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate_config()
        
    def validate_config(self):
        """Validate configuration parameters."""
        # Validate model paths
        required_models = [
            self.general_model_path,
            self.ball_model_path,
            self.field_keypoint_model_path
        ]
        
        for model_path in required_models:
            if not os.path.exists(model_path):
                print(f"⚠️  Warning: Model file not found: {model_path}")
                
        # Validate thresholds
        if not 0.0 <= self.ball_confidence_threshold <= 1.0:
            raise ValueError("ball_confidence_threshold must be between 0.0 and 1.0")
            
        if not 0.0 <= self.ball_fusion_confidence_threshold <= 1.0:
            raise ValueError("ball_fusion_confidence_threshold must be between 0.0 and 1.0")
            
        # Validate temporal settings
        if self.ball_temporal_consistency_frames < 1:
            raise ValueError("ball_temporal_consistency_frames must be >= 1")
            
        if self.temporal_smoothing_window < 1:
            raise ValueError("temporal_smoothing_window must be >= 1")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "general_model_path": self.general_model_path,
            "ball_model_path": self.ball_model_path,
            "field_keypoint_model_path": self.field_keypoint_model_path,
            "enable_enhanced_ball_detection": self.enable_enhanced_ball_detection,
            "enable_multi_model_fusion": self.enable_multi_model_fusion,
            "enable_temporal_consistency": self.enable_temporal_consistency,
            "ball_confidence_threshold": self.ball_confidence_threshold,
            "ball_fusion_confidence_threshold": self.ball_fusion_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "ball_temporal_consistency_frames": self.ball_temporal_consistency_frames,
            "temporal_smoothing_window": self.temporal_smoothing_window,
            "enable_enhanced_interpolation": self.enable_enhanced_interpolation,
            "interpolation_max_gap": self.interpolation_max_gap,
            "interpolation_confidence_threshold": self.interpolation_confidence_threshold,
            "goal_area_ball_confidence_boost": self.goal_area_ball_confidence_boost,
            "near_goal_detection_sensitivity": self.near_goal_detection_sensitivity,
            "ball_detection_batch_size": self.ball_detection_batch_size,
            "use_half_precision": self.use_half_precision,
            "optimized_batch_size": self.optimized_batch_size,
            "ball_batch_size": self.ball_batch_size,
            "imgsz": self.imgsz,
            "enable_model_warmup": self.enable_model_warmup,
            "enable_trajectory_prediction": self.enable_trajectory_prediction,
            "trajectory_prediction_frames": self.trajectory_prediction_frames,
            "enable_occlusion_handling": self.enable_occlusion_handling,
            "occlusion_recovery_frames": self.occlusion_recovery_frames
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedBallDetectionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    def get_goal_optimized_config(self) -> 'EnhancedBallDetectionConfig':
        """Get configuration optimized specifically for goal detection."""
        config = EnhancedBallDetectionConfig(
            # Use same model paths
            general_model_path=self.general_model_path,
            ball_model_path=self.ball_model_path,
            field_keypoint_model_path=self.field_keypoint_model_path,
            
            # Optimize for goal detection accuracy
            ball_confidence_threshold=0.2,  # Very sensitive
            ball_fusion_confidence_threshold=0.35,
            high_confidence_threshold=0.6,
            
            # Enhanced temporal tracking
            ball_temporal_consistency_frames=7,
            temporal_smoothing_window=5,
            
            # Aggressive interpolation
            enable_enhanced_interpolation=True,
            interpolation_max_gap=12,
            interpolation_confidence_threshold=0.3,
            
            # Goal-specific boosts
            goal_area_ball_confidence_boost=0.3,
            near_goal_detection_sensitivity=2.0,
            
            # Enhanced trajectory prediction
            enable_trajectory_prediction=True,
            trajectory_prediction_frames=8,
            enable_occlusion_handling=True,
            occlusion_recovery_frames=15
        )
        return config


# Predefined configurations for different scenarios
GOAL_DETECTION_OPTIMIZED_CONFIG = EnhancedBallDetectionConfig(
    ball_confidence_threshold=0.2,
    ball_fusion_confidence_threshold=0.35,
    ball_temporal_consistency_frames=7,
    goal_area_ball_confidence_boost=0.3,
    near_goal_detection_sensitivity=2.0,
    enable_trajectory_prediction=True,
    trajectory_prediction_frames=8
)

PERFORMANCE_BALANCED_CONFIG = EnhancedBallDetectionConfig(
    ball_confidence_threshold=0.25,
    ball_fusion_confidence_threshold=0.4,
    ball_temporal_consistency_frames=5,
    goal_area_ball_confidence_boost=0.2,
    near_goal_detection_sensitivity=1.5,
    enable_trajectory_prediction=True,
    trajectory_prediction_frames=5
)

HIGH_ACCURACY_CONFIG = EnhancedBallDetectionConfig(
    ball_confidence_threshold=0.15,
    ball_fusion_confidence_threshold=0.3,
    ball_temporal_consistency_frames=10,
    temporal_smoothing_window=7,
    goal_area_ball_confidence_boost=0.4,
    near_goal_detection_sensitivity=2.5,
    enable_trajectory_prediction=True,
    trajectory_prediction_frames=10,
    interpolation_max_gap=15,
    occlusion_recovery_frames=20
)


def get_config_for_video_size(total_frames: int) -> EnhancedBallDetectionConfig:
    """
    Get optimized configuration based on video size.
    
    Args:
        total_frames: Total number of frames in the video
        
    Returns:
        Optimized configuration
    """
    if total_frames > 10000:
        # Large video - balance performance and accuracy
        return PERFORMANCE_BALANCED_CONFIG
    elif total_frames > 5000:
        # Medium video - optimize for goal detection
        return GOAL_DETECTION_OPTIMIZED_CONFIG
    else:
        # Small video - maximize accuracy
        return HIGH_ACCURACY_CONFIG


def save_enhanced_config(config: EnhancedBallDetectionConfig, config_path: str):
    """Save enhanced ball detection configuration to file."""
    import json
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_enhanced_config(config_path: str) -> EnhancedBallDetectionConfig:
    """Load enhanced ball detection configuration from file."""
    import json
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    return EnhancedBallDetectionConfig.from_dict(config_dict)
