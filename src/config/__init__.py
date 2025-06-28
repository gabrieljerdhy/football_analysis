"""
Configuration module for football analysis system.
"""

from .ball_detection_config import (
    BallDetectionConfig,
    DEFAULT_BALL_DETECTION_CONFIG,
    PERFORMANCE_OPTIMIZED_CONFIG,
    ACCURACY_OPTIMIZED_CONFIG,
    BALANCED_CONFIG,
    get_ball_detection_config,
    save_ball_detection_config
)

__all__ = [
    'BallDetectionConfig',
    'DEFAULT_BALL_DETECTION_CONFIG',
    'PERFORMANCE_OPTIMIZED_CONFIG',
    'ACCURACY_OPTIMIZED_CONFIG',
    'BALANCED_CONFIG',
    'get_ball_detection_config',
    'save_ball_detection_config'
]
