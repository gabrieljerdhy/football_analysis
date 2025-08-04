"""
Cross Detection Module for Football Analysis

This module provides comprehensive cross detection capabilities for football video analysis,
including trajectory analysis, field zone detection, and cross classification.
"""

from .cross_detector import CrossDetector
from .cross_event import CrossEvent
from .field_zone_analyzer import FieldZoneAnalyzer
from .cross_trajectory_analyzer import CrossTrajectoryAnalyzer

__all__ = [
    "CrossDetector",
    "CrossEvent", 
    "FieldZoneAnalyzer",
    "CrossTrajectoryAnalyzer"
]
