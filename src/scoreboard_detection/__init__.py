"""
Scoreboard Detection Module

This module provides functionality to detect scoreboards in football/soccer video frames
and extract score information using computer vision and OCR techniques.

Classes:
    ScoreboardDetector: Detects scoreboard regions in video frames
    ScoreExtractor: Extracts numerical scores from detected scoreboard regions
    ScoreboardAnalyzer: Main class that combines detection and extraction
"""

from .scoreboard_detector import ScoreboardDetector
from .score_extractor import ScoreExtractor
from .scoreboard_analyzer import ScoreboardAnalyzer

__all__ = ["ScoreboardDetector", "ScoreExtractor", "ScoreboardAnalyzer"]
