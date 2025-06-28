"""
Scoreboard Analyzer

This module combines scoreboard detection and score extraction to provide
a complete scoreboard analysis solution for football videos.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .scoreboard_detector import ScoreboardDetector
from .score_extractor import ScoreExtractor


class ScoreboardAnalyzer:
    """
    Main class that combines scoreboard detection and score extraction.
    
    This class orchestrates the entire scoreboard analysis pipeline:
    1. Detects scoreboard regions in video frames
    2. Extracts scores from detected regions
    3. Tracks score changes over time
    4. Provides final authoritative score information
    """
    
    def __init__(self,
                 detection_interval: int = 30,  # Analyze every 30 frames
                 min_detection_confidence: float = 0.6,
                 min_extraction_confidence: float = 0.6,
                 score_stability_frames: int = 5):
        """
        Initialize the scoreboard analyzer.
        
        Args:
            detection_interval: How often to run detection (every N frames)
            min_detection_confidence: Minimum confidence for scoreboard detection
            min_extraction_confidence: Minimum confidence for score extraction
            score_stability_frames: Number of frames a score must be stable to be accepted
        """
        self.detection_interval = detection_interval
        self.min_detection_confidence = min_detection_confidence
        self.min_extraction_confidence = min_extraction_confidence
        self.score_stability_frames = score_stability_frames
        
        # Initialize components
        self.detector = ScoreboardDetector(confidence_threshold=min_detection_confidence)
        self.extractor = ScoreExtractor(min_score_confidence=min_extraction_confidence)
        
        # Score tracking
        self.score_history = []
        self.current_score = None
        self.score_changes = []
        self.frames_processed = 0
        
        # Statistics
        self.total_detections = 0
        self.successful_extractions = 0
        self.scoreboard_detected = False
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int) -> Optional[Dict]:
        """
        Analyze a single frame for scoreboard information.
        
        Args:
            frame: Input video frame
            frame_number: Frame number in the video
            
        Returns:
            Score information if detected, None otherwise
        """
        self.frames_processed += 1
        
        # Only analyze every N frames for efficiency
        if frame_number % self.detection_interval != 0:
            return self.current_score
        
        # Detect scoreboard regions
        regions = self.detector.detect_scoreboard_regions(frame)
        
        if regions:
            self.total_detections += 1
            self.scoreboard_detected = True
            
            # Extract scores from detected regions
            extracted_scores = self.extractor.extract_scores(frame, regions)
            
            if extracted_scores:
                self.successful_extractions += 1
                
                # Get the best score
                best_score = self.extractor.get_best_score(extracted_scores)
                
                if best_score:
                    # Update score tracking
                    self._update_score_tracking(best_score, frame_number)
                    
                    return {
                        'frame_number': frame_number,
                        'team1_score': best_score['team1_score'],
                        'team2_score': best_score['team2_score'],
                        'confidence': best_score['confidence'],
                        'detection_method': 'scoreboard',
                        'regions_detected': len(regions),
                        'scores_extracted': len(extracted_scores)
                    }
        
        return self.current_score
    
    def analyze_video_batch(self, frames: List[np.ndarray], 
                           start_frame_number: int) -> List[Optional[Dict]]:
        """
        Analyze a batch of frames for scoreboard information.
        
        Args:
            frames: List of video frames
            start_frame_number: Starting frame number for this batch
            
        Returns:
            List of score information for each frame
        """
        results = []
        
        for i, frame in enumerate(frames):
            frame_number = start_frame_number + i
            result = self.analyze_frame(frame, frame_number)
            results.append(result)
        
        return results
    
    def _update_score_tracking(self, score: Dict, frame_number: int):
        """Update internal score tracking with new detection."""
        score_tuple = (score['team1_score'], score['team2_score'])
        
        # Add to history
        self.score_history.append({
            'frame_number': frame_number,
            'score': score_tuple,
            'confidence': score['confidence']
        })
        
        # Check if this is a new stable score
        if self._is_score_stable(score_tuple):
            if self.current_score is None or (
                self.current_score['team1_score'] != score['team1_score'] or
                self.current_score['team2_score'] != score['team2_score']
            ):
                # Score change detected
                old_score = self.current_score
                self.current_score = score.copy()
                self.current_score['frame_number'] = frame_number
                
                self.score_changes.append({
                    'frame_number': frame_number,
                    'old_score': old_score,
                    'new_score': self.current_score,
                    'confidence': score['confidence']
                })
                
                self.logger.info(f"Score change detected at frame {frame_number}: "
                               f"{score['team1_score']}-{score['team2_score']}")
    
    def _is_score_stable(self, score_tuple: Tuple[int, int]) -> bool:
        """Check if a score has been stable for enough frames."""
        if len(self.score_history) < self.score_stability_frames:
            return False
        
        # Check last N detections
        recent_scores = self.score_history[-self.score_stability_frames:]
        
        # Count how many times this score appears in recent history
        count = sum(1 for entry in recent_scores if entry['score'] == score_tuple)
        
        # Score is stable if it appears in majority of recent frames
        return count >= (self.score_stability_frames * 0.6)
    
    def get_final_score(self) -> Optional[Dict]:
        """
        Get the final authoritative score from scoreboard analysis.
        
        Returns:
            Final score information or None if no reliable score detected
        """
        if not self.scoreboard_detected or not self.current_score:
            return None
        
        # Calculate overall confidence based on detection statistics
        detection_rate = self.successful_extractions / max(self.total_detections, 1)
        overall_confidence = self.current_score['confidence'] * detection_rate
        
        return {
            'team1_score': self.current_score['team1_score'],
            'team2_score': self.current_score['team2_score'],
            'confidence': overall_confidence,
            'detection_method': 'scoreboard',
            'frames_analyzed': self.frames_processed,
            'total_detections': self.total_detections,
            'successful_extractions': self.successful_extractions,
            'score_changes': len(self.score_changes),
            'detection_rate': detection_rate
        }
    
    def get_score_timeline(self) -> List[Dict]:
        """Get the timeline of score changes detected."""
        return self.score_changes.copy()
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics."""
        return {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'successful_extractions': self.successful_extractions,
            'scoreboard_detected': self.scoreboard_detected,
            'detection_rate': self.successful_extractions / max(self.total_detections, 1),
            'score_changes_detected': len(self.score_changes),
            'current_score': self.current_score,
            'analysis_interval': self.detection_interval
        }
    
    def reset(self):
        """Reset the analyzer state for a new video."""
        self.score_history.clear()
        self.current_score = None
        self.score_changes.clear()
        self.frames_processed = 0
        self.total_detections = 0
        self.successful_extractions = 0
        self.scoreboard_detected = False
        
        # Reset detector history
        self.detector.detection_history.clear()
    
    def set_detection_interval(self, interval: int):
        """Update the detection interval."""
        self.detection_interval = max(1, interval)
    
    def is_scoreboard_available(self) -> bool:
        """Check if scoreboard has been detected in the video."""
        return self.scoreboard_detected
    
    def get_confidence_threshold_recommendation(self) -> float:
        """
        Get recommended confidence threshold based on detection performance.
        
        Returns:
            Recommended confidence threshold
        """
        if self.total_detections == 0:
            return self.min_detection_confidence
        
        detection_rate = self.successful_extractions / self.total_detections
        
        if detection_rate > 0.8:
            # High success rate, can be more strict
            return min(self.min_detection_confidence + 0.1, 0.9)
        elif detection_rate < 0.3:
            # Low success rate, be more lenient
            return max(self.min_detection_confidence - 0.1, 0.3)
        else:
            return self.min_detection_confidence
