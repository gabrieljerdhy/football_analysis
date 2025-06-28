#!/usr/bin/env python3
"""
Demo script for enhanced ball detection system.

This script demonstrates how to use the enhanced ball detection system
with the specialized ball model for improved accuracy.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trackers import Tracker
from config import BallDetectionConfig, ACCURACY_OPTIMIZED_CONFIG, PERFORMANCE_OPTIMIZED_CONFIG
from utils import read_video, save_video
from goal_detection import FieldKeypointsDetector, GoalDetector


def demo_basic_usage():
    """Demonstrate basic usage of enhanced ball detection."""
    print("🚀 Demo: Basic Enhanced Ball Detection Usage")
    print("=" * 50)
    
    # Initialize tracker with enhanced ball detection
    tracker = Tracker(
        model_path="data/models/best_detect.pt",
        ball_model_path="data/models/best_ball.pt",
        enable_enhanced_ball_detection=True,
        enable_jersey_detection=True
    )
    
    print(f"✅ Tracker initialized")
    print(f"   Enhanced ball detection: {tracker.enable_enhanced_ball_detection}")
    print(f"   Ball model: {tracker.ball_model_path}")
    print(f"   Confidence threshold: {tracker.ball_confidence_threshold}")
    print(f"   Fusion threshold: {tracker.ball_fusion_confidence_threshold}")


def demo_configuration_presets():
    """Demonstrate different configuration presets."""
    print("\n🔧 Demo: Configuration Presets")
    print("=" * 50)
    
    configs = {
        "Accuracy Optimized": ACCURACY_OPTIMIZED_CONFIG,
        "Performance Optimized": PERFORMANCE_OPTIMIZED_CONFIG,
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name} Configuration:")
        print(f"   Ball confidence threshold: {config.ball_confidence_threshold}")
        print(f"   Ball fusion threshold: {config.ball_fusion_confidence_threshold}")
        print(f"   Temporal consistency frames: {config.ball_temporal_consistency_frames}")
        print(f"   Batch size: {config.ball_detection_batch_size}")


def demo_custom_configuration():
    """Demonstrate custom configuration creation."""
    print("\n⚙️ Demo: Custom Configuration")
    print("=" * 50)
    
    # Create custom configuration for specific use case
    custom_config = BallDetectionConfig(
        ball_confidence_threshold=0.4,
        ball_fusion_confidence_threshold=0.6,
        ball_temporal_consistency_frames=5,
        ball_detection_batch_size=8,
        enable_enhanced_interpolation=True
    )
    
    print("📝 Custom Configuration Created:")
    print(f"   Ball confidence threshold: {custom_config.ball_confidence_threshold}")
    print(f"   Ball fusion threshold: {custom_config.ball_fusion_confidence_threshold}")
    print(f"   Temporal consistency frames: {custom_config.ball_temporal_consistency_frames}")
    print(f"   Enhanced interpolation: {custom_config.enable_enhanced_interpolation}")
    
    # Initialize tracker with custom config
    tracker = Tracker(config=custom_config)
    print("✅ Tracker initialized with custom configuration")


def demo_ball_quality_analysis(video_path=None):
    """Demonstrate ball detection quality analysis."""
    print("\n📊 Demo: Ball Detection Quality Analysis")
    print("=" * 50)
    
    if not video_path or not os.path.exists(video_path):
        print("⚠️ No video file provided or file not found. Using mock data.")
        return demo_mock_quality_analysis()
    
    # Read video frames (limit to first 100 for demo)
    video_frames = read_video(video_path)[:100]
    print(f"📹 Loaded {len(video_frames)} frames for analysis")
    
    # Initialize tracker with accuracy-optimized config
    tracker = Tracker(config=ACCURACY_OPTIMIZED_CONFIG)
    
    # Get tracks with enhanced ball detection
    tracks = tracker.get_object_tracks(video_frames)
    
    # Analyze ball detection quality
    quality_stats = analyze_ball_quality(tracks)
    
    print("\n📈 Ball Detection Quality Report:")
    print(f"   Total frames: {quality_stats['total_frames']}")
    print(f"   Specialized detections: {quality_stats['specialized_ratio']:.1%}")
    print(f"   Interpolated detections: {quality_stats['interpolated_ratio']:.1%}")
    print(f"   Missing detections: {quality_stats['missing_ratio']:.1%}")
    print(f"   Average confidence: {quality_stats['avg_confidence']:.3f}")
    print(f"   Minimum confidence: {quality_stats['min_confidence']:.3f}")


def demo_mock_quality_analysis():
    """Demonstrate quality analysis with mock data."""
    # Create mock ball tracks with different quality levels
    mock_tracks = {
        "ball": [
            {1: {"bbox": [100, 100, 120, 120], "confidence": 0.9, "source": "specialized"}},
            {1: {"bbox": [105, 105, 125, 125], "confidence": 0.8, "source": "specialized"}},
            {1: {"bbox": [110, 110, 130, 130], "confidence": 0.4, "source": "interpolated_high_conf"}},
            {1: {"bbox": [115, 115, 135, 135], "confidence": 0.6, "source": "general"}},
            {},  # Missing detection
            {1: {"bbox": [125, 125, 145, 145], "confidence": 0.2, "source": "interpolated_standard"}},
        ]
    }
    
    quality_stats = analyze_ball_quality(mock_tracks)
    
    print("\n📈 Mock Ball Detection Quality Report:")
    print(f"   Total frames: {quality_stats['total_frames']}")
    print(f"   Specialized detections: {quality_stats['specialized_ratio']:.1%}")
    print(f"   Interpolated detections: {quality_stats['interpolated_ratio']:.1%}")
    print(f"   Missing detections: {quality_stats['missing_ratio']:.1%}")
    print(f"   Average confidence: {quality_stats['avg_confidence']:.3f}")
    print(f"   Minimum confidence: {quality_stats['min_confidence']:.3f}")


def analyze_ball_quality(tracks):
    """Analyze ball detection quality from tracks."""
    total_frames = len(tracks["ball"])
    specialized_count = 0
    interpolated_count = 0
    missing_count = 0
    confidence_scores = []
    
    for ball_track in tracks["ball"]:
        ball_info = ball_track.get(1, {})
        if not ball_info:
            missing_count += 1
            continue
            
        source = ball_info.get("source", "unknown")
        confidence = ball_info.get("confidence", 0.0)
        
        if source == "specialized":
            specialized_count += 1
        elif "interpolated" in source:
            interpolated_count += 1
            
        confidence_scores.append(confidence)
    
    return {
        "total_frames": total_frames,
        "specialized_ratio": specialized_count / total_frames if total_frames > 0 else 0,
        "interpolated_ratio": interpolated_count / total_frames if total_frames > 0 else 0,
        "missing_ratio": missing_count / total_frames if total_frames > 0 else 0,
        "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
        "min_confidence": np.min(confidence_scores) if confidence_scores else 0.0
    }


def demo_goal_detection_enhancement():
    """Demonstrate enhanced goal detection with ball quality."""
    print("\n⚽ Demo: Enhanced Goal Detection")
    print("=" * 50)
    
    # Initialize components
    tracker = Tracker(config=ACCURACY_OPTIMIZED_CONFIG)
    field_detector = FieldKeypointsDetector("data/models/best_keypoint.pt")
    goal_detector = GoalDetector(field_detector)
    
    print("✅ Goal detection components initialized")
    
    # Mock ball position near goal
    ball_position = (50, 300)  # Left side of field
    player_id = 7
    team = 1
    frame_num = 1250
    
    # Simulate different ball detection qualities
    scenarios = [
        {"confidence": 0.9, "source": "specialized", "description": "High-quality specialized detection"},
        {"confidence": 0.6, "source": "general", "description": "Medium-quality general detection"},
        {"confidence": 0.3, "source": "interpolated_standard", "description": "Low-quality interpolated detection"},
    ]
    
    print("\n🎯 Goal Detection Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['description']}")
        print(f"   Ball confidence: {scenario['confidence']}")
        print(f"   Ball source: {scenario['source']}")
        
        # Simulate goal detection (would normally require actual frame and field setup)
        print(f"   → Enhanced validation would consider ball quality in trajectory analysis")
        print(f"   → Goal confidence would be adjusted based on detection source")


def main():
    """Run all demo functions."""
    print("🎬 Enhanced Ball Detection System Demo")
    print("=" * 60)
    
    # Run demos
    demo_basic_usage()
    demo_configuration_presets()
    demo_custom_configuration()
    
    # Check for test video
    test_video_paths = [
        "data/input_videos/test_video.mp4",
        "data/input_videos/sample.mp4",
        "input_videos/test.mp4"
    ]
    
    video_path = None
    for path in test_video_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    demo_ball_quality_analysis(video_path)
    demo_goal_detection_enhancement()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! Enhanced ball detection system is ready for use.")
    print("\n📚 Next steps:")
    print("   1. Run your video analysis with enhanced ball detection")
    print("   2. Monitor ball detection quality in the output")
    print("   3. Adjust configuration based on your requirements")
    print("   4. Review goal detections with ball quality information")
    print("\n📖 For more information, see:")
    print("   - docs/ENHANCED_BALL_DETECTION.md")
    print("   - docs/BALL_DETECTION_RECOMMENDATIONS.md")


if __name__ == "__main__":
    main()
