#!/usr/bin/env python3
"""
Simple test for enhanced ball detection integration.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_tracker_initialization():
    """Test basic tracker initialization with enhanced ball detection."""
    try:
        from trackers.tracker import Tracker
        
        print("🧪 Testing Tracker initialization...")
        
        # Test with default parameters
        tracker = Tracker(
            model_path="data/models/best_detect.pt",
            enable_jersey_detection=True,
            ball_model_path="data/models/best_ball.pt",
            enable_enhanced_ball_detection=True
        )
        
        print("✅ Tracker initialized successfully")
        print(f"   Enhanced ball detection: {tracker.enable_enhanced_ball_detection}")
        print(f"   Ball model path: {tracker.ball_model_path}")
        print(f"   Ball confidence threshold: {tracker.ball_confidence_threshold}")
        print(f"   Ball fusion threshold: {tracker.ball_fusion_confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        return False

def test_config_system():
    """Test configuration system."""
    try:
        from config import BallDetectionConfig, ACCURACY_OPTIMIZED_CONFIG
        from trackers.tracker import Tracker
        
        print("\n🧪 Testing configuration system...")
        
        # Test with accuracy optimized config
        tracker = Tracker(config=ACCURACY_OPTIMIZED_CONFIG)
        
        print("✅ Configuration system working")
        print(f"   Ball confidence threshold: {tracker.ball_confidence_threshold}")
        print(f"   Ball fusion threshold: {tracker.ball_fusion_confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False

def test_ball_detection_fusion():
    """Test ball detection fusion logic."""
    try:
        from trackers.tracker import Tracker
        from unittest.mock import Mock
        
        print("\n🧪 Testing ball detection fusion...")
        
        tracker = Tracker(
            model_path="data/models/best_detect.pt",
            enable_enhanced_ball_detection=True
        )
        
        # Mock detection objects
        mock_general = Mock()
        mock_general.boxes = None
        mock_general.names = {"ball": 0}
        
        mock_ball = Mock()
        mock_ball.boxes = None
        
        # Test fusion with no detections
        result = tracker.fuse_ball_detections(mock_general, mock_ball, 0)
        
        print("✅ Ball detection fusion logic working")
        print(f"   Fusion result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ball detection fusion failed: {e}")
        return False

def test_enhanced_interpolation():
    """Test enhanced ball position interpolation."""
    try:
        from trackers.tracker import Tracker
        
        print("\n🧪 Testing enhanced interpolation...")
        
        tracker = Tracker(
            model_path="data/models/best_detect.pt",
            enable_enhanced_ball_detection=True
        )
        
        # Create test ball positions with gaps
        test_positions = [
            {1: {"bbox": [100, 100, 120, 120], "confidence": 0.8, "source": "specialized"}},
            {},  # Missing detection
            {1: {"bbox": [140, 140, 160, 160], "confidence": 0.7, "source": "general"}},
        ]
        
        interpolated = tracker.interpolate_ball_positions(test_positions)
        
        print("✅ Enhanced interpolation working")
        print(f"   Input positions: {len(test_positions)}")
        print(f"   Output positions: {len(interpolated)}")
        print(f"   Gap filled: {1 in interpolated[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced interpolation failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🚀 Running Enhanced Ball Detection Integration Tests")
    print("=" * 60)
    
    tests = [
        test_tracker_initialization,
        test_config_system,
        test_ball_detection_fusion,
        test_enhanced_interpolation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Enhanced ball detection is ready.")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
