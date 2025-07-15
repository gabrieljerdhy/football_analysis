#!/usr/bin/env python3
"""
Simple test to verify detection logic.
"""

import sys
sys.path.append('src')

from goal_detection import FieldKeypointsDetector
from goal_detection.improved_goal_system import ImprovedGoalDetectionSystem

def simple_test():
    """Simple test of detection logic."""
    print("🧪 SIMPLE DETECTION TEST")
    print("=" * 30)
    
    # Initialize system
    field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
    improved_goal_system = ImprovedGoalDetectionSystem(field_keypoints_detector)
    
    # Initialize with fake frame
    import numpy as np
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    improved_goal_system.update_keypoints(fake_frame, force_detection=True)
    
    # Test position
    ball_position = (1599, 867)
    x, y = ball_position
    
    print(f"Testing position: {ball_position}")
    
    # Manual calculation
    areas = improved_goal_system.enhanced_goal_areas
    weights = improved_goal_system.detection_weights
    
    detection_scores = {"left": 0.0, "right": 0.0}
    detection_methods = {"left": [], "right": []}
    
    print(f"\nDetection weights: {weights}")
    
    # Method 1: Keypoints
    keypoint_result = field_keypoints_detector.is_ball_in_goal_area(ball_position)
    print(f"Keypoints result: {keypoint_result}")
    if keypoint_result:
        detection_scores[keypoint_result] += weights["keypoints"]
        detection_methods[keypoint_result].append("keypoints")
    
    # Method 2: Primary areas
    print(f"\nPrimary areas:")
    for side in ["left", "right"]:
        primary_area = areas[side]["primary"]
        in_primary = (primary_area["x_min"] <= x <= primary_area["x_max"] and
                     primary_area["y_min"] <= y <= primary_area["y_max"])
        print(f"  {side}: {primary_area} -> {in_primary}")
        if in_primary:
            detection_scores[side] += weights["primary_areas"]
            detection_methods[side].append("primary_areas")
    
    # Method 3: Extended areas
    print(f"\nExtended areas:")
    for side in ["left", "right"]:
        extended_area = areas[side]["extended"]
        in_extended = (extended_area["x_min"] <= x <= extended_area["x_max"] and
                      extended_area["y_min"] <= y <= extended_area["y_max"])
        print(f"  {side}: {extended_area} -> {in_extended}")
        if in_extended:
            detection_scores[side] += weights["extended_areas"]
            detection_methods[side].append("extended_areas")
    
    # Method 4: Trajectory areas
    print(f"\nTrajectory areas:")
    for side in ["left", "right"]:
        trajectory_area = areas[side]["trajectory"]
        in_trajectory = (trajectory_area["x_min"] <= x <= trajectory_area["x_max"] and
                        trajectory_area["y_min"] <= y <= trajectory_area["y_max"])
        print(f"  {side}: {trajectory_area} -> {in_trajectory}")
        if in_trajectory:
            detection_scores[side] += weights["trajectory"]
            detection_methods[side].append("trajectory")
    
    print(f"\nFinal scores:")
    for side in ["left", "right"]:
        methods_str = "+".join(detection_methods[side]) if detection_methods[side] else "none"
        print(f"  {side}: {detection_scores[side]:.3f} ({methods_str})")
    
    # Find best
    best_side = max(detection_scores, key=detection_scores.get)
    best_score = detection_scores[best_side]
    
    print(f"\nBest: {best_side} with score {best_score:.3f}")
    
    # Check thresholds
    config = improved_goal_system.detection_config
    primary_threshold = config["primary_confidence_threshold"]
    extended_threshold = config["extended_confidence_threshold"]
    
    print(f"\nThresholds:")
    print(f"  Primary: {primary_threshold}")
    print(f"  Extended: {extended_threshold}")
    
    print(f"\nThreshold checks:")
    print(f"  Score >= primary: {best_score >= primary_threshold}")
    print(f"  Score >= extended: {best_score >= extended_threshold}")
    
    # Test the actual method
    print(f"\nTesting actual _detect_goal_multi_method:")
    result = improved_goal_system._detect_goal_multi_method(ball_position)
    if result:
        side, confidence, method = result
        print(f"  Result: {side} goal, confidence: {confidence:.3f}, method: {method}")
    else:
        print(f"  Result: None")

if __name__ == "__main__":
    simple_test()
