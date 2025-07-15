#!/usr/bin/env python3
"""
Debug specific ball position (1599, 867) to understand why it's not detected.
"""

import sys
sys.path.append('src')

from goal_detection import FieldKeypointsDetector
from goal_detection.improved_goal_system import ImprovedGoalDetectionSystem

def debug_position():
    """Debug specific position detection."""
    print("🔍 DEBUGGING POSITION (1599, 867)")
    print("=" * 40)
    
    # Initialize system
    field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
    improved_goal_system = ImprovedGoalDetectionSystem(field_keypoints_detector)
    
    # Initialize with fake frame
    import numpy as np
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    improved_goal_system.update_keypoints(fake_frame, force_detection=True)
    
    # Test position
    test_position = (1599, 867)
    print(f"Testing position: {test_position}")
    
    # Check enhanced goal areas
    areas = improved_goal_system.enhanced_goal_areas
    
    print(f"\nGoal areas:")
    for side in ['left', 'right']:
        print(f"\n{side.upper()} GOAL:")
        for area_type in ['primary', 'extended', 'trajectory']:
            area = areas[side][area_type]
            print(f"  {area_type}: x={area['x_min']}-{area['x_max']}, y={area['y_min']}-{area['y_max']}")
            
            x, y = test_position
            in_area = (area['x_min'] <= x <= area['x_max'] and area['y_min'] <= y <= area['y_max'])
            print(f"    Position {test_position} in area: {in_area}")
    
    # Test multi-method detection
    print(f"\nTesting multi-method detection:")
    result = improved_goal_system._detect_goal_multi_method(test_position)
    if result:
        side, confidence, method = result
        print(f"  Result: {side} goal (confidence: {confidence:.3f}, method: {method})")
    else:
        print(f"  Result: No goal detected")
    
    # Test individual methods
    print(f"\nTesting individual methods:")
    
    # Method 1: Keypoints
    keypoint_result = field_keypoints_detector.is_ball_in_goal_area(test_position)
    print(f"  Keypoints: {keypoint_result}")
    
    # Method 2-4: Area checks
    x, y = test_position
    detection_scores = {'left': 0.0, 'right': 0.0}
    detection_methods = {'left': [], 'right': []}
    
    # Primary areas
    for side in ['left', 'right']:
        primary_area = areas[side]['primary']
        if (primary_area['x_min'] <= x <= primary_area['x_max'] and
            primary_area['y_min'] <= y <= primary_area['y_max']):
            detection_scores[side] += 0.3
            detection_methods[side].append('primary_areas')
            print(f"  Primary {side}: YES")
        else:
            print(f"  Primary {side}: NO")
    
    # Extended areas
    for side in ['left', 'right']:
        extended_area = areas[side]['extended']
        if (extended_area['x_min'] <= x <= extended_area['x_max'] and
            extended_area['y_min'] <= y <= extended_area['y_max']):
            detection_scores[side] += 0.2
            detection_methods[side].append('extended_areas')
            print(f"  Extended {side}: YES")
        else:
            print(f"  Extended {side}: NO")
    
    # Trajectory areas
    for side in ['left', 'right']:
        trajectory_area = areas[side]['trajectory']
        if (trajectory_area['x_min'] <= x <= trajectory_area['x_max'] and
            trajectory_area['y_min'] <= y <= trajectory_area['y_max']):
            detection_scores[side] += 0.1
            detection_methods[side].append('trajectory')
            print(f"  Trajectory {side}: YES")
        else:
            print(f"  Trajectory {side}: NO")
    
    print(f"\nFinal scores:")
    for side in ['left', 'right']:
        print(f"  {side}: {detection_scores[side]:.3f} ({'+'.join(detection_methods[side]) if detection_methods[side] else 'none'})")
    
    # Check thresholds
    config = improved_goal_system.detection_config
    print(f"\nThresholds:")
    print(f"  Primary: {config['primary_confidence_threshold']}")
    print(f"  Extended: {config['extended_confidence_threshold']}")
    
    best_score = max(detection_scores.values())
    print(f"\nBest score: {best_score:.3f}")
    print(f"Meets primary threshold: {best_score >= config['primary_confidence_threshold']}")
    print(f"Meets extended threshold: {best_score >= config['extended_confidence_threshold']}")

if __name__ == "__main__":
    debug_position()
