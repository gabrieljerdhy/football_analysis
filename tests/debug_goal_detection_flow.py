#!/usr/bin/env python3
"""
Debug the full goal detection flow for position (1599, 867).
"""

import sys
sys.path.append('src')

from goal_detection import FieldKeypointsDetector
from goal_detection.improved_goal_system import ImprovedGoalDetectionSystem

def debug_goal_detection_flow():
    """Debug the full goal detection flow."""
    print("🔍 DEBUGGING FULL GOAL DETECTION FLOW")
    print("=" * 50)
    
    # Initialize system
    field_keypoints_detector = FieldKeypointsDetector("data/models/best_field_keypoint.pt")
    improved_goal_system = ImprovedGoalDetectionSystem(field_keypoints_detector)
    
    # Initialize with fake frame
    import numpy as np
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    improved_goal_system.update_keypoints(fake_frame, force_detection=True)
    
    # Test the problematic position
    ball_position = (1599, 867)
    player_id = -1
    team = 1
    frame_num = 1200
    ball_confidence = 0.2
    ball_source = "interpolated_standard"
    
    print(f"Testing goal detection for:")
    print(f"  Ball position: {ball_position}")
    print(f"  Player ID: {player_id}")
    print(f"  Team: {team}")
    print(f"  Frame: {frame_num}")
    print(f"  Ball confidence: {ball_confidence}")
    print(f"  Ball source: {ball_source}")
    
    # Step 1: Check cooldown
    print(f"\n1. Checking cooldown...")
    cooldown_frames = improved_goal_system.detection_config['goal_cooldown_frames']
    last_goal_frame = improved_goal_system.last_goal_frame
    print(f"   Cooldown frames: {cooldown_frames}")
    print(f"   Last goal frame: {last_goal_frame}")
    
    if last_goal_frame >= 0 and frame_num - last_goal_frame < cooldown_frames:
        print(f"   ❌ BLOCKED BY COOLDOWN")
        return
    else:
        print(f"   ✅ Cooldown OK")
    
    # Step 2: Test multi-method detection
    print(f"\n2. Testing multi-method detection...")
    detection_results = improved_goal_system._detect_goal_multi_method(ball_position)
    
    if detection_results:
        goal_side, confidence, method = detection_results
        print(f"   ✅ Detection result: {goal_side} goal")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Method: {method}")
    else:
        print(f"   ❌ No detection result")
        return
    
    # Step 3: Test validation
    print(f"\n3. Testing validation...")
    is_valid = improved_goal_system._validate_goal_detection(
        goal_side, ball_position, confidence, ball_confidence
    )
    
    if is_valid:
        print(f"   ✅ Validation passed")
    else:
        print(f"   ❌ Validation failed")
        
        # Check specific validation criteria
        extended_threshold = improved_goal_system.detection_config["extended_confidence_threshold"]
        print(f"   Extended threshold: {extended_threshold}")
        print(f"   Confidence: {confidence}")
        print(f"   Confidence >= threshold: {confidence >= extended_threshold}")
        
        print(f"   Ball confidence: {ball_confidence}")
        print(f"   Ball confidence >= 0.1: {ball_confidence >= 0.1 if ball_confidence is not None else 'N/A'}")
        return
    
    # Step 4: Create goal event
    print(f"\n4. Creating goal event...")
    goal_event = improved_goal_system._create_goal_event(
        goal_side, player_id, team, frame_num, ball_position,
        confidence, method, ball_confidence, ball_source
    )
    
    print(f"   ✅ Goal event created:")
    print(f"   Frame: {goal_event.frame_num}")
    print(f"   Team: {goal_event.team}")
    print(f"   Side: {goal_event.goal_side}")
    print(f"   Confidence: {goal_event.confidence}")
    print(f"   Method: {goal_event.detection_method}")
    
    # Step 5: Test full detect_goal method
    print(f"\n5. Testing full detect_goal method...")
    
    # Reset system state
    improved_goal_system.goal_events = []
    improved_goal_system.last_goal_frame = -1
    improved_goal_system.ball_trajectory = []
    
    result = improved_goal_system.detect_goal(
        ball_position=ball_position,
        player_id=player_id,
        team=team,
        frame_num=frame_num,
        ball_confidence=ball_confidence,
        ball_source=ball_source,
    )
    
    if result:
        print(f"   ✅ GOAL DETECTED!")
        print(f"   Side: {result.goal_side}")
        print(f"   Team: {result.team}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Method: {result.detection_method}")
    else:
        print(f"   ❌ NO GOAL DETECTED")
        
        # Check trajectory length
        print(f"   Ball trajectory length: {len(improved_goal_system.ball_trajectory)}")
        min_length = improved_goal_system.detection_config['min_trajectory_length']
        print(f"   Min trajectory length: {min_length}")
        
        if len(improved_goal_system.ball_trajectory) < min_length:
            print(f"   ⚠️  Trajectory too short for some detection methods")

if __name__ == "__main__":
    debug_goal_detection_flow()
