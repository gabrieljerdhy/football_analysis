#!/usr/bin/env python3
"""
Debug Dribble Detection

This script helps debug the dribble detection system by adding detailed logging
to understand why dribbles are not being detected.
"""

import os
import sys

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dribble_detection import DribbleAnalyzer


def debug_dribble_detection():
    """Debug the dribble detection process step by step."""
    print("🔍 Debugging Dribble Detection Process")
    print("=" * 50)
    
    analyzer = DribbleAnalyzer(
        min_dribble_distance=10.0,  # Very low threshold
        min_dribble_duration=3,     # Very low threshold
        confidence_threshold=0.1    # Very low threshold
    )
    
    # Make thresholds very lenient
    analyzer.detector.opponent_proximity_threshold = 200.0
    analyzer.detector.ball_control_threshold = 100.0
    
    # Test data
    players = {
        1: {"bbox": [100, 100, 150, 200], "team": 1},
        2: {"bbox": [160, 100, 210, 200], "team": 2}  # Close opponent
    }
    
    print("Initial setup:")
    print(f"  Player 1 position: {players[1]['bbox']}")
    print(f"  Player 2 position: {players[2]['bbox']}")
    print(f"  Opponent proximity threshold: {analyzer.detector.opponent_proximity_threshold}")
    
    # Simulate movement over multiple frames
    for frame_num in range(1, 30):
        # Move player 1 significantly
        x_offset = frame_num * 8  # Larger movement
        y_offset = frame_num * 2
        
        players[1]["bbox"] = [100 + x_offset, 100 + y_offset, 150 + x_offset, 200 + y_offset]
        ball_position = (125 + x_offset, 150 + y_offset)
        
        print(f"\nFrame {frame_num}:")
        print(f"  Player 1 position: {players[1]['bbox']}")
        print(f"  Ball position: {ball_position}")
        
        # Check current possessions
        print(f"  Current possessions: {list(analyzer.detector.current_possessions.keys())}")
        print(f"  Potential dribbles: {list(analyzer.detector.potential_dribbles.keys())}")
        
        result = analyzer.analyze_frame(
            frame_num=frame_num,
            players=players,
            ball_position=ball_position,
            ball_possessor=1,
            ball_team=1
        )
        
        # Check what happened
        if frame_num in analyzer.detector.current_possessions:
            possession = analyzer.detector.current_possessions[frame_num]
            print(f"  Possession duration: {frame_num - possession['start_frame']} frames")
        
        if 1 in analyzer.detector.current_possessions:
            possession = analyzer.detector.current_possessions[1]
            print(f"  Player 1 possession duration: {frame_num - possession['start_frame']} frames")
        
        if 1 in analyzer.detector.potential_dribbles:
            dribble_data = analyzer.detector.potential_dribbles[1]
            print(f"  Potential dribble tracking: {len(dribble_data['positions'])} positions")
            print(f"  Total distance: {dribble_data['distances'][-1] if dribble_data['distances'] else 0}")
            print(f"  Direction changes: {dribble_data['direction_changes']}")
        
        if result:
            print(f"  🎉 DRIBBLE DETECTED!")
            print(f"    Player: {result.player_id}, Team: {result.team}")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Duration: {result.duration_frames} frames")
            print(f"    Distance: {result.distance_covered:.1f} pixels")
            return True
        else:
            print(f"  No dribble detected")
    
    print("\n❌ No dribble detected in debug test")
    return False


def debug_opponent_detection():
    """Debug opponent detection specifically."""
    print("\n🔍 Debugging Opponent Detection")
    print("=" * 30)
    
    analyzer = DribbleAnalyzer()
    analyzer.detector.opponent_proximity_threshold = 200.0
    
    players = {
        1: {"bbox": [100, 100, 150, 200], "team": 1},
        2: {"bbox": [160, 100, 210, 200], "team": 2}
    }
    
    # Test opponent detection
    nearby_opponents = analyzer.detector._find_nearby_opponents(players, 1, 1)
    
    print(f"Player 1 center: {(125, 150)}")
    print(f"Player 2 center: {(185, 150)}")
    print(f"Distance between players: ~60 pixels")
    print(f"Proximity threshold: {analyzer.detector.opponent_proximity_threshold}")
    print(f"Nearby opponents found: {len(nearby_opponents)}")
    
    for opponent in nearby_opponents:
        print(f"  Opponent {opponent['player_id']}: distance {opponent['distance']:.1f}")
    
    return len(nearby_opponents) > 0


def debug_possession_tracking():
    """Debug possession tracking."""
    print("\n🔍 Debugging Possession Tracking")
    print("=" * 30)
    
    analyzer = DribbleAnalyzer()
    
    players = {1: {"bbox": [100, 100, 150, 200], "team": 1}}
    ball_position = (125, 150)
    
    # Test possession tracking over multiple frames
    for frame_num in range(1, 10):
        analyzer.detector._update_possessions(
            frame_num, players, ball_position, 1, 1
        )
        
        if 1 in analyzer.detector.current_possessions:
            possession = analyzer.detector.current_possessions[1]
            duration = frame_num - possession['start_frame']
            print(f"Frame {frame_num}: Player 1 possession duration = {duration} frames")
        else:
            print(f"Frame {frame_num}: No possession tracked for player 1")
    
    return 1 in analyzer.detector.current_possessions


def main():
    """Run all debug tests."""
    print("🐛 Dribble Detection Debug Session")
    print("=" * 50)
    
    tests = [
        ("Opponent Detection", debug_opponent_detection),
        ("Possession Tracking", debug_possession_tracking),
        ("Full Dribble Detection", debug_dribble_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR: {test_name} - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Debug Summary:")
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")


if __name__ == "__main__":
    main()
