"""
Cross Detection Demo

Demonstrates the cross detection feature with sample data and shows
how to use the cross detection API.
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cross_detection import CrossDetector, FieldZoneAnalyzer, CrossTrajectoryAnalyzer
from cross_detection.cross_event import CrossType, CrossOrigin, CrossTarget


def demo_field_zones():
    """Demonstrate field zone detection."""
    print("🏟️  Field Zone Detection Demo")
    print("=" * 40)
    
    analyzer = FieldZoneAnalyzer(video_width=1920, video_height=1080)
    
    # Test various positions
    test_positions = [
        ((400, 540), "Left Wing"),
        ((1520, 540), "Right Wing"),
        ((200, 540), "Left Byline"),
        ((1720, 540), "Right Byline"),
        ((960, 540), "Center Field"),
        ((300, 540), "Left Penalty Area"),
        ((1620, 540), "Right Penalty Area"),
    ]
    
    for position, description in test_positions:
        origin_zone = analyzer.get_cross_origin_zone(position)
        target_zone = analyzer.get_cross_target_zone(position)
        
        print(f"{description:20} {position}")
        print(f"  Origin Zone: {origin_zone.value if origin_zone else 'None'}")
        print(f"  Target Zone: {target_zone.value if target_zone else 'None'}")
        print()


def demo_trajectory_analysis():
    """Demonstrate trajectory analysis for crosses."""
    print("📈 Trajectory Analysis Demo")
    print("=" * 40)
    
    analyzer = CrossTrajectoryAnalyzer(frame_rate=24.0)
    
    # Create sample trajectories
    trajectories = {
        "Wing Cross": create_wing_cross_trajectory(),
        "High Cross": create_high_cross_trajectory(),
        "Low Cross": create_low_cross_trajectory(),
        "Cutback": create_cutback_trajectory(),
    }
    
    for name, trajectory in trajectories.items():
        print(f"\n{name}:")
        result = analyzer.analyze_trajectory_for_cross(trajectory)
        
        if result and result.get("is_cross"):
            print(f"  ✅ Cross detected!")
            print(f"  Score: {result['cross_score']:.3f}")
            print(f"  Type: {result['cross_type'].value}")
            print(f"  Distance: {result['distance']:.1f} pixels")
            print(f"  Max Height: {result['height_analysis']['estimated_max_height']:.1f}")
        else:
            print(f"  ❌ Not detected as cross")


def demo_cross_detector():
    """Demonstrate the main cross detector."""
    print("🎯 Cross Detector Demo")
    print("=" * 40)
    
    detector = CrossDetector(video_width=1920, video_height=1080, frame_rate=24.0)
    
    # Simulate a cross sequence
    print("\nSimulating cross sequence...")
    
    # Player starts with ball in left wing
    player_id = 7
    team_id = 1
    
    cross_positions = [
        (400, 540),   # Start in left wing
        (450, 535),   # Moving toward center
        (500, 530),   # Continuing
        (550, 525),   # Getting closer to penalty area
        (600, 520),   # End in penalty area
    ]
    
    for frame_num, position in enumerate(cross_positions):
        print(f"Frame {frame_num}: Ball at {position}")
        
        cross_event = detector.detect_cross(
            ball_position=position,
            player_id=player_id,
            team_id=team_id,
            frame_num=frame_num,
            ball_confidence=0.8
        )
        
        if cross_event:
            print(f"  🎯 Cross detected!")
            print(f"  Player: {cross_event.player_id}, Team: {cross_event.team_id}")
            print(f"  Type: {cross_event.cross_type.value}")
            print(f"  Origin: {cross_event.origin_zone.value}")
            print(f"  Target: {cross_event.target_zone.value}")
            print(f"  Confidence: {cross_event.confidence_score:.3f}")
            break
    
    # Trigger analysis by changing possession
    print(f"\nChanging possession to trigger analysis...")
    cross_event = detector.detect_cross(
        ball_position=(650, 515),
        player_id=9,  # Different player
        team_id=team_id,
        frame_num=len(cross_positions),
        ball_confidence=0.8
    )
    
    if cross_event:
        print(f"  🎯 Cross detected on possession change!")
        print(f"  Type: {cross_event.cross_type.value}")
        print(f"  Success: {cross_event.is_successful}")
    
    # Show statistics
    stats = detector.get_cross_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total crosses: {stats['total_crosses']}")
    team_stats = stats['team_statistics'].get(team_id, {})
    print(f"  Team {team_id} crosses: {team_stats.get('crosses_attempted', 0)}")
    print(f"  Success rate: {team_stats.get('cross_accuracy_percentage', 0):.1f}%")


def create_wing_cross_trajectory():
    """Create a sample wing cross trajectory."""
    trajectory = []
    start_x, start_y = 400, 540  # Left wing
    end_x, end_y = 600, 520      # Penalty box
    
    for i in range(8):
        t = i / 7.0
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t + 15 * np.sin(t * np.pi)  # Slight arc
        
        trajectory.append({
            "position": (int(x), int(y)),
            "frame_num": i,
            "confidence": 0.8,
            "source": "detection"
        })
    
    return trajectory


def create_high_cross_trajectory():
    """Create a sample high cross trajectory."""
    trajectory = []
    start_x, start_y = 200, 540  # Left byline
    end_x, end_y = 500, 520      # Central penalty area
    
    for i in range(10):
        t = i / 9.0
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t + 40 * np.sin(t * np.pi)  # High arc
        
        trajectory.append({
            "position": (int(x), int(y)),
            "frame_num": i,
            "confidence": 0.8,
            "source": "detection"
        })
    
    return trajectory


def create_low_cross_trajectory():
    """Create a sample low cross trajectory."""
    trajectory = []
    start_x, start_y = 1520, 540  # Right wing
    end_x, end_y = 1320, 530     # Left side of penalty area
    
    for i in range(6):
        t = i / 5.0
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t + 5 * np.sin(t * np.pi)  # Low arc
        
        trajectory.append({
            "position": (int(x), int(y)),
            "frame_num": i,
            "confidence": 0.8,
            "source": "detection"
        })
    
    return trajectory


def create_cutback_trajectory():
    """Create a sample cutback trajectory."""
    trajectory = []
    
    # Start near byline, move toward goal, then cut back
    positions = [
        (200, 540),   # Near byline
        (250, 535),   # Moving toward goal
        (300, 530),   # Continuing
        (280, 540),   # Start cutting back
        (260, 550),   # Cutting back more
        (240, 560),   # Final cutback position
    ]
    
    for i, position in enumerate(positions):
        trajectory.append({
            "position": position,
            "frame_num": i,
            "confidence": 0.8,
            "source": "detection"
        })
    
    return trajectory


def main():
    """Run the cross detection demo."""
    print("🚀 Cross Detection Feature Demo")
    print("=" * 50)
    print()
    
    try:
        demo_field_zones()
        print()
        
        demo_trajectory_analysis()
        print()
        
        demo_cross_detector()
        print()
        
        print("🎉 Demo completed successfully!")
        print("\nTo use cross detection in your analysis:")
        print("1. Run main.py with your video file")
        print("2. Check the CSV output for cross statistics")
        print("3. Look for columns like 'crosses_attempted', 'cross_accuracy_percentage'")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
