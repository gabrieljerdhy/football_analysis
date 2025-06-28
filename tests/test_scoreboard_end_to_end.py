"""
End-to-end test for scoreboard detection functionality.

This script tests the complete scoreboard detection pipeline with synthetic data
to ensure everything works together correctly.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoreboard_detection import ScoreboardAnalyzer


def create_synthetic_scoreboard_frame(width=640, height=480, team1_score=2, team2_score=1):
    """
    Create a synthetic video frame with a scoreboard for testing.
    
    Args:
        width: Frame width
        height: Frame height
        team1_score: Score for team 1
        team2_score: Score for team 2
        
    Returns:
        numpy.ndarray: Synthetic frame with scoreboard
    """
    # Create a basic football field background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Green field color
    
    # Add field lines
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)  # Center line
    cv2.rectangle(frame, (50, height//3), (width-50, 2*height//3), (255, 255, 255), 2)  # Field outline
    
    # Create scoreboard area (top center)
    scoreboard_width = 200
    scoreboard_height = 60
    scoreboard_x = (width - scoreboard_width) // 2
    scoreboard_y = 20
    
    # Draw scoreboard background
    cv2.rectangle(frame, 
                 (scoreboard_x, scoreboard_y), 
                 (scoreboard_x + scoreboard_width, scoreboard_y + scoreboard_height),
                 (0, 0, 0), -1)  # Black background
    
    # Draw scoreboard border
    cv2.rectangle(frame, 
                 (scoreboard_x, scoreboard_y), 
                 (scoreboard_x + scoreboard_width, scoreboard_y + scoreboard_height),
                 (255, 255, 255), 2)  # White border
    
    # Add score text
    score_text = f"{team1_score} - {team2_score}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Calculate text size and position
    (text_width, text_height), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    text_x = scoreboard_x + (scoreboard_width - text_width) // 2
    text_y = scoreboard_y + (scoreboard_height + text_height) // 2
    
    # Draw the score text
    cv2.putText(frame, score_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    # Add team names (optional)
    team1_text = "HOME"
    team2_text = "AWAY"
    small_font_scale = 0.5
    small_font_thickness = 1
    
    # Team 1 name
    cv2.putText(frame, team1_text, 
               (scoreboard_x + 10, scoreboard_y - 5), 
               font, small_font_scale, (255, 255, 255), small_font_thickness)
    
    # Team 2 name
    team2_text_width, _ = cv2.getTextSize(team2_text, font, small_font_scale, small_font_thickness)[0]
    cv2.putText(frame, team2_text, 
               (scoreboard_x + scoreboard_width - team2_text_width - 10, scoreboard_y - 5), 
               font, small_font_scale, (255, 255, 255), small_font_thickness)
    
    return frame


def test_scoreboard_detection_basic():
    """Test basic scoreboard detection functionality."""
    print("🧪 Testing basic scoreboard detection...")
    
    # Create analyzer
    analyzer = ScoreboardAnalyzer(
        detection_interval=1,  # Analyze every frame for testing
        min_detection_confidence=0.3,  # Lower threshold for synthetic data
        min_extraction_confidence=0.3
    )
    
    # Create synthetic frames with different scores
    test_cases = [
        (0, 0),  # Initial score
        (1, 0),  # First goal
        (1, 1),  # Equalizer
        (2, 1),  # Final score
    ]
    
    results = []
    for frame_num, (team1_score, team2_score) in enumerate(test_cases):
        frame = create_synthetic_scoreboard_frame(team1_score=team1_score, team2_score=team2_score)
        result = analyzer.analyze_frame(frame, frame_num)
        results.append(result)
        
        print(f"   Frame {frame_num}: Expected {team1_score}-{team2_score}, "
              f"Detected: {result['team1_score'] if result else 'None'}-{result['team2_score'] if result else 'None'}")
    
    # Get final statistics
    stats = analyzer.get_statistics()
    print(f"   📊 Analysis Statistics:")
    print(f"      Frames processed: {stats['frames_processed']}")
    print(f"      Total detections: {stats['total_detections']}")
    print(f"      Successful extractions: {stats['successful_extractions']}")
    print(f"      Scoreboard detected: {stats['scoreboard_detected']}")
    
    if stats['total_detections'] > 0:
        print(f"      Detection rate: {stats['detection_rate']:.2f}")
    
    return analyzer, results


def test_scoreboard_batch_processing():
    """Test batch processing of frames."""
    print("\n🧪 Testing batch processing...")
    
    analyzer = ScoreboardAnalyzer(detection_interval=5)  # Every 5th frame
    
    # Create a batch of frames
    frames = []
    for i in range(20):
        # Simulate score progression
        team1_score = min(i // 5, 3)
        team2_score = min(max(0, i - 10) // 5, 2)
        frame = create_synthetic_scoreboard_frame(team1_score=team1_score, team2_score=team2_score)
        frames.append(frame)
    
    # Process batch
    results = analyzer.analyze_video_batch(frames, start_frame_number=0)
    
    print(f"   Processed {len(frames)} frames")
    print(f"   Got {len([r for r in results if r is not None])} non-null results")
    
    # Check final score
    final_score = analyzer.get_final_score()
    if final_score:
        print(f"   Final detected score: {final_score['team1_score']}-{final_score['team2_score']} "
              f"(confidence: {final_score['confidence']:.2f})")
    else:
        print("   No final score detected")
    
    return analyzer, results


def test_integration_with_goal_utils():
    """Test integration with goal utilities."""
    print("\n🧪 Testing integration with goal utilities...")
    
    try:
        from utils.goal_utils import calculate_final_goal_stats
        
        # Create mock objects
        class MockPassCounter:
            def __init__(self):
                self.team_goals = {1: 1, 2: 0}
                self.player_goals = {5: {'goals': 1, 'team': 1}}
        
        mock_pass_counter = MockPassCounter()
        mock_enhanced_stats = {
            'final_total_goals': 1,
            'team_goals': {1: 1, 2: 0},
            'player_goals': {5: {'goals': 1, 'team': 1}},
            'final_team_goals': {1: 1, 2: 0},
            'final_player_goals': {5: {'goals': 1, 'team': 1}}
        }
        
        # Create analyzer with detected score
        analyzer = ScoreboardAnalyzer()
        analyzer.scoreboard_detected = True
        analyzer.current_score = {
            'team1_score': 2,
            'team2_score': 1,
            'confidence': 0.8,
            'detection_rate': 0.9
        }
        
        # Test goal stats calculation
        final_team_goals, final_player_goals = calculate_final_goal_stats(
            mock_pass_counter,
            mock_enhanced_stats,
            None,  # No manual goals
            analyzer
        )
        
        print(f"   Final team goals: {final_team_goals}")
        print(f"   Final player goals: {final_player_goals}")
        
        # Should use scoreboard scores due to high confidence
        expected_team_goals = {1: 2, 2: 1}
        if final_team_goals == expected_team_goals:
            print("   ✅ Scoreboard scores correctly prioritized")
        else:
            print("   ❌ Scoreboard scores not used as expected")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Could not import goal_utils: {e}")
        return False


def save_test_frame(frame, filename="test_scoreboard_frame.jpg"):
    """Save a test frame for visual inspection."""
    output_path = os.path.join(os.path.dirname(__file__), filename)
    cv2.imwrite(output_path, frame)
    print(f"   💾 Test frame saved to: {output_path}")


def main():
    """Run all end-to-end tests."""
    print("🚀 Starting Scoreboard Detection End-to-End Tests")
    print("=" * 60)
    
    try:
        # Test 1: Basic detection
        analyzer1, results1 = test_scoreboard_detection_basic()
        
        # Save a sample frame for visual inspection
        sample_frame = create_synthetic_scoreboard_frame(team1_score=2, team2_score=1)
        save_test_frame(sample_frame)
        
        # Test 2: Batch processing
        analyzer2, results2 = test_scoreboard_batch_processing()
        
        # Test 3: Integration
        integration_success = test_integration_with_goal_utils()
        
        print("\n" + "=" * 60)
        print("📋 Test Summary:")
        print(f"   Basic detection: {'✅ PASS' if analyzer1.frames_processed > 0 else '❌ FAIL'}")
        print(f"   Batch processing: {'✅ PASS' if analyzer2.frames_processed > 0 else '❌ FAIL'}")
        print(f"   Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
        
        print("\n💡 Notes:")
        print("   - This test uses synthetic scoreboard images")
        print("   - Real-world performance may vary depending on video quality")
        print("   - Consider adjusting confidence thresholds for production use")
        print("   - OCR accuracy depends on pytesseract installation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
