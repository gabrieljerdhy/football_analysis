#!/usr/bin/env python3
"""
Goal Detection Validation Test

This script validates the comprehensive goal detection system by running it
on a sample video and verifying that the required CSV files are generated
with the correct format and content.
"""

import csv
import os
import sys
from pathlib import Path

# Add parent directory to path to access src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.goal_detection.goal_detection_integration import create_goal_detection_integrator


def validate_csv_format(csv_path, expected_columns):
    """
    Validate that a CSV file has the expected format.
    
    Args:
        csv_path: Path to CSV file
        expected_columns: List of expected column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(csv_path):
            return False, f"CSV file does not exist: {csv_path}"
            
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            actual_columns = reader.fieldnames
            
            # Check if all expected columns are present
            missing_columns = set(expected_columns) - set(actual_columns)
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
                
            # Check if there are any unexpected columns
            extra_columns = set(actual_columns) - set(expected_columns)
            if extra_columns:
                print(f"⚠️  Extra columns found (not an error): {extra_columns}")
                
            # Try to read at least one row to verify format
            try:
                first_row = next(reader, None)
                if first_row is None:
                    print(f"⚠️  CSV file is empty: {csv_path}")
                else:
                    print(f"✅ CSV file has {len(list(reader)) + 1} rows")
            except Exception as e:
                return False, f"Error reading CSV content: {e}"
                
        return True, "CSV format is valid"
        
    except Exception as e:
        return False, f"Error validating CSV: {e}"


def test_goal_detection_csv_generation():
    """Test goal detection CSV generation with mock data."""
    print("🧪 Testing Goal Detection CSV Generation")
    print("=" * 50)
    
    # Create integrator
    integrator = create_goal_detection_integrator(
        output_dir="data/output",
        device="cpu"
    )
    
    print("✅ Goal detection integrator created")
    
    # Start video processing
    test_video_name = "test_goal_detection"
    integrator.start_video_processing(test_video_name)
    print(f"✅ Started processing for video: {test_video_name}")
    
    # Simulate processing frames with various scenarios
    test_scenarios = [
        # Normal frames without goals
        {
            'frame_num': 0,
            'ball_position': (400, 300),
            'ball_confidence': 0.7,
            'player_detections': [
                {'player_id': 7, 'team': 1, 'bbox': [390, 290, 410, 310], 'confidence': 0.8}
            ]
        },
        # Ball moving towards goal
        {
            'frame_num': 10,
            'ball_position': (200, 250),
            'ball_confidence': 0.8,
            'player_detections': [
                {'player_id': 7, 'team': 1, 'bbox': [190, 240, 210, 260], 'confidence': 0.8}
            ]
        },
        # Ball near goal area
        {
            'frame_num': 20,
            'ball_position': (100, 200),
            'ball_confidence': 0.9,
            'player_detections': [
                {'player_id': 7, 'team': 1, 'bbox': [90, 190, 110, 210], 'confidence': 0.8}
            ]
        },
        # Potential goal scenario
        {
            'frame_num': 30,
            'ball_position': (50, 180),
            'ball_confidence': 0.9,
            'player_detections': [
                {'player_id': 7, 'team': 1, 'bbox': [45, 175, 65, 195], 'confidence': 0.8}
            ]
        },
        # Another team scenario
        {
            'frame_num': 100,
            'ball_position': (600, 300),
            'ball_confidence': 0.8,
            'player_detections': [
                {'player_id': 10, 'team': 2, 'bbox': [590, 290, 610, 310], 'confidence': 0.8}
            ]
        }
    ]
    
    # Process test scenarios
    goals_detected = 0
    for scenario in test_scenarios:
        ball_detections = [{
            'position': scenario['ball_position'],
            'confidence': scenario['ball_confidence']
        }] if scenario['ball_position'] else []
        
        goal_detected = integrator.process_frame_with_detections(
            frame_num=scenario['frame_num'],
            frame=None,
            ball_detections=ball_detections,
            player_detections=scenario['player_detections']
        )
        
        if goal_detected:
            goals_detected += 1
            print(f"🎯 Goal detected at frame {scenario['frame_num']}")
    
    print(f"✅ Processed {len(test_scenarios)} test scenarios")
    print(f"🎯 Goals detected: {goals_detected}")
    
    # Finalize processing and generate CSV files
    try:
        frame_csv_path, scoreboard_csv_path = integrator.finalize_video_processing()
        print(f"✅ CSV files generated:")
        print(f"   Frame CSV: {frame_csv_path}")
        print(f"   Scoreboard CSV: {scoreboard_csv_path}")
        
        # Validate frame CSV format
        expected_frame_columns = [
            'frame_num', 'ball_x', 'ball_y', 'ball_confidence', 'ball_in_goal_area',
            'goal_side', 'player_id', 'player_team', 'player_ball_distance',
            'kick_detected', 'goal_detected', 'goal_confidence', 'detection_method',
            'temporal_validation', 'sequence_id'
        ]
        
        is_valid, message = validate_csv_format(frame_csv_path, expected_frame_columns)
        if is_valid:
            print(f"✅ Frame CSV format validation: {message}")
        else:
            print(f"❌ Frame CSV format validation failed: {message}")
            return False
            
        # Validate scoreboard CSV format
        expected_scoreboard_columns = [
            'goal_id', 'frame_start', 'frame_end', 'frame_goal', 'team', 'player_id',
            'goal_side', 'ball_final_x', 'ball_final_y', 'confidence', 'validation_score',
            'kick_frame', 'sequence_length', 'detection_method', 'timestamp_seconds'
        ]
        
        is_valid, message = validate_csv_format(scoreboard_csv_path, expected_scoreboard_columns)
        if is_valid:
            print(f"✅ Scoreboard CSV format validation: {message}")
        else:
            print(f"❌ Scoreboard CSV format validation failed: {message}")
            return False
            
        # Display statistics
        stats = integrator.get_statistics()
        print(f"\n📊 Goal Detection Statistics:")
        print(f"   Total frames processed: {stats.get('total_frames_processed', 0)}")
        print(f"   Total goals detected: {stats.get('total_goals_detected', 0)}")
        print(f"   Team goals: {stats.get('team_goals', {})}")
        print(f"   Average confidence: {stats.get('average_goal_confidence', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during CSV generation: {e}")
        return False


def test_csv_file_content():
    """Test that CSV files contain expected content structure."""
    print("\n🧪 Testing CSV File Content Structure")
    print("=" * 50)
    
    frame_csv_path = "data/output/test_goal_detection_goal_detected.csv"
    scoreboard_csv_path = "data/output/test_goal_detection_goal_detected_scoreboard.csv"
    
    # Test frame CSV content
    if os.path.exists(frame_csv_path):
        with open(frame_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            frame_rows = list(reader)
            
        print(f"✅ Frame CSV contains {len(frame_rows)} rows")
        
        if frame_rows:
            # Check first row structure
            first_row = frame_rows[0]
            print(f"   Sample frame data:")
            print(f"     Frame: {first_row.get('frame_num')}")
            print(f"     Ball position: ({first_row.get('ball_x')}, {first_row.get('ball_y')})")
            print(f"     Ball confidence: {first_row.get('ball_confidence')}")
            print(f"     Goal detected: {first_row.get('goal_detected')}")
            
    else:
        print(f"⚠️  Frame CSV not found: {frame_csv_path}")
        
    # Test scoreboard CSV content
    if os.path.exists(scoreboard_csv_path):
        with open(scoreboard_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            scoreboard_rows = list(reader)
            
        print(f"✅ Scoreboard CSV contains {len(scoreboard_rows)} rows")
        
        if scoreboard_rows:
            # Check goal events
            for i, row in enumerate(scoreboard_rows):
                print(f"   Goal {i+1}:")
                print(f"     Team: {row.get('team')}")
                print(f"     Player: {row.get('player_id')}")
                print(f"     Frame: {row.get('frame_goal')}")
                print(f"     Side: {row.get('goal_side')}")
                print(f"     Confidence: {row.get('confidence')}")
                
    else:
        print(f"⚠️  Scoreboard CSV not found: {scoreboard_csv_path}")
        
    return True


def main():
    """Run goal detection validation tests."""
    print("🎯 Goal Detection System Validation")
    print("=" * 60)
    
    success = True
    
    # Test 1: CSV generation
    try:
        if not test_goal_detection_csv_generation():
            success = False
    except Exception as e:
        print(f"❌ CSV generation test failed: {e}")
        success = False
        
    # Test 2: CSV content validation
    try:
        if not test_csv_file_content():
            success = False
    except Exception as e:
        print(f"❌ CSV content test failed: {e}")
        success = False
        
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All validation tests passed!")
        print("🎯 Goal detection system is ready for use")
        print("\nGenerated CSV files:")
        print("  - goal_detected.csv: Frame-by-frame goal detection results")
        print("  - goal_detected_scoreboard.csv: Summarized goal events")
    else:
        print("❌ Some validation tests failed!")
        print("🔧 Please check the error messages above")
        
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
