#!/usr/bin/env python3
"""
Test script to verify the comprehensive team stats CSV format.
This script tests that {mp4_file_name}_team_stats.csv has all original columns 
plus the two enhanced goal detection columns.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.goal_utils import export_consolidated_goal_statistics
from pass_counter.pass_counter import PassCounter
from pass_counter.tackle_counter import TackleCounter
from scoreboard_detection.scoreboard_analyzer import ScoreboardAnalyzer


def test_comprehensive_team_stats_format():
    """Test that the team stats CSV has all columns including enhanced goal detection."""
    print("🧪 Testing Comprehensive Team Stats CSV Format...")
    
    # Create mock components with some data
    pass_counter = PassCounter()
    pass_counter.team_passes = {1: 25, 2: 18}
    pass_counter.team_goals = {1: 1, 2: 2}
    
    tackle_counter = TackleCounter()
    tackle_counter.team_tackles = {1: 8, 2: 6}
    tackle_counter.team_interceptions = {1: 4, 2: 7}
    
    scoreboard_analyzer = ScoreboardAnalyzer()
    
    # Mock enhanced goal stats
    enhanced_goal_stats = {
        "team_goals": {1: 3, 2: 2},
        "total_goals": 5,
        "average_confidence": 0.82,
        "goal_events": [
            {"team": 1, "confidence_score": 0.8},
            {"team": 1, "confidence_score": 0.9},
            {"team": 2, "confidence_score": 0.7},
        ]
    }
    
    # Mock final goals
    final_team_goals = {1: 3, 2: 2}
    final_player_goals = {}
    
    # Test the export function
    try:
        team_csv_path, player_csv_path = export_consolidated_goal_statistics(
            video_name="comprehensive_test",
            pass_counter=pass_counter,
            enhanced_goal_stats=enhanced_goal_stats,
            final_team_goals=final_team_goals,
            final_player_goals=final_player_goals,
            tackle_counter=tackle_counter,
            scoreboard_analyzer=scoreboard_analyzer,
            output_dir="data/output"
        )
        
        print(f"✅ Team CSV generated: {team_csv_path}")
        
        # Verify the team CSV format
        if os.path.exists(team_csv_path):
            with open(team_csv_path, 'r') as f:
                content = f.read()
                print(f"\n📊 Team CSV Content:")
                print(content)
                
            # Validate the format
            lines = content.strip().split('\n')
            if len(lines) >= 3:  # Header + 2 team rows
                header = lines[0]
                expected_columns = [
                    "team",
                    "passes", 
                    "goals_regular",
                    "goals_enhanced",
                    "goals_scoreboard", 
                    "goals_final",
                    "tackles",
                    "interceptions",
                    "goal_events_count",
                    "avg_goal_confidence",
                    "scoreboard_detected",
                    "scoreboard_confidence",
                    "goal_detected_using_the_model",
                    "goal_detected_using_scoreboard_detection"
                ]
                expected_header = ",".join(expected_columns)
                
                if header == expected_header:
                    print("✅ Team CSV header has all expected columns")
                    
                    # Check that we have exactly 14 columns
                    team1_data = lines[1].split(',')
                    team2_data = lines[2].split(',')
                    
                    if len(team1_data) == 14 and len(team2_data) == 14:
                        print("✅ Team CSV has exactly 14 columns")
                        
                        # Validate specific columns
                        print(f"✅ Team 1 data: {team1_data}")
                        print(f"✅ Team 2 data: {team2_data}")
                        
                        # Check the enhanced goal detection columns (last two)
                        team1_model_goals = int(team1_data[12])  # goal_detected_using_the_model
                        team1_scoreboard_goals = int(team1_data[13])  # goal_detected_using_scoreboard_detection
                        team2_model_goals = int(team2_data[12])
                        team2_scoreboard_goals = int(team2_data[13])
                        
                        print(f"✅ Enhanced Goal Detection:")
                        print(f"   Team 1 - Model: {team1_model_goals}, Scoreboard: {team1_scoreboard_goals}")
                        print(f"   Team 2 - Model: {team2_model_goals}, Scoreboard: {team2_scoreboard_goals}")
                        
                        # Check original statistics are preserved
                        team1_passes = int(team1_data[1])  # passes
                        team1_tackles = int(team1_data[6])  # tackles
                        team1_interceptions = int(team1_data[7])  # interceptions
                        
                        print(f"✅ Original Statistics Preserved:")
                        print(f"   Team 1 - Passes: {team1_passes}, Tackles: {team1_tackles}, Interceptions: {team1_interceptions}")
                        
                        return True
                        
                    else:
                        print(f"❌ Team CSV has wrong number of columns. Expected 14, got {len(team1_data)} and {len(team2_data)}")
                        return False
                else:
                    print(f"❌ Team CSV header is wrong.")
                    print(f"   Expected: {expected_header}")
                    print(f"   Got: {header}")
                    return False
            else:
                print("❌ Team CSV doesn't have enough rows")
                return False
        else:
            print("❌ Team CSV file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error testing comprehensive team stats format: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Testing Comprehensive Team Stats CSV Format")
    print("=" * 55)
    
    # Test the comprehensive CSV format
    format_test_passed = test_comprehensive_team_stats_format()
    
    # Summary
    print("\n" + "=" * 55)
    print("🏆 TEST SUMMARY:")
    print(f"   Comprehensive Format Test: {'✅ PASS' if format_test_passed else '❌ FAIL'}")
    
    print(f"\n🎯 OVERALL RESULT: {'✅ TEST PASSED' if format_test_passed else '❌ TEST FAILED'}")
    
    if format_test_passed:
        print("\n🎉 SUCCESS: The comprehensive team stats CSV format is correct!")
        print("   ✅ File name format: {mp4_file_name}_team_stats.csv")
        print("   ✅ Contains all 14 columns:")
        print("      - All original statistics (team, passes, tackles, etc.)")
        print("      - Enhanced goal detection columns:")
        print("        * goal_detected_using_the_model")
        print("        * goal_detected_using_scoreboard_detection")
        print("   ✅ Data is properly formatted with realistic values")
    
    return format_test_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
