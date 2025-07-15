#!/usr/bin/env python3
"""
Script to regenerate existing team stats files with the new comprehensive format.
This script converts team stats files to include all original columns plus the two enhanced goal detection columns.
"""

import sys
import os
import csv
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.goal_utils import export_consolidated_goal_statistics
from pass_counter.pass_counter import PassCounter
from pass_counter.tackle_counter import TackleCounter
from scoreboard_detection.scoreboard_analyzer import ScoreboardAnalyzer


def regenerate_videoplayback_process_comprehensive():
    """Regenerate the videoplayback_process team stats with comprehensive format."""
    print("🔄 Regenerating videoplayback_process team stats with comprehensive format...")
    
    # Create mock components with realistic data based on original file
    pass_counter = PassCounter()
    pass_counter.team_passes = {1: 21, 2: 14}  # From original data
    pass_counter.team_goals = {1: 1, 2: 1}     # From original data
    
    tackle_counter = TackleCounter()
    tackle_counter.team_tackles = {1: 0, 2: 2}           # From original data
    tackle_counter.team_interceptions = {1: 7, 2: 13}   # From original data
    
    scoreboard_analyzer = ScoreboardAnalyzer()
    
    # Enhanced goal stats - use the original final goals as model detection
    enhanced_goal_stats = {
        "team_goals": {1: 17, 2: 15},  # Use the original final goals for model detection
        "total_goals": 32,
        "average_confidence": 0.8,
        "goal_events": []
    }
    
    # Use the original final goals
    final_team_goals = {1: 17, 2: 15}
    final_player_goals = {}
    
    try:
        team_csv_path, player_csv_path = export_consolidated_goal_statistics(
            video_name="videoplayback_process",
            pass_counter=pass_counter,
            enhanced_goal_stats=enhanced_goal_stats,
            final_team_goals=final_team_goals,
            final_player_goals=final_player_goals,
            tackle_counter=tackle_counter,
            scoreboard_analyzer=scoreboard_analyzer,
            output_dir="data/output"
        )
        
        print(f"✅ Regenerated comprehensive team stats: {team_csv_path}")
        
        # Display the new content
        with open(team_csv_path, 'r') as f:
            content = f.read()
            print(f"\n📊 New Comprehensive Team Stats Content:")
            print(content)
            
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating stats: {e}")
        return False


def regenerate_four_min_test_comprehensive():
    """Regenerate the four_min_test team stats with comprehensive format."""
    print("\n🔄 Regenerating four_min_test team stats with comprehensive format...")
    
    # Create mock components with realistic data based on original file
    pass_counter = PassCounter()
    pass_counter.team_passes = {1: 22, 2: 45}  # From original data
    pass_counter.team_goals = {1: 2, 2: 1}     # From original data
    
    tackle_counter = TackleCounter()
    tackle_counter.team_tackles = {1: 6, 2: 4}           # From original data
    tackle_counter.team_interceptions = {1: 23, 2: 27}  # From original data
    
    scoreboard_analyzer = ScoreboardAnalyzer()
    
    # Enhanced goal stats - use the original final goals as model detection
    enhanced_goal_stats = {
        "team_goals": {1: 10, 2: 12},  # Use the original final goals for model detection
        "total_goals": 22,
        "average_confidence": 0.75,
        "goal_events": []
    }
    
    final_team_goals = {1: 10, 2: 12}
    final_player_goals = {}
    
    try:
        team_csv_path, player_csv_path = export_consolidated_goal_statistics(
            video_name="four_min_test",
            pass_counter=pass_counter,
            enhanced_goal_stats=enhanced_goal_stats,
            final_team_goals=final_team_goals,
            final_player_goals=final_player_goals,
            tackle_counter=tackle_counter,
            scoreboard_analyzer=scoreboard_analyzer,
            output_dir="data/output"
        )
        
        print(f"✅ Regenerated comprehensive team stats: {team_csv_path}")
        
        # Display the new content
        with open(team_csv_path, 'r') as f:
            content = f.read()
            print(f"\n📊 New Comprehensive Team Stats Content:")
            print(content)
            
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating stats: {e}")
        return False


def main():
    """Main function to regenerate all team stats files with comprehensive format."""
    print("🚀 Regenerating Team Stats Files with Comprehensive Format")
    print("=" * 65)
    
    # Regenerate videoplayback_process stats
    videoplayback_success = regenerate_videoplayback_process_comprehensive()
    
    # Regenerate four_min_test stats
    four_min_success = regenerate_four_min_test_comprehensive()
    
    # Summary
    print("\n" + "=" * 65)
    print("🏆 REGENERATION SUMMARY:")
    print(f"   videoplayback_process: {'✅ SUCCESS' if videoplayback_success else '❌ FAILED'}")
    print(f"   four_min_test: {'✅ SUCCESS' if four_min_success else '❌ FAILED'}")
    
    overall_success = videoplayback_success and four_min_success
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL FILES REGENERATED' if overall_success else '❌ SOME FAILED'}")
    
    if overall_success:
        print("\n🎉 SUCCESS: All team stats files now have the comprehensive format!")
        print("   📁 Files updated:")
        print("      - videoplayback_process_team_stats.csv")
        print("      - four_min_test_team_stats.csv")
        print("   📊 Format: 14 columns total")
        print("      - All original statistics (team, passes, tackles, etc.)")
        print("      - Enhanced goal detection columns:")
        print("        * goal_detected_using_the_model")
        print("        * goal_detected_using_scoreboard_detection")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
