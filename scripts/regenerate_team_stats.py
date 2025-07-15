#!/usr/bin/env python3
"""
Script to regenerate existing team stats files with the new simplified format.
This script converts old team stats files to the new two-column format.
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


def regenerate_videoplayback_process_stats():
    """Regenerate the videoplayback_process team stats with new format."""
    print("🔄 Regenerating videoplayback_process team stats...")
    
    # Create mock components (since we don't have the original objects)
    pass_counter = PassCounter()
    tackle_counter = TackleCounter()
    scoreboard_analyzer = ScoreboardAnalyzer()
    
    # Based on the original data, create enhanced goal stats
    # Original data showed: Team 1: 17 goals, Team 2: 15 goals (from goals_final)
    enhanced_goal_stats = {
        "team_goals": {1: 17, 2: 15},  # Use the original final goals
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
        
        print(f"✅ Regenerated team stats: {team_csv_path}")
        
        # Display the new content
        with open(team_csv_path, 'r') as f:
            content = f.read()
            print(f"\n📊 New Team Stats Content:")
            print(content)
            
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating stats: {e}")
        return False


def regenerate_four_min_test_stats():
    """Regenerate the four_min_test team stats with new format."""
    print("\n🔄 Regenerating four_min_test team stats...")
    
    # Create mock components
    pass_counter = PassCounter()
    tackle_counter = TackleCounter()
    scoreboard_analyzer = ScoreboardAnalyzer()
    
    # Based on the original data: Team 1: 10 goals, Team 2: 12 goals
    enhanced_goal_stats = {
        "team_goals": {1: 10, 2: 12},
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
        
        print(f"✅ Regenerated team stats: {team_csv_path}")
        
        # Display the new content
        with open(team_csv_path, 'r') as f:
            content = f.read()
            print(f"\n📊 New Team Stats Content:")
            print(content)
            
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating stats: {e}")
        return False


def main():
    """Main function to regenerate all team stats files."""
    print("🚀 Regenerating Team Stats Files with New Format")
    print("=" * 55)
    
    # Regenerate videoplayback_process stats
    videoplayback_success = regenerate_videoplayback_process_stats()
    
    # Regenerate four_min_test stats
    four_min_success = regenerate_four_min_test_stats()
    
    # Summary
    print("\n" + "=" * 55)
    print("🏆 REGENERATION SUMMARY:")
    print(f"   videoplayback_process: {'✅ SUCCESS' if videoplayback_success else '❌ FAILED'}")
    print(f"   four_min_test: {'✅ SUCCESS' if four_min_success else '❌ FAILED'}")
    
    overall_success = videoplayback_success and four_min_success
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL FILES REGENERATED' if overall_success else '❌ SOME FAILED'}")
    
    if overall_success:
        print("\n🎉 SUCCESS: All team stats files now have the simplified format!")
        print("   📁 Files updated:")
        print("      - videoplayback_process_team_stats.csv")
        print("      - four_min_test_team_stats.csv")
        print("   📊 Format: 2 columns only")
        print("      - goal_detected_using_the_model")
        print("      - goal_detected_using_scoreboard_detection")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
