#!/usr/bin/env python3
"""
Simple script to check processing progress and output files.
"""

import os
import time
from pathlib import Path
from datetime import datetime

def check_output_files():
    """Check what output files exist."""
    print("📁 CHECKING OUTPUT FILES")
    print("=" * 50)
    
    # Check output videos
    output_videos_dir = Path("output_videos")
    if output_videos_dir.exists():
        print(f"\n🎬 Output Videos ({output_videos_dir}):")
        video_files = list(output_videos_dir.glob("*.avi"))
        if video_files:
            for video in sorted(video_files):
                size_mb = video.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(video.stat().st_mtime)
                print(f"   ✅ {video.name} ({size_mb:.1f} MB, {mod_time.strftime('%H:%M:%S')})")
        else:
            print("   ❌ No .avi files found")
    
    # Check CSV files
    output_dir = Path("output")
    if output_dir.exists():
        print(f"\n📊 CSV Statistics ({output_dir}):")
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            for csv_file in sorted(csv_files):
                size_kb = csv_file.stat().st_size / 1024
                mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
                print(f"   ✅ {csv_file.name} ({size_kb:.1f} KB, {mod_time.strftime('%H:%M:%S')})")
        else:
            print("   ❌ No .csv files found")
    
    # Check for your specific large video files
    large_video_files = [
        "output_videos/FULL_MATCHPortugalvSpain_output.avi",
        "output/FULL_MATCHPortugalvSpain_team_stats.csv", 
        "output/FULL_MATCHPortugalvSpain_player_stats.csv"
    ]
    
    print(f"\n🎯 Large Video Processing Status:")
    all_exist = True
    for file_path in large_video_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            if file_path.endswith('.avi'):
                size_str = f"{size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"
            print(f"   ✅ {Path(file_path).name} ({size_str})")
        else:
            print(f"   ⏳ {Path(file_path).name} (not ready yet)")
            all_exist = False
    
    if all_exist:
        print(f"\n🎉 ALL FILES READY! Your large video processing is complete!")
        return True
    else:
        print(f"\n⏳ Processing still in progress...")
        return False

def check_memory_usage():
    """Check current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)
        print(f"\n💾 Current memory usage: {memory_gb:.2f} GB")
    except ImportError:
        print(f"\n💾 Install psutil to see memory usage: pip install psutil")

def main():
    print(f"🔍 FOOTBALL ANALYSIS PROGRESS CHECKER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check if processing is complete
    is_complete = check_output_files()
    
    # Check memory usage
    check_memory_usage()
    
    # Show expected file locations
    print(f"\n📍 Expected Output Locations:")
    print(f"   🎬 Video: output_videos/FULL_MATCHPortugalvSpain_output.avi")
    print(f"   📊 Team Stats: output/FULL_MATCHPortugalvSpain_team_stats.csv")
    print(f"   📊 Player Stats: output/FULL_MATCHPortugalvSpain_player_stats.csv")
    
    if not is_complete:
        print(f"\n💡 Tips:")
        print(f"   • Processing is still running - be patient!")
        print(f"   • Large video (179,649 frames) takes time")
        print(f"   • Run this script again to check progress")
        print(f"   • Files will appear when processing completes")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    main()
