#!/usr/bin/env python3
"""
Simple utility to measure total execution time of any command.
Usage: python time_job.py [command and arguments]
"""

import sys
import time
import subprocess
from datetime import datetime


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minutes {secs:.1f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours {minutes} minutes {secs:.1f} seconds"


def main():
    if len(sys.argv) < 2:
        print("Usage: python time_job.py [command and arguments]")
        print("\nExamples:")
        print("  python time_job.py python main.py --input video.mp4")
        print("  python time_job.py python main.py --input video.mp4 --memory-efficient")
        print("  python time_job.py python main.py --input input_videos/full_match_manchester.mp4 --memory-efficient --batch-size 30 --upload-to-spaces --force-regenerate")
        sys.exit(1)
    
    # Get the command to run
    command = sys.argv[1:]
    
    print("=" * 80)
    print("⏱️  TIMING JOB EXECUTION")
    print("=" * 80)
    print(f"Command: {' '.join(command)}")
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.fromtimestamp(start_time)
    print(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    try:
        # Run the command
        result = subprocess.run(command, cwd=".")
        
        # Record end time
        end_time = time.time()
        end_datetime = datetime.fromtimestamp(end_time)
        duration = end_time - start_time
        
        print()
        print("=" * 80)
        print("⏱️  JOB COMPLETED")
        print("=" * 80)
        print(f"Started at:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time:  {format_duration(duration)}")
        
        if result.returncode == 0:
            print("Status:      ✅ SUCCESS")
        else:
            print(f"Status:      ❌ FAILED (exit code: {result.returncode})")
        
        print("=" * 80)
        
        # Save timing to a simple log file
        log_entry = f"{datetime.now().isoformat()},{' '.join(command)},{duration:.2f},{format_duration(duration)},{result.returncode == 0}\n"
        with open("job_timing_log.csv", "a") as f:
            f.write(log_entry)
        
        print(f"📝 Timing logged to: job_timing_log.csv")
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⚠️  Job interrupted after {format_duration(duration)}")
        sys.exit(1)
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ Error after {format_duration(duration)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
