#!/usr/bin/env python3
"""
Timed wrapper for football analysis that measures end-to-end execution time.
This script wraps the main analysis functions with detailed timing measurements.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.timing_util import TimingCollector, timed_function, get_global_collector
from main import main as original_main


def timed_football_analysis(
    input_video_path,
    output_video_path=None,
    use_stubs=True,
    force_regenerate=False,
    goals_config=None,
    enable_camera_movement=False,
    enable_speed_distance=False,
    memory_efficient=False,
    batch_size=50,
    memory_limit=8.0,
    upload_to_spaces=False,
    spaces_access_key_id=None,
    spaces_secret_access_key=None,
    spaces_bucket=None,
    spaces_region="nyc3",
    spaces_folder_prefix="football_analysis",
    upload_csv_only=False,
    save_timing_results=True,
    timing_output_path=None
):
    """
    Timed version of the main football analysis function.
    
    Args:
        All the same arguments as the original main() function, plus:
        save_timing_results (bool): Whether to save timing results to files
        timing_output_path (str): Custom path for timing results (optional)
    """
    
    # Initialize timing collector
    collector = get_global_collector()
    
    # Determine job name based on input video
    video_name = Path(input_video_path).stem
    job_name = f"Football Analysis: {video_name}"
    
    # Start overall job timing
    collector.start_job(job_name)
    
    try:
        print(f"🎬 Starting timed analysis of: {input_video_path}")
        print(f"📊 Memory efficient mode: {'ON' if memory_efficient else 'OFF'}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔄 Using stubs: {'YES' if use_stubs else 'NO'}")
        print(f"🚀 Force regenerate: {'YES' if force_regenerate else 'NO'}")
        
        # Phase 1: Setup and initialization
        collector.start_phase("Setup and Initialization")
        print(f"🔧 Initializing analysis components...")
        collector.end_phase("Setup and Initialization")
        
        # Phase 2: Main analysis (this will include all the heavy processing)
        collector.start_phase("Main Analysis Pipeline")
        
        # Call the original main function
        result = original_main(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            use_stubs=use_stubs,
            force_regenerate=force_regenerate,
            goals_config=goals_config,
            enable_camera_movement=enable_camera_movement,
            enable_speed_distance=enable_speed_distance,
            memory_efficient=memory_efficient,
            batch_size=batch_size,
            memory_limit=memory_limit,
            upload_to_spaces=upload_to_spaces,
            spaces_access_key_id=spaces_access_key_id,
            spaces_secret_access_key=spaces_secret_access_key,
            spaces_bucket=spaces_bucket,
            spaces_region=spaces_region,
            spaces_folder_prefix=spaces_folder_prefix,
            upload_csv_only=upload_csv_only,
        )
        
        collector.end_phase("Main Analysis Pipeline")
        
        # Phase 3: Finalization
        collector.start_phase("Results Finalization")
        print(f"📊 Finalizing results and cleanup...")
        collector.end_phase("Results Finalization")
        
        # End overall job timing
        collector.end_job()
        
        # Print timing summary
        collector.print_summary()
        
        # Save timing results if requested
        if save_timing_results:
            if timing_output_path is None:
                # Create timing results in output directory
                output_dir = Path("timing_results")
                output_dir.mkdir(exist_ok=True)
                timing_output_path = output_dir / f"{video_name}_timing"
            
            collector.save_to_file(str(timing_output_path))
            
        return result
        
    except Exception as e:
        # End job timing even if there's an error
        collector.end_job()
        collector.print_summary()
        
        print(f"❌ Analysis failed with error: {e}")
        
        # Still save timing results for debugging
        if save_timing_results:
            if timing_output_path is None:
                output_dir = Path("timing_results")
                output_dir.mkdir(exist_ok=True)
                timing_output_path = output_dir / f"{video_name}_timing_FAILED"
            collector.save_to_file(str(timing_output_path))
            
        raise


def main():
    """Command line interface for timed analysis."""
    parser = argparse.ArgumentParser(
        description="Timed Football Analysis - Measure end-to-end execution time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic timed analysis
  python timed_analysis.py --input input_videos/match.mp4
  
  # Memory-efficient mode with timing
  python timed_analysis.py --input input_videos/large_match.mp4 --memory-efficient
  
  # Custom timing output location
  python timed_analysis.py --input input_videos/match.mp4 --timing-output my_timing_results
  
  # Full analysis with all features timed
  python timed_analysis.py --input input_videos/match.mp4 \\
    --enable-camera-movement --enable-speed-distance --memory-efficient
        """
    )
    
    # Video processing arguments (same as main.py)
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--no-stubs", action="store_true", help="Don't use stub files")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate all data")
    parser.add_argument("--goals-config", help="Goals configuration file")
    parser.add_argument("--enable-camera-movement", action="store_true", help="Enable camera movement estimation")
    parser.add_argument("--enable-speed-distance", action="store_true", help="Enable speed and distance calculation")
    parser.add_argument("--memory-efficient", action="store_true", help="Use memory-efficient processing")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for memory-efficient processing")
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    
    # Digital Ocean Spaces arguments
    parser.add_argument("--upload-to-spaces", action="store_true", help="Upload results to DigitalOcean Spaces")
    parser.add_argument("--spaces-access-key-id", help="DigitalOcean Spaces access key ID")
    parser.add_argument("--spaces-secret-access-key", help="DigitalOcean Spaces secret access key")
    parser.add_argument("--spaces-bucket", help="DigitalOcean Spaces bucket name")
    parser.add_argument("--spaces-region", default="nyc3", help="DigitalOcean Spaces region")
    parser.add_argument("--spaces-folder-prefix", default="football_analysis", help="Folder prefix in Spaces")
    parser.add_argument("--upload-csv-only", action="store_true", help="Upload only CSV files to Spaces")
    
    # Timing-specific arguments
    parser.add_argument("--no-timing-save", action="store_true", help="Don't save timing results to files")
    parser.add_argument("--timing-output", help="Custom path for timing results (without extension)")
    
    args = parser.parse_args()
    
    # Run timed analysis
    try:
        timed_football_analysis(
            input_video_path=args.input,
            output_video_path=args.output,
            use_stubs=not args.no_stubs,
            force_regenerate=args.force_regenerate,
            goals_config=args.goals_config,
            enable_camera_movement=args.enable_camera_movement,
            enable_speed_distance=args.enable_speed_distance,
            memory_efficient=args.memory_efficient,
            batch_size=args.batch_size,
            memory_limit=args.memory_limit,
            upload_to_spaces=args.upload_to_spaces,
            spaces_access_key_id=args.spaces_access_key_id,
            spaces_secret_access_key=args.spaces_secret_access_key,
            spaces_bucket=args.spaces_bucket,
            spaces_region=args.spaces_region,
            spaces_folder_prefix=args.spaces_folder_prefix,
            upload_csv_only=args.upload_csv_only,
            save_timing_results=not args.no_timing_save,
            timing_output_path=args.timing_output
        )
        
        print(f"\n✅ Timed analysis completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
