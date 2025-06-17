#!/usr/bin/env python3
"""
Simple benchmark script for measuring football analysis execution time.
This is a lightweight wrapper that focuses on overall timing.
"""

import time
import argparse
import subprocess
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import os


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_memory_usage():
    """Get current memory usage in GB."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 ** 3)
    except:
        return 0.0


def get_system_info():
    """Get system information for benchmarking context."""
    try:
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
            'python_version': sys.version.split()[0],
            'platform': sys.platform
        }
    except:
        return {}


def run_analysis_with_timing(command_args, job_name="Football Analysis"):
    """Run the analysis command and measure execution time."""
    
    print("="*80)
    print(f"🚀 BENCHMARK: {job_name}")
    print("="*80)
    
    # System info
    system_info = get_system_info()
    print(f"💻 System: {system_info.get('cpu_count', 'Unknown')} CPUs, "
          f"{system_info.get('memory_total_gb', 0):.1f}GB RAM")
    print(f"🐍 Python: {system_info.get('python_version', 'Unknown')}")
    
    # Command info
    print(f"📝 Command: {' '.join(command_args)}")
    
    # Start timing
    start_time = time.time()
    start_memory = get_memory_usage()
    start_datetime = datetime.fromtimestamp(start_time)
    
    print(f"🕐 Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Initial memory: {start_memory:.2f} GB")
    print("-"*80)
    
    # Run the command
    try:
        result = subprocess.run(
            command_args,
            capture_output=False,  # Let output go to console
            text=True,
            cwd=os.getcwd()
        )
        
        # End timing
        end_time = time.time()
        end_memory = get_memory_usage()
        end_datetime = datetime.fromtimestamp(end_time)
        
        duration = end_time - start_time
        
        print("-"*80)
        print(f"🕐 End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Final memory: {end_memory:.2f} GB")
        print(f"⏰ Total duration: {format_duration(duration)}")
        
        if result.returncode == 0:
            print(f"✅ Analysis completed successfully!")
        else:
            print(f"❌ Analysis failed with return code: {result.returncode}")
            
        # Create benchmark results
        benchmark_results = {
            'job_name': job_name,
            'command': ' '.join(command_args),
            'start_time': start_datetime.isoformat(),
            'end_time': end_datetime.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'return_code': result.returncode,
            'success': result.returncode == 0,
            'memory_usage': {
                'start_gb': start_memory,
                'end_gb': end_memory,
                'delta_gb': end_memory - start_memory
            },
            'system_info': system_info
        }
        
        return benchmark_results
        
    except KeyboardInterrupt:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⚠️  Benchmark interrupted after {format_duration(duration)}")
        return {
            'job_name': job_name,
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'interrupted': True
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ Benchmark failed after {format_duration(duration)}: {e}")
        return {
            'job_name': job_name,
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'error': str(e)
        }


def save_benchmark_results(results, output_path=None):
    """Save benchmark results to a JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmark_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Benchmark results saved to: {output_path}")
    return output_path


def main():
    """Command line interface for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark Football Analysis - Simple execution time measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark basic analysis
  python benchmark_analysis.py --input input_videos/match.mp4
  
  # Benchmark memory-efficient mode
  python benchmark_analysis.py --input input_videos/match.mp4 --memory-efficient
  
  # Benchmark with custom script
  python benchmark_analysis.py --script complete_analysis.py --input input_videos/match.mp4
  
  # Benchmark with all features
  python benchmark_analysis.py --input input_videos/match.mp4 \\
    --enable-camera-movement --enable-speed-distance
        """
    )
    
    # Benchmark options
    parser.add_argument("--script", default="main.py", help="Script to benchmark (default: main.py)")
    parser.add_argument("--job-name", help="Custom name for this benchmark job")
    parser.add_argument("--output", help="Output file for benchmark results")
    parser.add_argument("--no-save", action="store_true", help="Don't save benchmark results")
    
    # Analysis arguments (pass-through to the analysis script)
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-video", help="Output video path")
    parser.add_argument("--no-stubs", action="store_true", help="Don't use stub files")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate all data")
    parser.add_argument("--goals-config", help="Goals configuration file")
    parser.add_argument("--enable-camera-movement", action="store_true", help="Enable camera movement estimation")
    parser.add_argument("--enable-speed-distance", action="store_true", help="Enable speed and distance calculation")
    parser.add_argument("--memory-efficient", action="store_true", help="Use memory-efficient processing")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for memory-efficient processing")
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    
    args = parser.parse_args()
    
    # Build command for the analysis script
    command = ["python", args.script, "--input", args.input]
    
    if args.output_video:
        command.extend(["--output", args.output_video])
    if args.no_stubs:
        command.append("--no-stubs")
    if args.force_regenerate:
        command.append("--force-regenerate")
    if args.goals_config:
        command.extend(["--goals-config", args.goals_config])
    if args.enable_camera_movement:
        command.append("--enable-camera-movement")
    if args.enable_speed_distance:
        command.append("--enable-speed-distance")
    if args.memory_efficient:
        command.append("--memory-efficient")
    if args.batch_size != 50:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.memory_limit != 8.0:
        command.extend(["--memory-limit", str(args.memory_limit)])
    
    # Determine job name
    if args.job_name:
        job_name = args.job_name
    else:
        video_name = Path(args.input).stem
        mode = "Memory-Efficient" if args.memory_efficient else "Standard"
        job_name = f"Football Analysis: {video_name} ({mode})"
    
    # Run benchmark
    try:
        results = run_analysis_with_timing(command, job_name)
        
        # Save results
        if not args.no_save:
            save_benchmark_results(results, args.output)
        
        # Print final summary
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        print(f"Job: {results.get('job_name', 'Unknown')}")
        print(f"Duration: {results.get('duration_formatted', 'Unknown')}")
        if 'success' in results:
            status = "✅ SUCCESS" if results['success'] else "❌ FAILED"
            print(f"Status: {status}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
