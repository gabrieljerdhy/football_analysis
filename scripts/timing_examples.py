#!/usr/bin/env python3
"""
Examples of how to use the timing utilities for measuring football analysis performance.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.timing_util import TimingCollector, time_phase, timed_function


def example_basic_timing():
    """Example 1: Basic timing with TimingCollector."""
    print("="*60)
    print("📊 EXAMPLE 1: Basic Timing")
    print("="*60)
    
    collector = TimingCollector()
    
    # Start overall job
    collector.start_job("Example Analysis Job")
    
    # Time different phases
    collector.start_phase("Data Loading")
    import time
    time.sleep(1)  # Simulate work
    collector.end_phase("Data Loading")
    
    collector.start_phase("Processing")
    time.sleep(2)  # Simulate work
    collector.end_phase("Processing")
    
    collector.start_phase("Output Generation")
    time.sleep(0.5)  # Simulate work
    collector.end_phase("Output Generation")
    
    # End job and show results
    collector.end_job()
    collector.print_summary()


def example_context_manager():
    """Example 2: Using context manager for timing."""
    print("\n" + "="*60)
    print("📊 EXAMPLE 2: Context Manager Timing")
    print("="*60)
    
    collector = TimingCollector()
    collector.start_job("Context Manager Example")
    
    # Use context manager for automatic timing
    with time_phase("Phase 1: Setup", collector):
        import time
        time.sleep(0.8)
        print("  🔧 Setting up components...")
    
    with time_phase("Phase 2: Main Work", collector):
        time.sleep(1.5)
        print("  ⚙️  Doing main processing...")
    
    with time_phase("Phase 3: Cleanup", collector):
        time.sleep(0.3)
        print("  🧹 Cleaning up...")
    
    collector.end_job()
    collector.print_summary()


@timed_function("Function Processing")
def example_function_with_timing():
    """Example 3: Function with automatic timing decorator."""
    import time
    print("  🎯 Running function with automatic timing...")
    time.sleep(1.2)
    return "Function completed successfully"


def example_decorator_timing():
    """Example 3: Using function decorator for timing."""
    print("\n" + "="*60)
    print("📊 EXAMPLE 3: Function Decorator Timing")
    print("="*60)
    
    collector = TimingCollector()
    collector.start_job("Decorator Example")
    
    # Call function with automatic timing
    result = example_function_with_timing()
    print(f"  ✅ Result: {result}")
    
    collector.end_job()
    collector.print_summary()


def example_real_world_usage():
    """Example 4: Real-world usage pattern for football analysis."""
    print("\n" + "="*60)
    print("📊 EXAMPLE 4: Real-World Usage Pattern")
    print("="*60)
    
    collector = TimingCollector()
    collector.start_job("Football Analysis Simulation")
    
    # Simulate the main phases of football analysis
    phases = [
        ("Video Loading", 0.5),
        ("Object Detection", 2.0),
        ("Player Tracking", 1.8),
        ("Team Assignment", 0.7),
        ("Pass Detection", 1.2),
        ("Goal Detection", 0.4),
        ("Statistics Generation", 0.3),
        ("Video Output", 1.0),
    ]
    
    import time
    for phase_name, duration in phases:
        with time_phase(phase_name, collector):
            print(f"  🔄 {phase_name}...")
            time.sleep(duration)
    
    collector.end_job()
    collector.print_summary()
    
    # Save results
    collector.save_to_file("example_timing_results")


def show_usage_instructions():
    """Show instructions for using the timing utilities."""
    print("\n" + "="*80)
    print("📖 HOW TO USE TIMING UTILITIES IN YOUR CODE")
    print("="*80)
    
    print("""
🎯 METHOD 1: Basic TimingCollector
    from utils.timing_util import TimingCollector
    
    collector = TimingCollector()
    collector.start_job("My Analysis Job")
    
    collector.start_phase("Phase 1")
    # ... your code here ...
    collector.end_phase("Phase 1")
    
    collector.end_job()
    collector.print_summary()

🎯 METHOD 2: Context Manager
    from utils.timing_util import time_phase
    
    with time_phase("My Phase"):
        # ... your code here ...

🎯 METHOD 3: Function Decorator
    from utils.timing_util import timed_function
    
    @timed_function("My Function")
    def my_function():
        # ... your code here ...

🎯 METHOD 4: Ready-to-use Scripts
    # Simple benchmark (measures overall time)
    python benchmark_analysis.py --input video.mp4
    
    # Detailed timing (measures phases)
    python timed_analysis.py --input video.mp4
    
    # Memory-efficient with timing
    python timed_analysis.py --input video.mp4 --memory-efficient

📊 OUTPUT FILES:
    - JSON: Complete timing data with metadata
    - CSV: Phase breakdown for spreadsheet analysis
    - Console: Real-time progress and final summary
    """)


def main():
    """Run all timing examples."""
    print("🚀 TIMING UTILITIES EXAMPLES")
    print("This script demonstrates different ways to measure execution time.")
    
    # Run examples
    example_basic_timing()
    example_context_manager()
    example_decorator_timing()
    example_real_world_usage()
    
    # Show usage instructions
    show_usage_instructions()
    
    print(f"\n✅ All examples completed!")
    print(f"📁 Check the 'example_timing_results.json' and '.csv' files for saved results.")


if __name__ == "__main__":
    main()
