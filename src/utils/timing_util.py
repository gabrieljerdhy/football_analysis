#!/usr/bin/env python3
"""
Timing utility for measuring end-to-end execution time of football analysis jobs.
Provides detailed timing breakdowns and performance metrics.
"""

import csv
import functools
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


class TimingCollector:
    """Collects and manages timing data for different phases of the analysis."""

    def __init__(self):
        self.timings = {}
        self.start_time = None
        self.end_time = None
        self.memory_snapshots = []
        self.phase_stack = []

    def start_job(self, job_name: str = "Football Analysis"):
        """Start timing the overall job."""
        self.job_name = job_name
        self.start_time = time.time()
        self.memory_snapshots.append(
            {
                "phase": "job_start",
                "timestamp": self.start_time,
                "memory_gb": self._get_memory_usage(),
            }
        )
        print(f"⏱️  Started timing: {job_name}")
        print(
            f"🕐 Start time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def end_job(self):
        """End timing the overall job."""
        self.end_time = time.time()
        self.memory_snapshots.append(
            {
                "phase": "job_end",
                "timestamp": self.end_time,
                "memory_gb": self._get_memory_usage(),
            }
        )
        total_time = self.end_time - self.start_time
        print(f"⏱️  Finished timing: {self.job_name}")
        print(
            f"🕐 End time: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"⏰ Total execution time: {self._format_duration(total_time)}")

    def start_phase(self, phase_name: str):
        """Start timing a specific phase."""
        timestamp = time.time()
        self.phase_stack.append({"name": phase_name, "start_time": timestamp})
        self.memory_snapshots.append(
            {
                "phase": f"{phase_name}_start",
                "timestamp": timestamp,
                "memory_gb": self._get_memory_usage(),
            }
        )
        print(f"  🔄 Starting phase: {phase_name}")

    def end_phase(self, phase_name: str = None):
        """End timing the current phase."""
        if not self.phase_stack:
            print("⚠️  Warning: No active phase to end")
            return

        current_phase = self.phase_stack.pop()
        end_time = time.time()
        duration = end_time - current_phase["start_time"]

        # Verify phase name matches if provided
        if phase_name and phase_name != current_phase["name"]:
            print(
                f"⚠️  Warning: Phase name mismatch. Expected: {current_phase['name']}, Got: {phase_name}"
            )

        self.timings[current_phase["name"]] = {
            "start_time": current_phase["start_time"],
            "end_time": end_time,
            "duration": duration,
            "formatted_duration": self._format_duration(duration),
        }

        self.memory_snapshots.append(
            {
                "phase": f"{current_phase['name']}_end",
                "timestamp": end_time,
                "memory_gb": self._get_memory_usage(),
            }
        )

        print(
            f"  ✅ Completed phase: {current_phase['name']} ({self._format_duration(duration)})"
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)  # Convert to GB
        except:
            return 0.0

    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way."""
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

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timing data."""
        if not self.start_time or not self.end_time:
            return {"error": "Job timing not completed"}

        total_time = self.end_time - self.start_time

        summary = {
            "job_name": getattr(self, "job_name", "Unknown Job"),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "total_duration_seconds": total_time,
            "total_duration_formatted": self._format_duration(total_time),
            "phases": {},
            "memory_usage": {
                "peak_memory_gb": max(
                    [snap["memory_gb"] for snap in self.memory_snapshots], default=0
                ),
                "start_memory_gb": (
                    self.memory_snapshots[0]["memory_gb"]
                    if self.memory_snapshots
                    else 0
                ),
                "end_memory_gb": (
                    self.memory_snapshots[-1]["memory_gb"]
                    if self.memory_snapshots
                    else 0
                ),
            },
        }

        # Add phase timings
        for phase_name, timing_data in self.timings.items():
            summary["phases"][phase_name] = {
                "duration_seconds": timing_data["duration"],
                "duration_formatted": timing_data["formatted_duration"],
                "percentage_of_total": (timing_data["duration"] / total_time) * 100,
            }

        return summary

    def print_summary(self):
        """Print a detailed summary of timing results."""
        summary = self.get_summary()

        if "error" in summary:
            print(f"❌ {summary['error']}")
            return

        print("\n" + "=" * 80)
        print(f"📊 TIMING SUMMARY: {summary['job_name']}")
        print("=" * 80)

        print(f"🕐 Start Time: {summary['start_time']}")
        print(f"🕐 End Time: {summary['end_time']}")
        print(f"⏰ Total Duration: {summary['total_duration_formatted']}")

        print(f"\n💾 Memory Usage:")
        print(f"   Peak: {summary['memory_usage']['peak_memory_gb']:.2f} GB")
        print(f"   Start: {summary['memory_usage']['start_memory_gb']:.2f} GB")
        print(f"   End: {summary['memory_usage']['end_memory_gb']:.2f} GB")

        if summary["phases"]:
            print(f"\n📋 Phase Breakdown:")
            sorted_phases = sorted(
                summary["phases"].items(),
                key=lambda x: x[1]["duration_seconds"],
                reverse=True,
            )

            for phase_name, phase_data in sorted_phases:
                print(
                    f"   {phase_name:.<30} {phase_data['duration_formatted']:>12} ({phase_data['percentage_of_total']:.1f}%)"
                )

        print("=" * 80)

    def save_to_file(self, output_path: str = None):
        """Save timing data to JSON and CSV files."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"timing_results_{timestamp}"

        # Save JSON summary
        json_path = f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"📄 Timing summary saved to: {json_path}")

        # Save CSV with phase details
        csv_path = f"{output_path}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Phase",
                    "Duration (seconds)",
                    "Duration (formatted)",
                    "Percentage of Total",
                ]
            )

            summary = self.get_summary()
            total_time = summary["total_duration_seconds"]

            for phase_name, phase_data in summary["phases"].items():
                writer.writerow(
                    [
                        phase_name,
                        phase_data["duration_seconds"],
                        phase_data["duration_formatted"],
                        f"{phase_data['percentage_of_total']:.1f}%",
                    ]
                )

        print(f"📊 Phase details saved to: {csv_path}")


def timed_function(phase_name: str = None, collector: TimingCollector = None):
    """Decorator to automatically time function execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal phase_name, collector

            # Use function name if no phase name provided
            if phase_name is None:
                phase_name = func.__name__

            # Use global collector if none provided
            if collector is None:
                collector = get_global_collector()

            collector.start_phase(phase_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                collector.end_phase(phase_name)

        return wrapper

    return decorator


# Global timing collector instance
_global_collector = None


def get_global_collector() -> TimingCollector:
    """Get or create the global timing collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = TimingCollector()
    return _global_collector


def reset_global_collector():
    """Reset the global timing collector."""
    global _global_collector
    _global_collector = TimingCollector()


def create_timing_context_manager(phase_name: str, collector: TimingCollector = None):
    """Create a context manager for timing a code block."""
    if collector is None:
        collector = get_global_collector()

    class TimingContext:
        def __init__(self, phase_name: str, collector: TimingCollector):
            self.phase_name = phase_name
            self.collector = collector

        def __enter__(self):
            self.collector.start_phase(self.phase_name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.collector.end_phase(self.phase_name)

    return TimingContext(phase_name, collector)


def time_phase(phase_name: str, collector: TimingCollector = None):
    """Context manager for timing a phase. Usage: with time_phase('Phase Name'): ..."""
    return create_timing_context_manager(phase_name, collector)
