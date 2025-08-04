# Football Analysis Logging System

## Overview

The comprehensive logging system for the football analysis application provides detailed tracking of analysis runs, performance metrics, and debugging information. Each analysis run generates a unique log file with structured information about the entire processing pipeline.

## Features

- **Unique Log Files**: Each analysis run creates a timestamped log file in `data/output/`
- **Structured Logging**: Organized stages with start/end times and performance metrics
- **Multiple Log Levels**: INFO, WARNING, ERROR, DEBUG support
- **Performance Tracking**: Memory usage, processing speed, and timing metrics
- **Event Logging**: Detailed capture of goals, passes, tackles, and other detections
- **Human-Readable Format**: Easy to read logs with timestamps and emojis
- **JSON Summary**: Machine-readable summary file with complete analysis metadata

## Log File Naming Convention

Log files are automatically named using one of these patterns:
- `{match_name}_analysis_log_{timestamp}.log`
- `analysis_log_YYYY-MM-DD_HH-MM-SS.log`

Example: `bournemouth_vs_manchester_analysis_log_2024-07-28_14-30-45.log`

## Usage

### Basic Usage

```python
from src.utils.logging_utils import create_analysis_logger, set_global_logger

# Create logger for an analysis run
logger = create_analysis_logger(
    input_video_path="data/input_videos/match.mp4",
    output_dir="data/output",
    match_name="Premier League Match",
    enable_console=False  # Set to True for console output
)

# Set as global logger for convenience functions
set_global_logger(logger)

# Log configuration
config = {
    'enable_camera_movement': True,
    'enable_scoreboard_detection': True,
    'memory_efficient': True,
    'batch_size': 100
}
logger.set_configuration(config)
```

### Logging Analysis Stages

```python
# Start a processing stage
logger.start_stage("object_tracking", "Tracking players and ball")

# ... do processing work ...

# End the stage with results
logger.end_stage("object_tracking", {
    "total_detections": 1247,
    "processing_time": 45.2
})
```

### Logging Detection Events

```python
from src.utils import log_detection_event

# Log various detection events
log_detection_event("goal", frame_num=3456, confidence=0.91, 
                   details={"team": 1, "player_id": 10})

log_detection_event("pass", frame_num=1250, confidence=0.85,
                   details={"team": 2, "player_id": 7})

log_detection_event("tackle", frame_num=1890, confidence=0.72,
                   details={"team": 1, "player_id": 23})
```

### Logging Performance Metrics

```python
from src.utils import log_performance_metric, log_memory_usage

# Log performance metrics
log_performance_metric("processing_speed_fps", 24.3)
log_performance_metric("total_passes_detected", 234)
log_performance_metric("detection_accuracy", 0.87)

# Log memory usage
log_memory_usage(4.2, "object_tracking")
```

### Logging Scoreboard Detection

```python
from src.utils import log_scoreboard_event

# Log scoreboard detection results
log_scoreboard_event(frame_num=3500, score="1-0", confidence=0.87, success=True)
log_scoreboard_event(frame_num=5300, score="1-1", confidence=0.82, success=True)
```

## Log File Structure

### Sample Log Output

```
2024-07-28 14:30:45 | INFO     | ================================================================================
2024-07-28 14:30:45 | INFO     | 🚀 FOOTBALL ANALYSIS STARTED
2024-07-28 14:30:45 | INFO     | 📅 Timestamp: 2024-07-28 14:30:45
2024-07-28 14:30:45 | INFO     | 🏟️  Match: Bournemouth vs Manchester
2024-07-28 14:30:45 | INFO     | 📝 Log File: data/output/bournemouth_vs_manchester_analysis_log_2024-07-28_14-30-45.log
2024-07-28 14:30:45 | INFO     | ================================================================================
2024-07-28 14:30:45 | INFO     | ⚙️  Analysis Configuration:
2024-07-28 14:30:45 | INFO     |    • input_video_path: data/input_videos/match.mp4
2024-07-28 14:30:45 | INFO     |    • enable_camera_movement: False
2024-07-28 14:30:45 | INFO     |    • enable_scoreboard_detection: True
2024-07-28 14:30:45 | INFO     |    • memory_efficient: True
2024-07-28 14:30:45 | INFO     |    • batch_size: 100
2024-07-28 14:30:45 | INFO     | 📹 Input Video: data/input_videos/match.mp4
2024-07-28 14:30:45 | INFO     | 📊 Video Metadata:
2024-07-28 14:30:45 | INFO     |    • width: 1920
2024-07-28 14:30:45 | INFO     |    • height: 1080
2024-07-28 14:30:45 | INFO     |    • total_frames: 153294
2024-07-28 14:30:45 | INFO     |    • duration_seconds: 6387.25
2024-07-28 14:30:45 | INFO     | 🔄 Starting Stage: video_loading - Loading video frames
2024-07-28 14:30:47 | INFO     | ⏱️  Stage Duration: 2.15 seconds
2024-07-28 14:30:47 | INFO     | ✅ Completed Stage: video_loading
2024-07-28 14:30:47 | INFO     | 🔄 Starting Stage: object_tracking - Tracking players and ball
2024-07-28 14:32:15 | INFO     | 🎯 goal detected at frame 3456 (confidence: 0.910) - {'team': 1, 'player_id': 10}
2024-07-28 14:32:45 | INFO     | 🎯 pass detected at frame 4567 (confidence: 0.850) - {'team': 2, 'player_id': 7}
2024-07-28 14:33:12 | INFO     | ⏱️  Stage Duration: 85.23 seconds
2024-07-28 14:33:12 | INFO     | ✅ Completed Stage: object_tracking
```

## JSON Summary File

Each analysis run also generates a JSON summary file with complete metadata:

```json
{
  "analysis_info": {
    "match_name": "Bournemouth vs Manchester",
    "start_time": "2024-07-28T14:30:45",
    "end_time": "2024-07-28T14:45:23",
    "total_duration_seconds": 878.5,
    "log_file": "data/output/bournemouth_vs_manchester_analysis_log_2024-07-28_14-30-45.log"
  },
  "performance_metrics": {
    "total_frames_processed": 153294,
    "processing_speed_fps": 24.3,
    "memory_usage_peak_gb": 4.2,
    "stages_completed": [
      {"name": "video_loading", "duration_seconds": 2.15},
      {"name": "object_tracking", "duration_seconds": 85.23}
    ],
    "errors_encountered": 0,
    "warnings_encountered": 2
  },
  "analysis_context": {
    "input_video_path": "data/input_videos/match.mp4",
    "video_metadata": {
      "width": 1920,
      "height": 1080,
      "total_frames": 153294
    },
    "team_statistics": {
      "team_1": {"goals": 2, "passes": 145},
      "team_2": {"goals": 1, "passes": 89}
    },
    "output_files": {
      "output_video": "data/output_videos/match_output.avi",
      "team_statistics_csv": "data/output/match_team_stats.csv"
    }
  }
}
```

## Integration with Main Pipeline

The logging system is automatically integrated into the main analysis pipeline (`main.py`). It logs:

1. **Analysis Configuration**: All input parameters and settings
2. **Video Metadata**: Frame count, resolution, duration, etc.
3. **Processing Stages**: Each major stage with timing and results
4. **Detection Events**: Goals, passes, tackles, dribbles with confidence scores
5. **Scoreboard Detection**: Score extraction results and confidence
6. **Performance Metrics**: Processing speed, memory usage, accuracy
7. **Team Statistics**: Final team and player statistics
8. **Output Files**: All generated files (videos, CSVs, etc.)
9. **Error/Warning Tracking**: Any issues encountered during processing

## Demo Script

Run the demo script to see the logging system in action:

```bash
python examples/logging_demo.py
```

This will create a sample log file demonstrating all logging features.

## Benefits

1. **Debugging**: Detailed logs help identify issues in the analysis pipeline
2. **Performance Monitoring**: Track processing speed and memory usage over time
3. **Quality Assurance**: Monitor detection confidence and accuracy metrics
4. **Audit Trail**: Complete record of analysis runs for reproducibility
5. **Progress Tracking**: Real-time visibility into long-running analysis jobs
6. **Error Analysis**: Comprehensive error and warning capture for troubleshooting

## File Locations

- **Log Files**: `data/output/{match_name}_analysis_log_{timestamp}.log`
- **Summary Files**: `data/output/{match_name}_analysis_log_{timestamp}_summary.json`
- **Demo Script**: `examples/logging_demo.py`
- **Source Code**: `src/utils/logging_utils.py`
