# Enhanced Goal Detection Feature

This module provides advanced goal detection capabilities for football video analysis using field keypoints detection and ball tracking.

## Overview

The enhanced goal detection system consists of two main components:

1. **FieldKeypointsDetector**: Detects field keypoints (goal posts, penalty areas, etc.) using a YOLO model
2. **GoalDetector**: Uses field keypoints and ball tracking to accurately detect goals

## Features

### FieldKeypointsDetector
- Detects field keypoints using YOLO model (`best_keypoint.pt`)
- Identifies goal posts, penalty areas, center circle, and other field markers
- Dynamically calculates goal area boundaries based on detected keypoints
- Provides fallback to hardcoded areas if keypoints are not detected
- Visualizes detected keypoints and goal areas

### GoalDetector
- Enhanced goal detection using field keypoints
- Ball trajectory analysis for goal validation
- Cooldown mechanism to prevent duplicate goal detections
- Comprehensive goal statistics tracking
- Integration with existing player and team tracking

## Usage

### Basic Usage

```python
from goal_detection import FieldKeypointsDetector, GoalDetector

# Initialize components
keypoints_detector = FieldKeypointsDetector("models/best_keypoint.pt")
goal_detector = GoalDetector(keypoints_detector)

# Process a frame
goal_detector.update_keypoints(frame)

# Detect goals
goal_event = goal_detector.detect_goal(ball_position, player_id, team, frame_num)

if goal_event:
    print(f"Goal detected: {goal_event}")
```

### Integration with Main Pipeline

The goal detection is automatically integrated into the main analysis pipeline in `main.py`. It runs alongside the existing goal detection system for comparison.

## Configuration

### Field Keypoint Classes

The system recognizes the following field keypoints:

- Goal posts (left/right, top/bottom)
- Penalty areas (left/right, top/bottom)
- Center circle (top/bottom/left/right)
- Halfway line (top/bottom)

### Parameters

- `confidence_threshold`: Minimum confidence for keypoint detection (default: 0.7)
- `goal_cooldown_frames`: Frames to wait before detecting another goal (default: 60)
- `max_trajectory_length`: Maximum ball trajectory points to track (default: 10)

## Output

### Goal Statistics

The system provides comprehensive statistics:

```python
stats = goal_detector.get_goal_statistics()
# Returns:
# {
#     "team_goals": {1: 2, 2: 1},
#     "player_goals": {5: {"goals": 1, "team": 1}},
#     "goal_events": [...],
#     "total_goals": 3
# }
```

### Goal Events

Each detected goal includes:
- Team that scored
- Player who scored (if known)
- Frame number
- Goal side (left/right)
- Ball position
- Ball trajectory leading to goal

### CSV Export

Enhanced goal statistics are exported to CSV files:
- Player stats include both regular and enhanced goal counts
- Team stats include both regular and enhanced goal counts

## Visualization

The system provides several visualization features:

1. **Field Keypoints**: Green circles showing detected keypoints
2. **Goal Areas**: Colored rectangles showing goal boundaries
3. **Ball Trajectory**: Yellow lines showing recent ball movement
4. **Goal Score**: Real-time score display on video

## Testing

Run the test suite to verify the implementation:

```bash
python test_goal_detection.py
```

## Model Requirements

The system requires the field keypoints YOLO model:
- `models/best_keypoint.pt`: Field keypoints detection model

Make sure this model is available in your models directory.

## Comparison with Original System

The enhanced system provides several improvements over the original goal detection:

1. **Dynamic Goal Areas**: Uses actual field keypoints instead of hardcoded coordinates
2. **Trajectory Validation**: Analyzes ball movement to validate goals
3. **Better Accuracy**: Reduces false positives through multiple validation methods
4. **Comprehensive Statistics**: Detailed goal events with context
5. **Visual Feedback**: Real-time visualization of detection process

## Troubleshooting

### Common Issues

1. **No keypoints detected**:
   - Check if `best_keypoint.pt` model exists
   - Verify model confidence threshold
   - System will fallback to hardcoded areas

2. **Goals not detected**:
   - Check ball tracking accuracy
   - Verify goal area boundaries
   - Adjust confidence thresholds

3. **False positive goals**:
   - Increase cooldown frames
   - Check trajectory validation logic
   - Verify goal area boundaries

### Debug Mode

Enable debug output by modifying the print statements in the goal detection classes to see detailed information about the detection process.

## Future Enhancements

Potential improvements for the goal detection system:

1. **Offside Detection**: Use field keypoints to detect offside situations
2. **Goal Line Technology**: More precise goal line crossing detection
3. **Celebration Detection**: Detect player celebrations after goals
4. **Replay Analysis**: Automatic replay generation for goal events
5. **Multi-Camera Support**: Combine multiple camera angles for better accuracy
