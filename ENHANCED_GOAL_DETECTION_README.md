# Enhanced Goal Detection and Counting System

## Overview

This document describes the enhanced goal detection and counting system implemented in the football analysis project. The system provides accurate goal tracking, comprehensive statistics, and consolidated CSV export functionality.

## Key Features

### 1. Enhanced Goal Detection Logic
- **Advanced Trajectory Analysis**: Multi-factor validation using direction consistency, speed analysis, trajectory smoothness, and goal approach angle
- **Confidence Scoring**: Each goal detection includes a confidence score based on trajectory analysis
- **Improved Cooldown System**: Prevents duplicate goal detections with enhanced timing controls
- **Field Keypoint Integration**: Uses field keypoint detection for accurate goal area identification

### 2. Dual Goal Counting System
- **Enhanced Goal Count**: Real-time detection results from the advanced goal detection system
- **Final Goal Count**: Validated/corrected goal tally (primary metric for CSV export)
- **Priority System**: Manual goals > Enhanced goals > Regular detection goals

### 3. Consolidated CSV Export
The system exports exactly **two CSV files** as requested:

#### Team Statistics CSV (`{video_name}_team_stats.csv`)
- `team`: Team ID (1 or 2)
- `passes`: Number of passes
- `goals_regular`: Goals from basic detection
- `goals_enhanced`: Goals from enhanced detection
- `goals_final`: Final validated goal count (primary metric)
- `tackles`: Number of tackles
- `interceptions`: Number of interceptions
- `goal_events_count`: Number of goal events detected
- `avg_goal_confidence`: Average confidence score for team's goals

#### Player Statistics CSV (`{video_name}_player_stats.csv`)
- `player_id`: Player identifier
- `team`: Player's team
- `passes`: Number of passes
- `goals_regular`: Goals from basic detection
- `goals_enhanced`: Goals from enhanced detection
- `goals_final`: Final validated goal count (primary metric)
- `tackles`: Number of tackles
- `interceptions`: Number of interceptions
- `goal_events_count`: Number of goal events by player
- `avg_goal_confidence`: Average confidence score for player's goals

### 4. Performance Optimizations
- Maintains existing memory optimizations (deactivated camera movement and speed/distance estimators)
- Efficient trajectory analysis with configurable parameters
- Optimized goal area calculations based on video dimensions

## Technical Implementation

### Enhanced GoalDetector Class

#### New Attributes
```python
# Enhanced goal tracking
self.final_team_goals = {1: 0, 2: 0}     # Primary metric
self.final_player_goals = {}              # Primary metric
self.goal_validation_history = []         # Validation decisions
self.goal_confidence_scores = []          # Confidence tracking

# Enhanced trajectory analysis
self.min_trajectory_for_goal = 5          # Minimum points for validation
self.goal_direction_threshold = 0.6       # Directional consistency
self.goal_speed_threshold = 10            # Minimum speed threshold
```

#### Enhanced Methods
- `_validate_goal_trajectory()`: Comprehensive trajectory validation
- `_calculate_direction_consistency()`: Direction analysis
- `_calculate_speed_consistency()`: Speed validation
- `_calculate_trajectory_smoothness()`: Smoothness analysis
- `_calculate_goal_approach_angle()`: Approach angle validation
- `set_final_goal_counts()`: Set final validated counts

### Goal Priority System

The system uses a three-tier priority system:

1. **Manual Goals** (Highest Priority)
   - Loaded from CSV configuration file
   - Override all automatic detection
   - Used for ground truth validation

2. **Enhanced Goals** (Medium Priority)
   - Advanced trajectory-based detection
   - Confidence scoring and validation
   - Used when no manual goals provided

3. **Regular Goals** (Lowest Priority)
   - Basic position-based detection
   - Fallback when enhanced detection unavailable

### Trajectory Validation Algorithm

The enhanced system validates goals using four metrics:

1. **Direction Consistency (40% weight)**
   - Ensures ball moves toward the correct goal
   - Calculates percentage of correct directional movements

2. **Speed Consistency (30% weight)**
   - Validates realistic ball movement speed
   - Penalizes very slow movement (likely tracking errors)

3. **Trajectory Smoothness (20% weight)**
   - Penalizes erratic ball movement
   - Detects sudden direction changes

4. **Goal Approach Angle (10% weight)**
   - Validates ball approaches goal from reasonable angle
   - Uses dot product similarity with ideal approach vector

**Final Confidence Score**: Weighted sum of all metrics (threshold: 0.5)

## Usage

### Basic Usage
```python
from goal_detection import FieldKeypointsDetector, GoalDetector
from utils.goal_utils import export_consolidated_goal_statistics

# Initialize enhanced goal detection
keypoints_detector = FieldKeypointsDetector("models/best_fk.pt")
goal_detector = GoalDetector(keypoints_detector)

# Process video frames...
for frame_num, frame in enumerate(video_frames):
    goal_detector.update_keypoints(frame)
    goal_event = goal_detector.detect_goal(ball_position, player_id, team, frame_num)

# Get comprehensive statistics
stats = goal_detector.get_goal_statistics()

# Export consolidated CSV files
team_csv, player_csv = export_consolidated_goal_statistics(
    video_name, pass_counter, stats, final_team_goals, final_player_goals, tackle_counter
)
```

### Manual Goal Configuration
Create a CSV file with manual goals:
```csv
team,player_id,frame_num,comment
1,5,1200,Goal by player 5 at 40 seconds
2,,3600,Goal by unknown player at 120 seconds
```

Use with `--goals-config` parameter:
```bash
python main.py --input video.mp4 --goals-config manual_goals.csv
```

## Testing

Run the comprehensive test suite:
```bash
python debug_and_tests/test_enhanced_goal_detection.py
```

Tests include:
- Enhanced GoalDetector functionality
- Trajectory validation algorithms
- Consolidated CSV export
- Goal priority system

## Configuration Parameters

### Goal Detection Parameters
- `goal_cooldown_frames`: Frames to wait before detecting another goal (default: 60)
- `min_trajectory_for_goal`: Minimum trajectory points for validation (default: 5)
- `goal_direction_threshold`: Minimum directional consistency (default: 0.6)
- `goal_speed_threshold`: Minimum ball speed for validation (default: 10)

### Trajectory Analysis Weights
- Direction consistency: 40%
- Speed consistency: 30%
- Trajectory smoothness: 20%
- Goal approach angle: 10%

## Performance Considerations

- **Memory Optimization**: Camera movement and speed/distance estimators remain deactivated by default
- **Real-time Processing**: Enhanced detection maintains real-time performance
- **Scalability**: System handles videos of various resolutions and frame rates
- **Accuracy vs Speed**: Configurable parameters allow tuning for specific requirements

## Output Files

The system generates exactly two consolidated CSV files per video:
1. `{video_name}_team_stats.csv` - Team-level statistics
2. `{video_name}_player_stats.csv` - Player-level statistics

Both files include the **final goal count** as the primary metric, with enhanced and regular detection counts provided for comparison and analysis.

## Integration

The enhanced goal detection system integrates seamlessly with:
- Existing player tracking system
- Team identification components
- Jersey number recognition
- Tackle and interception detection
- Video annotation and visualization

## Future Enhancements

Potential improvements for future versions:
- Machine learning-based goal validation
- Real-time confidence adjustment
- Advanced trajectory prediction
- Integration with referee decision analysis
- Multi-camera angle support
