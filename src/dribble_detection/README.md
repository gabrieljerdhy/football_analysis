# Dribble Detection Module

This module provides comprehensive dribble detection capabilities for football video analysis. A dribble is detected when a player successfully moves past an opponent while maintaining ball control.

## Overview

The dribble detection system analyzes:
1. **Player movement patterns** and direction changes
2. **Ball possession transitions** between players
3. **Proximity between attacking and defending players**
4. **Speed and acceleration changes** during ball control

## Features

### DribbleDetector

- **Movement Analysis**: Tracks player and ball positions to detect dribbling patterns
- **Opponent Interaction**: Identifies when players are in close proximity to opponents
- **Trajectory Analysis**: Analyzes direction changes and movement smoothness
- **Confidence Scoring**: Provides confidence scores for each detected dribble
- **Ball Control Quality**: Measures consistency of ball control during dribbles

### DribbleAnalyzer

- **Integration**: Seamlessly integrates with existing tracking and detection systems
- **Statistics Generation**: Provides comprehensive team and player dribble statistics
- **CSV Export**: Exports results in CSV format matching existing output structure
- **Real-time Processing**: Processes frames in real-time during video analysis

### DribbleEvent

- **Comprehensive Tracking**: Stores detailed information about each dribble event
- **Performance Metrics**: Includes duration, distance, speed, and quality metrics
- **Opponent Data**: Tracks interaction with defending players
- **Validation**: Built-in validation to ensure quality and consistency

## Usage

### Basic Usage

```python
from dribble_detection import DribbleAnalyzer

# Initialize analyzer
analyzer = DribbleAnalyzer(
    min_dribble_distance=30.0,
    min_dribble_duration=10,
    confidence_threshold=0.4
)

# Analyze frame
dribble_event = analyzer.analyze_frame(
    frame_num=100,
    players=players_dict,
    ball_position=(x, y),
    ball_possessor=player_id,
    ball_team=team_id
)

# Get statistics
team_stats = analyzer.detector.get_team_statistics()
player_stats = analyzer.detector.get_player_statistics()
```

### Integration with Main Pipeline

```python
# In main processing loop
for frame_num, frame in enumerate(video_frames):
    # ... existing tracking code ...
    
    # Detect dribbles
    dribble_event = dribble_analyzer.analyze_frame(
        frame_num, players, ball_position, ball_possessor, ball_team
    )
    
    if dribble_event:
        print(f"Dribble detected: Player {dribble_event.player_id}, "
              f"Team {dribble_event.team}, Confidence {dribble_event.confidence:.3f}")

# Export results
analyzer.export_to_csv(output_dir="data/output", video_name="match")
```

### Complete Match Analysis

```python
# Analyze complete match data
results = analyzer.analyze_complete_match(tracks, player_assignments)

# Access results
team_statistics = results['team_statistics']
player_statistics = results['player_statistics']
dribble_events = results['dribble_events']
```

## Configuration

### Detection Parameters

- `min_dribble_distance`: Minimum distance for a valid dribble (default: 30.0 pixels)
- `min_dribble_duration`: Minimum duration for a valid dribble (default: 10 frames)
- `max_dribble_duration`: Maximum duration for a valid dribble (default: 120 frames)
- `opponent_proximity_threshold`: Distance threshold for opponent interaction (default: 80.0 pixels)
- `ball_control_threshold`: Maximum distance between player and ball for control (default: 50.0 pixels)
- `confidence_threshold`: Minimum confidence for dribble detection (default: 0.4)

### Confidence Scoring

The confidence score is calculated using multiple factors:
- **Distance factor** (30% weight): Total distance covered during dribble
- **Duration factor** (20% weight): Duration of the dribble
- **Direction changes** (25% weight): Number of direction changes
- **Speed factor** (15% weight): Maximum speed achieved
- **Opponent proximity** (10% weight): Number of nearby opponents

## Output Format

### Team Statistics CSV

Columns added to existing team statistics:
- `dribbles_detected`: Total number of detected dribbles
- `avg_dribble_confidence`: Average confidence score for dribble detections
- `dribble_events_count`: Number of dribble events identified

### Player Statistics CSV

- `player_id`: Player identifier
- `dribbles_detected`: Total dribbles by player
- `successful_dribbles`: Successful dribbles by player
- `dribble_success_rate`: Success rate (0-1)
- `avg_dribble_confidence`: Average confidence score
- `dribble_events_count`: Number of events

### Dribble Events CSV

Detailed information for each dribble event:
- `frame_num`: Frame where dribble was completed
- `team`: Team of the dribbling player
- `player_id`: ID of the dribbling player
- `confidence`: Confidence score (0-1)
- `start_frame`: Frame where dribble started
- `end_frame`: Frame where dribble ended
- `duration_frames`: Duration in frames
- `distance_covered`: Total distance covered
- `direction_changes`: Number of direction changes
- `avg_speed`: Average speed during dribble
- `opponent_player_id`: ID of closest opponent
- `opponent_distance`: Distance to closest opponent
- `ball_control_consistency`: Ball control quality score

## Integration with Existing Systems

The dribble detection module follows the same patterns as existing detection features:

1. **Similar Architecture**: Follows the same structure as `goal_detection` and `pass_counter`
2. **CSV Integration**: Adds columns to existing team statistics CSV format
3. **Statistics Pattern**: Uses the same statistics aggregation patterns
4. **Event Tracking**: Similar event-based tracking as goal and tackle detection
5. **Confidence Scoring**: Uses similar confidence scoring methodology

## Performance Considerations

- **Memory Efficient**: Uses deques with limited history for position tracking
- **Configurable Thresholds**: Adjustable parameters for different video qualities
- **Validation**: Built-in validation to prevent false positives
- **Real-time Processing**: Designed for real-time video processing

## Testing

```python
# Test basic functionality
python tests/test_dribble_detection.py

# Test integration with main pipeline
python tests/test_dribble_integration.py
```

## Dependencies

- numpy: For mathematical calculations
- collections: For efficient data structures
- typing: For type hints
- csv: For CSV export functionality
- os: For file operations

The module integrates with existing utilities:
- `src.utils.bbox_utils`: For bounding box operations
- Existing tracking and detection systems
