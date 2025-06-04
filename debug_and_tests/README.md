# Debug and Test Scripts

This directory contains all debug and test scripts for the football analysis project. These scripts help troubleshoot issues, test individual components, and analyze system performance.

## 🔧 Debug Scripts

### `debug_ball_simple.py`

**Purpose**: Simple step-by-step debugging of ball tracking

- Tests ball detection, position calculation, and interpolation
- Shows raw tracking data before and after processing
- Useful for understanding ball tracking pipeline

**Usage**:

```bash
cd debug_and_tests
python debug_ball_simple.py
```

### `debug_goal_detection.py`

**Purpose**: Comprehensive goal detection debugging tool

- Analyzes ball tracking, field keypoints, and goal detection
- Creates debug videos with visual annotations
- Command-line interface with options

**Usage**:

```bash
cd debug_and_tests
python debug_goal_detection.py --input ../input_videos/video.mp4 --output ../output_videos/debug.avi --frames 100
```

### `debug_goal_real.py`

**Purpose**: Real-world goal detection analysis

- Analyzes ball positions in actual game videos
- Identifies potential goal events and areas
- Creates visual debug videos with goal area overlays

**Usage**:

```bash
cd debug_and_tests
python debug_goal_real.py
```

### `simple_debug.py`

**Purpose**: Quick and simple goal detection testing

- Tests goal detection with manual ball positions
- Validates goal area detection logic
- Good for quick verification

**Usage**:

```bash
cd debug_and_tests
python simple_debug.py
```

## 🧪 Test Scripts

### `test_goal_detection.py`

**Purpose**: Unit tests for goal detection components

- Tests FieldKeypointsDetector functionality
- Tests GoalDetector integration
- Validates component initialization and basic operations

**Usage**:

```bash
cd debug_and_tests
python test_goal_detection.py
```

### `yolo_inference.py`

**Purpose**: YOLO model inference testing

- Tests YOLO model loading and inference
- Useful for debugging object detection issues
- Validates model performance

**Usage**:

```bash
cd debug_and_tests
python yolo_inference.py
```

## 📁 File Organization

All debug and test files have been moved to this directory to keep the main project clean and organized. The scripts use relative paths to access the main project modules and resources.

## 🚀 Running Scripts

**Important**: All scripts should be run from within the `debug_and_tests` directory to ensure proper path resolution:

```bash
# Navigate to debug directory first
cd debug_and_tests

# Then run any script
python script_name.py
```

**Note**: All scripts have been updated to use relative paths (`../models/`, `../input_videos/`, etc.) to access project resources from the debug directory.

## 📊 Common Use Cases

### Debugging Goal Detection Issues

1. Start with `simple_debug.py` for quick testing
2. Use `debug_goal_detection.py` for comprehensive analysis
3. Try `debug_goal_real.py` for real-world video analysis

### Testing Components

1. Use `test_goal_detection.py` to verify component functionality
2. Use `debug_ball_simple.py` to debug ball tracking specifically

### Creating Debug Videos

- `debug_goal_detection.py` and `debug_goal_real.py` can create annotated debug videos
- Useful for visual analysis of detection performance

## 🔗 Dependencies

All scripts depend on the main project modules:

- `goal_detection/`
- `trackers/`
- `utils/`
- `player_ball_assigner/`
- Model files in `models/`
- Input videos in `input_videos/`

Make sure the main project is properly set up before running these debug scripts.

## 🚀 Memory Optimization Testing

### `demo_memory_optimization.py`

**Purpose**: Demonstration and help for memory optimization features

- Shows how to use new memory optimization command line options
- Provides system requirements check
- Explains memory impact of different configurations

**Usage**:

```bash
cd debug_and_tests
python demo_memory_optimization.py --help
```

### `test_memory_optimization_simple.py`

**Purpose**: Unit tests for memory optimization changes

- Tests that new command line parameters work correctly
- Verifies default values for memory optimization flags
- Validates that imports still work correctly

**Usage**:

```bash
cd debug_and_tests
python test_memory_optimization_simple.py
```

## 💾 Memory Optimization Features

The main application now includes memory optimization options:

- **Camera movement estimation**: Disabled by default (use `--enable-camera-movement` to enable)
- **Speed and distance estimation**: Disabled by default (use `--enable-speed-distance` to enable)
- **Memory savings**: 40-60% reduction in memory usage with default settings
- **Core features**: Goal detection, tracking, and statistics always enabled
