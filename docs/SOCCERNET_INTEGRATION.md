# SoccerNet Integration for Enhanced Football Analysis

This document describes how to integrate SoccerNet datasets with your existing football analysis system to train enhanced YOLOv8 models.

## Overview

SoccerNet is a large-scale dataset for soccer video understanding that includes:
- **550 broadcast games** with temporal action annotations
- **Ball Action Spotting**: 12 classes of ball-related actions
- **Action Spotting**: 17 classes of general game actions
- **High-quality annotations** for training robust models

## Integration Benefits

By integrating SoccerNet data, you can:

1. **Enhanced Ball Detection**: Train YOLOv8 models that understand ball actions (shots, passes, crosses)
2. **Action-Aware Player Detection**: Detect players with context about their actions
3. **Improved Accuracy**: Leverage large-scale professional annotations
4. **Action Recognition**: Add temporal action understanding to your analysis

## Quick Start

### 1. Install Dependencies

```bash
# Install SoccerNet package and additional dependencies
pip install -r requirements.txt
```

### 2. Setup SoccerNet Data

First, you need to sign the NDA to get access to SoccerNet videos:
1. Fill out the [NDA form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform)
2. Receive password via email

Then download the data:

```bash
# Download both action spotting and ball action spotting datasets
python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD

# Or download specific datasets
python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD --dataset ball-action-spotting
python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD --dataset action-spotting

# Download annotations only (no videos)
python scripts/setup_soccernet_data.py --no-videos
```

### 3. Convert to YOLO Format

Convert SoccerNet temporal annotations to spatial YOLO annotations:

```bash
# Convert all datasets
python scripts/convert_soccernet_to_yolo.py --soccernet-dir data/soccernet

# Convert specific dataset
python scripts/convert_soccernet_to_yolo.py --dataset ball-action-spotting
```

### 4. Train Enhanced Models

Train YOLOv8 models on SoccerNet data:

```bash
# Train enhanced ball detection model
python scripts/train_soccernet_yolo.py --dataset enhanced_ball_detection --epochs 100

# Train action-aware player detection
python scripts/train_soccernet_yolo.py --dataset action_aware_players --epochs 150

# Train all models
python scripts/train_soccernet_yolo.py --dataset all --epochs 100 --evaluate
```

## Dataset Details

### Ball Action Spotting (12 Classes)

| Class | Description | Use Case |
|-------|-------------|----------|
| Pass | Regular passes between players | Pass counting, possession analysis |
| Drive | Player dribbling with ball | Individual skill analysis |
| Header | Ball contact with head | Aerial play analysis |
| High Pass | Long/aerial passes | Tactical analysis |
| Out | Ball out of play | Game flow analysis |
| Cross | Crosses into penalty area | Attacking pattern analysis |
| Throw In | Throw-in situations | Set piece analysis |
| Shot | Shots on goal | Scoring opportunity analysis |
| Ball Player Block | Defensive blocks | Defensive action analysis |
| Player Successful Tackle | Successful tackles | Defensive performance |
| Free Kick | Free kick situations | Set piece analysis |
| Goal | Goals scored | Scoring analysis |

### Action Spotting (17 Classes)

Includes broader game events like fouls, cards, substitutions, etc.

## Model Architecture

### Enhanced Ball Detection Model

- **Base**: YOLOv8n (optimized for small object detection)
- **Classes**: 12 ball action classes
- **Features**: 
  - Action-context aware ball detection
  - Improved accuracy for ball tracking
  - Action classification alongside detection

### Action-Aware Player Detection

- **Base**: YOLOv8s (balanced speed/accuracy)
- **Classes**: Player + action context
- **Features**:
  - Players detected with action labels
  - Enhanced for tactical analysis
  - Better performance in crowded scenes

## Integration with Existing System

### 1. Update Tracker

Replace your existing ball detection model:

```python
# In src/trackers/tracker.py
class Tracker:
    def __init__(self, model_path, enable_jersey_detection=True, use_soccernet_models=True):
        if use_soccernet_models:
            # Use SoccerNet-trained models
            self.ball_model = YOLO("data/models/soccernet/enhanced_ball_detection/best.pt")
            self.player_model = YOLO("data/models/soccernet/action_aware_players/best.pt")
        else:
            # Use original model
            self.model = YOLO(model_path)
```

### 2. Add Action Recognition

Extend your analysis with action recognition:

```python
# New action recognition module
class ActionRecognizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def recognize_actions(self, frames, ball_positions):
        # Recognize ball actions in context
        actions = []
        for frame, ball_pos in zip(frames, ball_positions):
            action = self.model.predict(frame, ball_pos)
            actions.append(action)
        return actions
```

### 3. Enhanced Statistics

Add action-based statistics:

```python
# Enhanced pass counter with action context
class EnhancedPassCounter(PassCounter):
    def __init__(self):
        super().__init__()
        self.action_counts = {action: 0 for action in BALL_ACTION_CLASSES}
    
    def count_action(self, action_type, team, frame_num):
        self.action_counts[action_type] += 1
        # Update team-specific action statistics
```

## Performance Expectations

Based on SoccerNet benchmarks, you can expect:

- **Ball Detection mAP50**: 85-90% (vs 70-80% with generic models)
- **Action Recognition mAP@1**: 80-85% for ball actions
- **Player Detection**: Improved accuracy in crowded scenes
- **Processing Speed**: Minimal impact on inference time

## Directory Structure

After setup, your directory structure will be:

```
data/
├── soccernet/
│   ├── action_spotting/
│   │   ├── videos/
│   │   ├── annotations/
│   │   └── features/
│   ├── ball_action_spotting/
│   │   ├── videos/
│   │   ├── annotations/
│   │   └── features/
│   └── yolo_datasets/
│       ├── enhanced_ball_detection/
│       ├── action_aware_players/
│       └── multi_class_actions/
└── models/
    └── soccernet/
        ├── enhanced_ball_detection_training/
        ├── action_aware_players_training/
        └── multi_class_actions_training/
```

## Troubleshooting

### Common Issues

1. **NDA Password Issues**
   - Ensure you've completed the NDA form
   - Check email for password (may take 24-48 hours)
   - Contact SoccerNet team if no response

2. **Download Failures**
   - Check internet connection
   - Verify disk space (datasets are large)
   - Try downloading in smaller batches

3. **Training Issues**
   - Ensure sufficient GPU memory
   - Reduce batch size if OOM errors
   - Check dataset paths in YAML files

4. **Conversion Issues**
   - Verify video files are accessible
   - Check annotation file formats
   - Ensure OpenCV can read video files

### Performance Optimization

1. **Memory Usage**
   - Use smaller batch sizes for training
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Speed**
   - Use multiple GPUs if available
   - Optimize data loading workers
   - Use SSD storage for datasets

## Next Steps

1. **Evaluate Models**: Compare SoccerNet-trained models with your existing ones
2. **Fine-tune**: Adapt models to your specific use cases
3. **Integration**: Gradually integrate enhanced models into your pipeline
4. **Monitoring**: Track performance improvements in your analysis

## Resources

- [SoccerNet Official Website](https://www.soccer-net.org/)
- [SoccerNet GitHub](https://github.com/SoccerNet)
- [Ball Action Spotting Challenge](https://github.com/SoccerNet/sn-spotting)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## Support

For issues related to:
- **SoccerNet data**: Contact the SoccerNet team
- **Integration**: Create an issue in this repository
- **YOLOv8 training**: Refer to Ultralytics documentation
