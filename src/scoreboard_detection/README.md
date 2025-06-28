# Scoreboard Detection Module

This module provides comprehensive scoreboard detection and score extraction functionality for football/soccer video analysis. It uses computer vision techniques and OCR to automatically detect scoreboards in video frames and extract the current score, which can then be used as the authoritative source for goal counting.

## Features

- **Automatic Scoreboard Detection**: Detects scoreboard regions using multiple computer vision techniques
- **Score Extraction**: Uses OCR to extract numerical scores from detected scoreboards
- **Temporal Consistency**: Tracks scores across frames for stability and accuracy
- **Priority System Integration**: Integrates with existing goal detection to provide authoritative scoring
- **Configurable Parameters**: Adjustable confidence thresholds and detection intervals

## Components

### ScoreboardDetector
Detects potential scoreboard regions in video frames using:
- Text region detection (MSER)
- Rectangular region detection (contour analysis)
- Color-based region detection
- Temporal filtering for consistency

### ScoreExtractor
Extracts numerical scores from detected regions using:
- Tesseract OCR for text recognition
- Image preprocessing for better OCR results
- Pattern matching for common score formats
- Confidence scoring for extracted scores

### ScoreboardAnalyzer
Main orchestrator that combines detection and extraction:
- Manages the complete analysis pipeline
- Tracks score changes over time
- Provides final authoritative scores
- Maintains analysis statistics

## Usage

### Basic Usage

```python
from src.scoreboard_detection import ScoreboardAnalyzer

# Initialize analyzer
analyzer = ScoreboardAnalyzer(
    detection_interval=30,  # Analyze every 30 frames
    min_detection_confidence=0.6,
    min_extraction_confidence=0.6
)

# Analyze a frame
result = analyzer.analyze_frame(frame, frame_number)
if result:
    print(f"Score: {result['team1_score']}-{result['team2_score']}")

# Get final score
final_score = analyzer.get_final_score()
if final_score:
    print(f"Final: {final_score['team1_score']}-{final_score['team2_score']}")
```

### Command Line Usage

Enable scoreboard detection in the main analysis:

```bash
python main.py --input video.mp4 --enable-scoreboard-detection
```

### Integration with Main Pipeline

The scoreboard detection is automatically integrated into the main football analysis pipeline when enabled. It follows this priority system for final goal counts:

1. **Manual goals** (highest priority) - from CSV configuration
2. **Scoreboard-extracted scores** - when confidence ≥ 0.7
3. **Enhanced goal detection** - from field keypoint analysis
4. **Regular goal detection** - from ball position analysis (lowest priority)

## Configuration

### Detection Parameters

- `detection_interval`: How often to run detection (every N frames)
- `min_detection_confidence`: Minimum confidence for scoreboard detection
- `min_extraction_confidence`: Minimum confidence for score extraction
- `score_stability_frames`: Frames required for score stability

### Scoreboard Detection Parameters

- `min_scoreboard_width/height`: Minimum scoreboard dimensions
- `max_scoreboard_width/height`: Maximum scoreboard dimensions
- `confidence_threshold`: Overall confidence threshold

## Dependencies

### Required
- OpenCV (`cv2`)
- NumPy
- Python 3.7+

### Optional
- `pytesseract` - For OCR functionality (highly recommended)
- `tesseract-ocr` - System dependency for pytesseract

### Installing Tesseract

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

**macOS:**
```bash
brew install tesseract
pip install pytesseract
```

**Windows:**
1. Download Tesseract installer from GitHub
2. Install and add to PATH
3. `pip install pytesseract`

## Output

### CSV Output
When scoreboard detection is enabled, additional columns are added to the output CSV files:

**Team Stats CSV:**
- `goals_scoreboard`: Goals detected from scoreboard
- `scoreboard_detected`: Whether scoreboard was found
- `scoreboard_confidence`: Confidence of scoreboard detection

**Player Stats CSV:**
- `scoreboard_detected`: Whether scoreboard was found
- `scoreboard_confidence`: Confidence of scoreboard detection

### Console Output
```
🎯 Initializing scoreboard detection system...
📊 Calculating final goal statistics...
   Manual goals provided: 0
   Scoreboard score detected: 2-1 (confidence: 0.85)
   Enhanced goals detected: 3
   Regular goals detected: 2
📊 Using scoreboard-extracted scores as final goal count
   Scoreboard confidence: 0.85
   Detection rate: 0.80
```

## Performance Considerations

### Memory Usage
- Scoreboard detection adds minimal memory overhead
- Uses larger detection intervals for memory-efficient processing
- Processes only upper portion of frames (where scoreboards typically appear)

### Processing Speed
- Detection interval can be adjusted based on video length
- Larger intervals (60+ frames) recommended for very long videos
- OCR processing is the most computationally expensive step

### Accuracy Factors
- Video quality and resolution
- Scoreboard visibility and contrast
- Text clarity and font size
- Camera stability and angle

## Troubleshooting

### Common Issues

**No scoreboard detected:**
- Check if scoreboard is visible in upper portion of frame
- Adjust confidence thresholds
- Verify video quality and resolution

**Incorrect score extraction:**
- Ensure pytesseract is properly installed
- Check scoreboard text clarity
- Adjust OCR preprocessing parameters

**Low confidence scores:**
- Increase detection interval for more stable results
- Check for consistent scoreboard appearance
- Verify scoreboard meets size requirements

### Debug Mode

Enable debug logging to see detailed detection information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

Run the test suite to verify functionality:

```bash
# Unit tests
python -m pytest tests/test_scoreboard_detection.py

# Integration tests
python -m pytest tests/test_scoreboard_integration.py

# End-to-end test with synthetic data
python tests/test_scoreboard_end_to_end.py
```

## Limitations

- Requires visible scoreboard in video frames
- OCR accuracy depends on text quality and clarity
- May not work with highly stylized or graphical scoreboards
- Performance varies with video quality and scoreboard design
- Currently supports only numerical score formats (e.g., "2-1", "3:0")

## Future Improvements

- Support for more scoreboard formats and styles
- Template matching for specific broadcast layouts
- Machine learning-based scoreboard detection
- Support for additional score information (time, player names)
- Real-time processing optimizations
