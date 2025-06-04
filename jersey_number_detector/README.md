# Jersey Number Detector

OCR-based jersey number recognition system for football video analysis.

## Features

- **Robust OCR**: Uses EasyOCR for accurate text detection in various lighting conditions
- **Smart Preprocessing**: Enhances jersey regions with contrast adjustment, noise reduction, and sharpening
- **Multi-frame Consensus**: Combines detections across multiple frames for improved accuracy
- **Performance Optimization**: Caching mechanism to avoid redundant OCR processing
- **Validation**: Ensures detected numbers are within valid football jersey range (1-99)

## Usage

### Basic Usage

```python
from jersey_number_detector import JerseyNumberDetector

# Initialize detector
detector = JerseyNumberDetector()

# Detect jersey number for a player
jersey_number = detector.detect_jersey_number(frame, player_bbox, track_id)

if jersey_number:
    print(f"Player {track_id} has jersey number {jersey_number}")
```

### Advanced Configuration

```python
# Custom configuration
detector = JerseyNumberDetector(
    languages=['en'],           # OCR languages
    confidence_threshold=0.4,   # Minimum OCR confidence
    consensus_frames=5,         # Frames for consensus
    valid_number_range=(1, 99)  # Valid jersey numbers
)
```

## Integration with Tracking System

The detector is designed to work seamlessly with the existing player tracking system:

1. **Input**: Video frame, player bounding box, and tracking ID
2. **Processing**: Extracts jersey region, applies preprocessing, runs OCR
3. **Consensus**: Builds confidence through multi-frame detection
4. **Caching**: Stores confirmed jersey numbers for fast retrieval
5. **Output**: Returns jersey number or None if not detected

## Performance Features

- **Adaptive Region Extraction**: Focuses on chest area where numbers are typically located
- **Frame Sampling**: Can be configured to process every N frames for performance
- **GPU Acceleration**: Uses GPU when available for faster OCR processing
- **Cache Hit Rate**: Tracks performance metrics for optimization

## Error Handling

- Graceful fallback when OCR fails
- Validation of detected numbers
- Logging for debugging and monitoring
- Maintains backward compatibility with existing tracking IDs

## Dependencies

- `easyocr`: OCR engine
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `pillow`: Image handling
