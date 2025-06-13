# Enhanced Main.py Usage Guide

The `main.py` script has been enhanced with built-in memory-efficient processing for large videos. You no longer need separate scripts!

## Basic Usage

### For Small Videos (< 30 minutes)
```bash
# Standard processing (loads all frames into memory)
python main.py --input your_video.mp4

# With all features enabled
python main.py --input your_video.mp4 --enable-camera-movement --enable-speed-distance
```

### For Large Videos (> 30 minutes or > 10GB)
```bash
# Memory-efficient processing (recommended for large videos)
python main.py --input large_video.mp4 --memory-efficient

# With custom batch size and memory limit
python main.py --input large_video.mp4 --memory-efficient --batch-size 30 --memory-limit 6.0

# Enable all features with memory optimization
python main.py --input large_video.mp4 --memory-efficient --enable-camera-movement --enable-speed-distance
```

## New Features

### 1. Video Information Check
Check video requirements before processing:
```bash
python main.py --input your_video.mp4 --check-video-info
```

This will show:
- Video duration, resolution, frame count
- Memory requirements
- Recommended settings for your video

### 2. Memory-Efficient Processing
Automatically processes videos in batches to avoid memory issues:
```bash
python main.py --input large_video.mp4 --memory-efficient
```

### 3. Automatic Batch Size Adjustment
The script automatically adjusts batch size based on your memory limit:
```bash
python main.py --input video.mp4 --memory-efficient --memory-limit 4.0
```

### 4. Progress Monitoring
Real-time progress updates with memory usage monitoring during processing.

## Command Line Options

### Core Options
- `--input, -i`: Input video file (required)
- `--output, -o`: Output video file (optional, auto-generated if not provided)
- `--memory-efficient`: Enable memory-efficient processing for large videos

### Memory Options
- `--batch-size`: Number of frames to process at once (default: 50)
- `--memory-limit`: Memory limit in GB for automatic adjustment (default: 8.0)
- `--check-video-info`: Only check video info without processing

### Feature Options
- `--enable-camera-movement`: Enable camera movement estimation
- `--enable-speed-distance`: Enable speed and distance estimation
- `--goals-config`: Path to manual goals configuration file

### Processing Options
- `--no-stubs`: Don't use cached processing results
- `--force-regenerate`: Force regeneration of all cached data

## Memory Guidelines

### For Different System Configurations:
- **4GB RAM**: `--batch-size 20 --memory-limit 3.0`
- **8GB RAM**: `--batch-size 50 --memory-limit 6.0` (default)
- **16GB RAM**: `--batch-size 100 --memory-limit 12.0`
- **32GB+ RAM**: `--batch-size 200 --memory-limit 24.0`

## Examples

### Example 1: Quick Analysis
```bash
# Check what the video needs
python main.py --input match.mp4 --check-video-info

# Process with recommended settings
python main.py --input match.mp4 --memory-efficient
```

### Example 2: Full Analysis with All Features
```bash
python main.py --input match.mp4 --memory-efficient \
  --enable-camera-movement --enable-speed-distance \
  --batch-size 30 --memory-limit 6.0
```

### Example 3: Large Video Processing
```bash
# For a 2-hour match video
python main.py --input full_match.mp4 --memory-efficient \
  --batch-size 25 --memory-limit 4.0
```

### Example 4: With Manual Goals
```bash
# Create goals template first
python main.py --create-goals-template goals.csv

# Edit goals.csv with your manual goals, then:
python main.py --input match.mp4 --memory-efficient --goals-config goals.csv
```

## Output Files

The script generates:
1. **Output video**: `output_videos/{video_name}_output.avi`
2. **Team statistics**: `output/{video_name}_team_stats.csv`
3. **Player statistics**: `output/{video_name}_player_stats.csv`
4. **Cached data**: `stubs/{video_name}_tracks.pkl` (for faster re-runs)

## Tips

1. **Always check video info first** for large videos
2. **Use memory-efficient mode** for videos > 30 minutes
3. **Adjust batch size** based on your available RAM
4. **Cached data** in `stubs/` directory speeds up re-runs
5. **Use `--force-regenerate`** if you want fresh analysis

## Troubleshooting

### Out of Memory Errors
- Reduce `--batch-size` (try 20, 10, or even 5)
- Reduce `--memory-limit`
- Disable `--enable-camera-movement` and `--enable-speed-distance`

### Slow Processing
- Increase `--batch-size` if you have more RAM
- Use cached data (don't use `--force-regenerate`)
- Process without camera movement and speed estimation first

### Video Quality Issues
- Check input video format and codec
- Ensure sufficient disk space for output
- Try different output formats if needed
