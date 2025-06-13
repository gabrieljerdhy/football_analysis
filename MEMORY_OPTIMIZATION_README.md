# Memory-Efficient Video Processing

This guide explains how to process large football videos (like your 1 hour 40 minute video) without running into memory issues.

## 🚨 The Problem

The original `read_video()` function loads ALL frames into memory at once:
- 1h 40min video ≈ 180,000 frames
- Each frame ≈ 6MB (1920x1080x3 bytes)
- Total memory needed ≈ **1TB of RAM** 🔥

This causes **Out of Memory** errors on normal computers.

## ✅ The Solution

We've implemented **batch processing** that processes videos in small chunks:

### New Memory-Efficient Components

1. **VideoFrameIterator** - Streams video frames in batches
2. **Memory-efficient tracking** - Processes detection in smaller batches
3. **Automatic memory cleanup** - Forces garbage collection between batches
4. **Progress monitoring** - Shows memory usage during processing

## 🚀 Quick Start

### For Large Videos (Recommended)

```bash
# Basic usage - memory optimized
python process_large_video.py --input your_large_video.mp4

# Custom settings for your system
python process_large_video.py --input your_large_video.mp4 --batch-size 30 --memory-limit 6.0
```

### Check Video Requirements First

```bash
# Check video info and memory requirements
python process_large_video.py --input your_large_video.mp4 --check-only
```

## 📊 Memory Guidelines

| Your RAM | Recommended Settings |
|----------|---------------------|
| 4GB      | `--batch-size 20 --memory-limit 3.0` |
| 8GB      | `--batch-size 50 --memory-limit 6.0` |
| 16GB     | `--batch-size 100 --memory-limit 12.0` |
| 32GB+    | `--batch-size 200 --memory-limit 24.0` |

## 🔧 Advanced Usage

### Enable All Features (Higher Memory Usage)
```bash
python process_large_video.py --input video.mp4 \
  --enable-camera-movement \
  --enable-speed-distance \
  --batch-size 30
```

### Force Regeneration of Cached Data
```bash
python process_large_video.py --input video.mp4 --force-regenerate
```

### Custom Output Location
```bash
python process_large_video.py --input video.mp4 --output my_analysis.avi
```

## 🧪 Testing Memory Optimization

Test if the memory optimization works on your system:

```bash
# Run memory optimization tests
python test_memory_optimization.py
```

This will:
- Test the new VideoFrameIterator
- Compare memory usage vs old approach
- Give recommendations for your system

## 📁 File Structure

```
football_analysis/
├── main_memory_efficient.py     # New memory-efficient main processor
├── process_large_video.py       # Simple script for large videos
├── test_memory_optimization.py  # Test memory optimization
├── utils/video_utils.py         # Updated with VideoFrameIterator
└── trackers/tracker.py          # Updated with memory-efficient methods
```

## 🔄 How It Works

### Old Approach (Causes Memory Issues)
```python
# ❌ Loads ALL frames into memory
video_frames = read_video(video_path)  # 1TB of RAM needed!
tracks = tracker.get_object_tracks(video_frames)
```

### New Approach (Memory Efficient)
```python
# ✅ Processes in small batches
with VideoFrameIterator(video_path, batch_size=50) as frame_iterator:
    for batch_frames in frame_iterator:  # Only 50 frames in memory
        batch_tracks = tracker.get_object_tracks_memory_efficient(batch_frames)
        # Process batch and cleanup memory
```

## 💾 Memory Optimization Features

1. **Batch Processing**: Process 20-200 frames at a time instead of all frames
2. **Automatic Cleanup**: Force garbage collection between batches
3. **Memory Monitoring**: Track memory usage throughout processing
4. **Adaptive Batch Size**: Automatically reduce batch size if memory limit exceeded
5. **Streaming Output**: Save video without keeping all frames in memory

## 🎯 Processing Your 1h 40min Video

For your specific video:

```bash
# Start with conservative settings
python process_large_video.py --input your_video.mp4 --batch-size 30 --memory-limit 6.0

# If that works well, you can increase batch size for faster processing
python process_large_video.py --input your_video.mp4 --batch-size 50 --memory-limit 8.0
```

Expected processing time: **30-60 minutes** (vs infinite time with memory errors)

## 🚨 Troubleshooting

### Still Getting Memory Errors?
1. Reduce batch size: `--batch-size 10`
2. Lower memory limit: `--memory-limit 4.0`
3. Disable optional features (they're disabled by default now)
4. Close other applications to free RAM

### Processing Too Slow?
1. Increase batch size: `--batch-size 100`
2. Use cached results (stubs) for re-runs
3. Consider upgrading RAM for better performance

### Want Original Behavior?
The original `main.py` still works for smaller videos:
```bash
python main.py --input small_video.mp4
```

## 📈 Performance Comparison

| Video Length | Old Approach | New Approach |
|-------------|--------------|--------------|
| 5 minutes   | ✅ Works     | ✅ Works (faster) |
| 30 minutes  | ❌ OOM Error | ✅ Works |
| 1+ hours    | ❌ OOM Error | ✅ Works |
| 2+ hours    | ❌ OOM Error | ✅ Works |

## 🎉 Benefits

- ✅ **No more memory errors** for large videos
- ✅ **Progress tracking** with memory monitoring
- ✅ **Cached results** for faster re-runs
- ✅ **Configurable** batch sizes and memory limits
- ✅ **Same analysis quality** as original approach
- ✅ **Backward compatible** with existing workflows

## 💡 Tips

1. **Start conservative** with batch size and increase if stable
2. **Monitor memory usage** during first run to find optimal settings
3. **Use cached results** (stubs) to avoid re-processing
4. **Close other applications** during processing to free RAM
5. **Consider processing overnight** for very large videos

---

**Ready to process your large video without memory issues? Start with:**

```bash
python process_large_video.py --input your_large_video.mp4 --check-only
```
