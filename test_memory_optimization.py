#!/usr/bin/env python3
"""
Test script to verify memory optimization is working correctly.
"""

import os
import sys
from pathlib import Path

from utils.video_utils import VideoFrameIterator, get_video_info, monitor_memory_usage, cleanup_memory


def test_video_iterator(video_path: str, batch_size: int = 50):
    """Test the VideoFrameIterator for memory efficiency."""
    
    print(f"🧪 Testing VideoFrameIterator with batch size {batch_size}")
    print(f"📹 Video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Get video info first
        video_info = get_video_info(video_path)
        print(f"📊 Video info: {video_info['total_frames']} frames, {video_info['duration_seconds']/60:.1f} min")
        
        initial_memory = monitor_memory_usage()
        print(f"💾 Initial memory: {initial_memory:.2f} GB")
        
        frames_processed = 0
        max_memory = initial_memory
        
        with VideoFrameIterator(video_path, batch_size) as frame_iterator:
            for batch_num, batch_frames in enumerate(frame_iterator):
                frames_processed += len(batch_frames)
                current_memory = monitor_memory_usage()
                max_memory = max(max_memory, current_memory)
                
                print(f"  Batch {batch_num + 1}: {len(batch_frames)} frames, Memory: {current_memory:.2f} GB")
                
                # Force cleanup
                cleanup_memory()
                
                # Stop after a few batches for testing
                if batch_num >= 5:
                    print(f"  ... (stopping test after {batch_num + 1} batches)")
                    break
        
        final_memory = monitor_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"✅ Test completed:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Initial memory: {initial_memory:.2f} GB")
        print(f"   Peak memory: {max_memory:.2f} GB")
        print(f"   Final memory: {final_memory:.2f} GB")
        print(f"   Memory increase: {memory_increase:.2f} GB")
        
        # Check if memory usage is reasonable
        if memory_increase < 2.0:  # Less than 2GB increase is good
            print(f"🎉 Memory usage looks good! (increase: {memory_increase:.2f} GB)")
            return True
        else:
            print(f"⚠️  High memory usage detected (increase: {memory_increase:.2f} GB)")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_memory_approaches(video_path: str):
    """Compare memory usage between old and new approaches."""
    
    print(f"\n🔬 MEMORY COMPARISON TEST")
    print("=" * 50)
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Test 1: VideoFrameIterator (new approach)
    print(f"\n1️⃣  Testing NEW approach (VideoFrameIterator):")
    success_new = test_video_iterator(video_path, batch_size=30)
    
    # Test 2: Simulate old approach (just check video info)
    print(f"\n2️⃣  OLD approach would load ALL frames into memory:")
    try:
        video_info = get_video_info(video_path)
        frame_size_mb = (video_info['width'] * video_info['height'] * 3) / (1024**2)
        total_memory_gb = (frame_size_mb * video_info['total_frames']) / 1024
        
        print(f"   Would need: {total_memory_gb:.1f} GB of RAM")
        print(f"   Current available: {monitor_memory_usage():.2f} GB")
        
        if total_memory_gb > 16:
            print(f"   🔥 This would cause OUT OF MEMORY error!")
        elif total_memory_gb > 8:
            print(f"   ⚠️  This would use most of your RAM")
        else:
            print(f"   ✅ This might fit in memory")
            
    except Exception as e:
        print(f"   ❌ Error calculating memory requirements: {e}")
    
    print(f"\n📋 SUMMARY:")
    if success_new:
        print(f"   ✅ NEW approach: Memory efficient, can handle large videos")
    else:
        print(f"   ⚠️  NEW approach: Some memory concerns detected")
    print(f"   ❌ OLD approach: Would likely cause out of memory errors")


def main():
    print("🧪 MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    # Look for test videos
    test_videos = []
    
    # Check common video locations
    video_dirs = ["input_videos", ".", "test_videos"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    test_videos.append(os.path.join(video_dir, file))
    
    if not test_videos:
        print("❌ No test videos found!")
        print("   Please put a video file in one of these locations:")
        for video_dir in video_dirs:
            print(f"   - {video_dir}/")
        print("   Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    print(f"📹 Found {len(test_videos)} test video(s):")
    for i, video in enumerate(test_videos):
        print(f"   {i+1}. {video}")
    
    # Test with the first video
    test_video = test_videos[0]
    print(f"\n🎯 Testing with: {test_video}")
    
    # Run comparison test
    compare_memory_approaches(test_video)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Use process_large_video.py for processing large videos")
    print(f"   • Start with batch-size 50 and adjust based on your RAM")
    print(f"   • Monitor memory usage during processing")
    print(f"   • Enable camera movement and speed estimation only if you have enough RAM")


if __name__ == "__main__":
    main()
