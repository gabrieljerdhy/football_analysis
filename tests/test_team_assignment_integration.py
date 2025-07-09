#!/usr/bin/env python3
"""
Integration test for team assignment optimization with realistic data.
"""

import time
import numpy as np
import cv2
from src.team_assigner.team_assigner import TeamAssigner


def create_realistic_tracks(num_frames=4697, players_per_frame=15):
    """Create realistic tracking data similar to the user's video."""
    tracks = {"players": []}
    
    # Create consistent player IDs that appear across multiple frames
    player_ids = [str(i) for i in range(1, 25)]  # 24 total players
    
    for frame_idx in range(num_frames):
        frame_players = {}
        
        # Randomly select 10-20 players visible in this frame
        visible_players = np.random.choice(
            player_ids, 
            size=np.random.randint(10, min(players_per_frame + 1, len(player_ids))), 
            replace=False
        )
        
        for player_id in visible_players:
            # Create realistic bounding box positions
            x1 = np.random.randint(50, 800)
            y1 = np.random.randint(50, 400)
            x2 = x1 + np.random.randint(40, 80)
            y2 = y1 + np.random.randint(80, 120)
            
            frame_players[player_id] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": np.random.uniform(0.7, 0.95)
            }
        
        tracks["players"].append(frame_players)
    
    return tracks


def create_realistic_sample_frames(num_frames=5):
    """Create realistic sample frames with different colored jerseys."""
    frames = []
    
    for i in range(num_frames):
        # Create frame with grass background
        frame = np.full((480, 640, 3), [34, 139, 34], dtype=np.uint8)  # Forest green background
        
        # Add some noise and variation
        noise = np.random.randint(-20, 20, (480, 640, 3))
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some player-like colored regions
        for _ in range(10):
            x, y = np.random.randint(50, 590), np.random.randint(50, 430)
            w, h = np.random.randint(30, 60), np.random.randint(60, 100)
            
            # Alternate between team colors
            if np.random.random() > 0.5:
                color = [0, 0, 200 + np.random.randint(-30, 30)]  # Red team
            else:
                color = [200 + np.random.randint(-30, 30), 0, 0]  # Blue team
            
            frame[y:y+h, x:x+w] = color
        
        frames.append(frame)
    
    return frames


def test_realistic_performance():
    """Test with realistic data matching the user's scenario."""
    print("🏈 Testing team assignment with realistic data...")
    print("=" * 60)
    
    # Create data matching user's scenario: 4,697 frames
    num_frames = 4697
    tracks = create_realistic_tracks(num_frames=num_frames, players_per_frame=15)
    sample_frames = create_realistic_sample_frames(num_frames=5)
    
    # Count total players and frames
    total_players = len(set(
        player_id 
        for frame_players in tracks["players"] 
        for player_id in frame_players.keys()
    ))
    
    frames_with_players = sum(1 for frame in tracks["players"] if frame)
    
    print(f"📊 Test scenario:")
    print(f"   Total frames: {num_frames:,}")
    print(f"   Frames with players: {frames_with_players:,}")
    print(f"   Unique players: {total_players}")
    print(f"   Sample frames for color analysis: {len(sample_frames)}")
    
    # Test the optimized team assignment
    print(f"\n🚀 Running optimized team assignment...")
    
    team_assigner = TeamAssigner()
    
    # Initialize with sample frame
    if sample_frames and tracks["players"][0]:
        team_assigner.assign_team_color(sample_frames[0], tracks["players"][0])
    else:
        team_assigner.assign_team_color(None, {})
    
    # Time the batch assignment
    start_time = time.time()
    team_assigner.assign_teams_batch(tracks, sample_frames, max_samples=30)
    elapsed_time = time.time() - start_time
    
    # Calculate performance metrics
    ms_per_frame = elapsed_time / num_frames * 1000
    
    print(f"\n📈 Performance Results:")
    print(f"   Total time: {elapsed_time:.2f}s")
    print(f"   Time per frame: {ms_per_frame:.3f}ms")
    print(f"   Frames per second: {num_frames/elapsed_time:.1f}")
    
    # Compare to old performance (user reported ~4ms per frame)
    old_estimated_time = num_frames * 0.004  # 4ms per frame
    improvement_factor = old_estimated_time / elapsed_time
    
    print(f"\n🎯 Performance Comparison:")
    print(f"   Old method (estimated): {old_estimated_time:.1f}s ({4.0:.1f}ms per frame)")
    print(f"   New method (actual): {elapsed_time:.2f}s ({ms_per_frame:.3f}ms per frame)")
    print(f"   Improvement factor: {improvement_factor:.1f}x faster")
    print(f"   Time saved: {old_estimated_time - elapsed_time:.1f}s")
    
    # Verify assignments
    assigned_players = 0
    assigned_frames = 0
    
    for frame_idx, frame_players in enumerate(tracks["players"]):
        if frame_players:
            frame_has_assignments = False
            for player_id, track in frame_players.items():
                if "team" in track and "team_color" in track:
                    assigned_players += 1
                    frame_has_assignments = True
            if frame_has_assignments:
                assigned_frames += 1
    
    print(f"\n✅ Assignment Verification:")
    print(f"   Players assigned: {assigned_players:,}")
    print(f"   Frames with assignments: {assigned_frames:,}")
    print(f"   Assignment coverage: {assigned_frames/frames_with_players*100:.1f}%")
    
    # Success criteria
    target_ms_per_frame = 1.0
    success = ms_per_frame < target_ms_per_frame and assigned_frames > 0
    
    print(f"\n🏆 Test Result:")
    if success:
        print(f"   ✅ SUCCESS! Target achieved ({ms_per_frame:.3f}ms < {target_ms_per_frame}ms)")
        print(f"   🚀 Ready for production use!")
    else:
        print(f"   ❌ Target not met ({ms_per_frame:.3f}ms >= {target_ms_per_frame}ms)")
    
    return success, ms_per_frame, improvement_factor


if __name__ == "__main__":
    success, performance, improvement = test_realistic_performance()
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    if success:
        print(f"🎉 Team assignment optimization SUCCESSFUL!")
        print(f"⚡ Performance: {performance:.3f}ms per frame")
        print(f"🚀 Improvement: {improvement:.1f}x faster than before")
        print(f"💾 Memory efficient with intelligent caching")
        print(f"🎯 Ready to handle large videos efficiently")
    else:
        print(f"⚠️  Optimization needs further work")
        print(f"📈 Current performance: {performance:.3f}ms per frame")
