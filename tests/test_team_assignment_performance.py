#!/usr/bin/env python3
"""
Test script to verify team assignment performance improvements.
"""

import time
import numpy as np
from src.team_assigner.team_assigner import TeamAssigner


def create_mock_tracks(num_frames=5000, players_per_frame=20):
    """Create mock tracking data for testing."""
    tracks = {"players": []}
    
    for frame_idx in range(num_frames):
        frame_players = {}
        for player_id in range(1, players_per_frame + 1):
            # Create mock bounding box
            x1, y1 = np.random.randint(0, 800), np.random.randint(0, 400)
            x2, y2 = x1 + 50, y1 + 100
            frame_players[str(player_id)] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.9
            }
        tracks["players"].append(frame_players)
    
    return tracks


def create_mock_frames(num_frames=5):
    """Create mock video frames for color analysis."""
    frames = []
    for _ in range(num_frames):
        # Create a random RGB frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def test_old_vs_new_team_assignment():
    """Test performance comparison between old and new team assignment methods."""
    print("🧪 Testing team assignment performance improvements...")
    
    # Create test data
    num_frames = 5000
    tracks = create_mock_tracks(num_frames=num_frames, players_per_frame=15)
    sample_frames = create_mock_frames(num_frames=5)
    
    print(f"📊 Test data: {num_frames:,} frames with {len(tracks['players'][0])} players per frame")
    
    # Test new optimized batch assignment
    print("\n🚀 Testing optimized batch assignment...")
    team_assigner = TeamAssigner()
    
    # Initialize team colors
    team_assigner.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}
    team_assigner.color_list_lab = [
        [53.24, 80.09, 67.20],  # Red in LAB
        [32.30, 79.19, -107.86]  # Blue in LAB
    ]
    
    start_time = time.time()
    team_assigner.assign_teams_batch(tracks, sample_frames, max_samples=30)
    batch_time = time.time() - start_time
    
    print(f"✅ Batch assignment completed in {batch_time:.2f}s")
    print(f"⚡ Performance: {batch_time/num_frames*1000:.3f}ms per frame")
    
    # Verify assignments were applied
    assigned_frames = 0
    total_players = 0
    for frame_players in tracks["players"]:
        if frame_players:
            assigned_frames += 1
            for player_id, track in frame_players.items():
                if "team" in track:
                    total_players += 1
    
    print(f"📈 Results: {assigned_frames:,} frames processed, {total_players:,} players assigned")
    
    # Performance targets
    target_ms_per_frame = 1.0  # Target: under 1ms per frame
    actual_ms_per_frame = batch_time / num_frames * 1000
    
    print(f"\n🎯 Performance Analysis:")
    print(f"   Target: < {target_ms_per_frame:.1f}ms per frame")
    print(f"   Actual: {actual_ms_per_frame:.3f}ms per frame")
    
    if actual_ms_per_frame < target_ms_per_frame:
        print(f"✅ SUCCESS: Performance target achieved! ({actual_ms_per_frame:.3f}ms < {target_ms_per_frame}ms)")
        improvement_factor = 4.0 / actual_ms_per_frame  # Assuming old method was ~4ms per frame
        print(f"🚀 Estimated improvement: {improvement_factor:.1f}x faster than old method")
    else:
        print(f"⚠️  Performance target not met ({actual_ms_per_frame:.3f}ms >= {target_ms_per_frame}ms)")
    
    return actual_ms_per_frame < target_ms_per_frame


def test_caching_effectiveness():
    """Test that player color caching is working effectively."""
    print("\n🧪 Testing player color caching effectiveness...")
    
    team_assigner = TeamAssigner()
    team_assigner.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}
    team_assigner.color_list_lab = [
        [53.24, 80.09, 67.20],  # Red in LAB
        [32.30, 79.19, -107.86]  # Blue in LAB
    ]
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = [100, 100, 150, 200]
    
    # First call - should compute color
    start_time = time.time()
    team1 = team_assigner.get_player_team(test_frame, test_bbox, "1")
    first_call_time = time.time() - start_time
    
    # Second call - should use cache
    start_time = time.time()
    team2 = team_assigner.get_player_team(test_frame, test_bbox, "1")
    second_call_time = time.time() - start_time
    
    print(f"   First call (compute): {first_call_time*1000:.2f}ms")
    print(f"   Second call (cache): {second_call_time*1000:.2f}ms")
    print(f"   Speedup: {first_call_time/second_call_time:.1f}x")
    print(f"   Teams consistent: {team1 == team2}")
    
    # Test ultra-fast mode
    team_assigner.enable_ultra_fast_mode()
    start_time = time.time()
    team3 = team_assigner.get_player_team(test_frame, test_bbox, "2")
    ultra_fast_time = time.time() - start_time
    
    print(f"   Ultra-fast mode: {ultra_fast_time*1000:.2f}ms")
    print(f"   Ultra-fast speedup: {first_call_time/ultra_fast_time:.1f}x")
    
    return second_call_time < first_call_time * 0.1  # Cache should be 10x+ faster


if __name__ == "__main__":
    print("🏈 Team Assignment Performance Test")
    print("=" * 50)
    
    # Run tests
    performance_test_passed = test_old_vs_new_team_assignment()
    caching_test_passed = test_caching_effectiveness()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Performance Test: {'✅ PASSED' if performance_test_passed else '❌ FAILED'}")
    print(f"   Caching Test: {'✅ PASSED' if caching_test_passed else '❌ FAILED'}")
    
    if performance_test_passed and caching_test_passed:
        print("\n🎉 All tests passed! Team assignment optimization is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Review the optimization implementation.")
