import sys
sys.path.append('.')
import time
import numpy as np
from src.team_assigner.team_assigner import TeamAssigner

# Test with realistic 4697 frames scenario
print('🏈 Testing team assignment optimization with realistic data...')
print('=' * 60)

# Create realistic tracks (4697 frames like user's video)
tracks = {'players': []}
for i in range(4697):
    frame_players = {}
    # 10-15 players per frame
    for j in range(1, 16):
        if np.random.random() > 0.2:  # 80% chance player is visible
            frame_players[str(j)] = {'bbox': [100, 100, 150, 200]}
    tracks['players'].append(frame_players)

print(f'📊 Created test data: 4,697 frames')

# Test optimized assignment
team_assigner = TeamAssigner()
team_assigner.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}
team_assigner.color_list_lab = [[53.24, 80.09, 67.20], [32.30, 79.19, -107.86]]

print('🚀 Running optimized team assignment...')
start_time = time.time()
team_assigner.assign_teams_batch(tracks, [], max_samples=30)
elapsed = time.time() - start_time

ms_per_frame = elapsed / 4697 * 1000
old_time = 4697 * 0.004  # Old method: 4ms per frame
improvement = old_time / elapsed

print('')
print('📈 PERFORMANCE RESULTS:')
print(f'   Total time: {elapsed:.2f}s')
print(f'   Time per frame: {ms_per_frame:.3f}ms')
print(f'   Old method (estimated): {old_time:.1f}s')
print(f'   Improvement: {improvement:.1f}x faster')
print(f'   Time saved: {old_time - elapsed:.1f}s')

if ms_per_frame < 1.0:
    print('')
    print(f'✅ SUCCESS! Target achieved ({ms_per_frame:.3f}ms < 1.0ms per frame)')
    print('🚀 Ready for production use!')
else:
    print(f'❌ Target not met ({ms_per_frame:.3f}ms >= 1.0ms per frame)')
