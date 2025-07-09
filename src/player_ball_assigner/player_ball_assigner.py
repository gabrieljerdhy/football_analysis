import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.append("../")
from src.utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        """Original single-frame ball assignment method."""
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_to_players_batch(
        self, tracks: Dict, batch_size: int = 100
    ) -> List[int]:
        """
        Optimized batch processing for ball assignment across multiple frames.

        Args:
            tracks: Dictionary containing player and ball tracking data
            batch_size: Number of frames to process in each batch

        Returns:
            List of assigned player IDs for each frame (-1 if no assignment)
        """
        total_frames = len(tracks.get("players", []))
        ball_assignments = []

        print(f"🚀 Processing ball assignments in batches of {batch_size} frames...")

        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_assignments = self._process_batch(tracks, batch_start, batch_end)
            ball_assignments.extend(batch_assignments)

            # Progress reporting
            if batch_start % (batch_size * 10) == 0:
                progress = (batch_end / total_frames) * 100
                print(
                    f"  Ball assignment progress: {progress:.1f}% ({batch_end}/{total_frames} frames)"
                )

        return ball_assignments

    def _process_batch(
        self, tracks: Dict, start_frame: int, end_frame: int
    ) -> List[int]:
        """Process a batch of frames for ball assignment."""
        batch_assignments = []

        for frame_num in range(start_frame, end_frame):
            # Skip if frame data is missing
            if frame_num >= len(tracks.get("players", [])) or frame_num >= len(
                tracks.get("ball", [])
            ):
                batch_assignments.append(-1)
                continue

            player_track = tracks["players"][frame_num]
            ball_data = tracks["ball"][frame_num].get(1, {})
            ball_bbox = ball_data.get("bbox", [])

            # Only assign ball if we have a valid bbox
            if ball_bbox and len(ball_bbox) == 4:
                assigned_player = self.assign_ball_to_player(player_track, ball_bbox)
            else:
                assigned_player = -1

            batch_assignments.append(assigned_player)

        return batch_assignments

    def assign_ball_with_caching(
        self, tracks: Dict, cache_interval: int = 5
    ) -> List[int]:
        """
        Ultra-fast ball assignment with intelligent caching and sampling.

        Args:
            tracks: Dictionary containing player and ball tracking data
            cache_interval: Process every Nth frame, interpolate others

        Returns:
            List of assigned player IDs for each frame
        """
        total_frames = len(tracks.get("players", []))
        ball_assignments = [-1] * total_frames

        print(f"⚡ Ultra-fast ball assignment with {cache_interval}x sampling...")

        # Process every cache_interval frames
        processed_frames = 0
        for frame_num in range(0, total_frames, cache_interval):
            if frame_num < len(tracks.get("players", [])) and frame_num < len(
                tracks.get("ball", [])
            ):

                player_track = tracks["players"][frame_num]
                ball_data = tracks["ball"][frame_num].get(1, {})
                ball_bbox = ball_data.get("bbox", [])

                if ball_bbox and len(ball_bbox) == 4:
                    assigned_player = self.assign_ball_to_player(
                        player_track, ball_bbox
                    )
                    ball_assignments[frame_num] = assigned_player
                    processed_frames += 1

        # Interpolate assignments for skipped frames
        self._interpolate_assignments(ball_assignments, cache_interval)

        print(
            f"✅ Processed {processed_frames} key frames, interpolated {total_frames - processed_frames}"
        )
        return ball_assignments

    def _interpolate_assignments(self, assignments: List[int], interval: int):
        """Fill in missing assignments using forward fill strategy."""
        last_valid_assignment = -1

        for i in range(len(assignments)):
            if assignments[i] != -1:
                last_valid_assignment = assignments[i]
            elif last_valid_assignment != -1:
                # Forward fill with last valid assignment
                assignments[i] = last_valid_assignment
