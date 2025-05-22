import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append("../")
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Check if features were found in the first frame
        if old_features is None or len(old_features) == 0:
            print(
                "WARNING: No features found in the first frame for camera movement estimation"
            )
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            try:
                # Calculate optical flow
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features, None, **self.lk_params
                )

                # Check if features were successfully tracked
                if new_features is None or len(new_features) == 0:
                    print(f"WARNING: Failed to track features in frame {frame_num}")
                    # Try to find new features in this frame
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    if old_features is None or len(old_features) == 0:
                        print(f"WARNING: No features found in frame {frame_num}")
                        old_gray = frame_gray.copy()
                        continue
                    old_gray = frame_gray.copy()
                    continue

                # Filter out points where the status is 0 (not found)
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                # Check if we have any good points
                if len(good_new) == 0 or len(good_old) == 0:
                    print(f"WARNING: No good features to track in frame {frame_num}")
                    # Try to find new features in this frame
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    if old_features is None or len(old_features) == 0:
                        print(f"WARNING: No features found in frame {frame_num}")
                    old_gray = frame_gray.copy()
                    continue

                max_distance = 0
                camera_movement_x, camera_movement_y = 0, 0

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = measure_distance(new_features_point, old_features_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(
                            old_features_point, new_features_point
                        )

                if max_distance > self.minimum_distance:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                    # Find new features for the next frame
                    new_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    if new_features is not None and len(new_features) > 0:
                        old_features = new_features
                    # If no new features found, keep using the current ones

                old_gray = frame_gray.copy()

            except Exception as e:
                print(f"ERROR in camera movement estimation for frame {frame_num}: {e}")
                # Try to find new features in this frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if old_features is None or len(old_features) == 0:
                    print(f"WARNING: No features found in frame {frame_num}")
                old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        # Ensure camera_movement_per_frame has enough elements
        if len(camera_movement_per_frame) < len(frames):
            # Extend camera_movement_per_frame if needed
            camera_movement_per_frame = camera_movement_per_frame + [[0, 0]] * (
                len(frames) - len(camera_movement_per_frame)
            )

        for frame_num, frame in enumerate(frames):
            try:
                # Process frames in a memory-efficient way
                # Instead of copying the entire frame, modify it directly
                # Create a small overlay for the text area only
                h, w = frame.shape[:2]
                overlay_height = min(100, h)
                overlay_width = min(500, w)

                # Create a small white rectangle for the overlay
                overlay = (
                    np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 255
                )

                # Blend the overlay with the frame
                alpha = 0.6
                frame[:overlay_height, :overlay_width] = cv2.addWeighted(
                    overlay, alpha, frame[:overlay_height, :overlay_width], 1 - alpha, 0
                )

                # Get camera movement for this frame
                if frame_num < len(camera_movement_per_frame):
                    x_movement, y_movement = camera_movement_per_frame[frame_num]
                else:
                    x_movement, y_movement = 0, 0

                # Add text
                cv2.putText(
                    frame,
                    f"Camera Movement X: {x_movement:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    frame,
                    f"Camera Movement Y: {y_movement:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )

                output_frames.append(frame)

            except Exception as e:
                print(f"Error drawing camera movement for frame {frame_num}: {e}")
                # Just add the original frame if there's an error
                output_frames.append(frame)

        return output_frames
