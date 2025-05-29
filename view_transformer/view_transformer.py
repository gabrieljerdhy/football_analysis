import cv2
import numpy as np
from sklearn.metrics import mean_squared_error


class ViewTransformer:
    def __init__(self):
        self.homography_matrix = None
        self.field_keypoints = {}  # Map of keypoint labels to positions on the field
        self.detected_keypoints_prev = []
        self.detected_keypoints_src_pts_prev = []
        self.keypoints_displacement_mean_tol = 10  # Tolerance in pixels

    def set_field_keypoints(self, field_keypoints):
        """
        Set the reference field keypoints

        Args:
            field_keypoints: Dictionary mapping keypoint labels to (x,y) positions
        """
        self.field_keypoints = field_keypoints

    def calculate_homography(self, detected_keypoints, force_update=False):
        """
        Calculate homography matrix from detected keypoints to field keypoints

        Args:
            detected_keypoints: List of (label, x, y) tuples for detected keypoints
            force_update: Force update of homography matrix regardless of displacement

        Returns:
            Boolean indicating if homography was updated
        """
        if len(detected_keypoints) < 4:
            return False

        # Extract keypoint labels and coordinates
        detected_labels = [kp[0] for kp in detected_keypoints]
        detected_src_pts = np.array([[kp[1], kp[2]] for kp in detected_keypoints])

        # Get corresponding field keypoints
        detected_dst_pts = np.array(
            [self.field_keypoints[label] for label in detected_labels]
        )

        update_homography = force_update

        # Check if we need to update homography based on keypoint displacement
        if (
            not force_update
            and self.homography_matrix is not None
            and len(self.detected_keypoints_prev) > 0
        ):
            # Find common keypoints between previous and current frame
            common_labels = set(self.detected_keypoints_prev) & set(detected_labels)

            if len(common_labels) > 3:
                # Get indices of common keypoints
                common_idx_prev = [
                    self.detected_keypoints_prev.index(label) for label in common_labels
                ]
                common_idx_curr = [
                    detected_labels.index(label) for label in common_labels
                ]

                # Get coordinates of common keypoints
                common_pts_prev = self.detected_keypoints_src_pts_prev[common_idx_prev]
                common_pts_curr = detected_src_pts[common_idx_curr]

                # Calculate mean squared error between previous and current keypoints
                mse = mean_squared_error(common_pts_prev, common_pts_curr)

                # Update homography if displacement exceeds tolerance
                update_homography = mse > self.keypoints_displacement_mean_tol

        if update_homography:
            # Calculate homography matrix
            self.homography_matrix, _ = cv2.findHomography(
                detected_src_pts, detected_dst_pts, cv2.RANSAC, 5.0
            )

            # Save current keypoints for next frame
            self.detected_keypoints_prev = detected_labels
            self.detected_keypoints_src_pts_prev = detected_src_pts

            return True

        return False

    def transform_point(self, point):
        """
        Transform a point using the current homography matrix

        Args:
            point: (x, y) coordinates to transform

        Returns:
            Transformed (x, y) coordinates or None if homography is not available
        """
        if self.homography_matrix is None:
            return None

        # Convert to homogeneous coordinates
        pt = np.array([point[0], point[1], 1.0])

        # Apply homography transformation
        transformed = np.matmul(self.homography_matrix, pt)

        # Convert back from homogeneous coordinates
        transformed = transformed / transformed[2]

        return (transformed[0], transformed[1])

    def add_transformed_position_to_tracks(self, tracks):
        """
        Add transformed positions to object tracks

        Args:
            tracks: Dictionary of object tracks
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    if "position" in track_info:
                        position = track_info["position"]
                        transformed_position = self.transform_point(position)
                        tracks[object_type][frame_num][track_id][
                            "position_transformed"
                        ] = transformed_position
