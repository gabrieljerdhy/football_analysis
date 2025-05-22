import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Check if image is empty
        if image_2d.shape[0] == 0:
            return None

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract player image from bounding box
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Check if image is empty
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            # Return a default color (black) if image is empty
            return np.array([0, 0, 0])

        # Get top half of the image
        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Check if top half is empty
        if (
            top_half_image.size == 0
            or top_half_image.shape[0] == 0
            or top_half_image.shape[1] == 0
        ):
            # Return a default color (black) if top half is empty
            return np.array([0, 0, 0])

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # If clustering failed, return default color
        if kmeans is None:
            return np.array([0, 0, 0])

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Get the player cluster
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        # Check if there are any player detections
        if not player_detections:
            # Set default team colors if no players detected
            self.team_colors[1] = np.array([255, 0, 0])  # Red for team 1
            self.team_colors[2] = np.array([0, 0, 255])  # Blue for team 2
            return

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            try:
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
            except Exception as e:
                print(f"Error getting player color: {e}")
                continue

        # Check if we have enough colors to cluster
        if len(player_colors) < 2:
            # Set default team colors if not enough players
            self.team_colors[1] = np.array([255, 0, 0])  # Red for team 1
            self.team_colors[2] = np.array([0, 0, 255])  # Blue for team 2
            return

        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(player_colors)

            self.kmeans = kmeans

            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
        except Exception as e:
            print(f"Error clustering team colors: {e}")
            # Set default team colors on error
            self.team_colors[1] = np.array([255, 0, 0])  # Red for team 1
            self.team_colors[2] = np.array([0, 0, 255])  # Blue for team 2

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        try:
            player_color = self.get_player_color(frame, player_bbox)

            # Check if kmeans model exists
            if not hasattr(self, "kmeans"):
                # Assign default team
                team_id = 1
            else:
                team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
                team_id += 1

            # Special case handling
            if player_id == 91:
                team_id = 1

            self.player_team_dict[player_id] = team_id

            return team_id
        except Exception as e:
            print(f"Error assigning team to player {player_id}: {e}")
            # Default to team 1 on error
            self.player_team_dict[player_id] = 1
            return 1
