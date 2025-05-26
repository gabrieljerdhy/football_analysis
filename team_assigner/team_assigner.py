import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
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
            print(
                "Warning: No players detected in the frame. Using default team colors."
            )
            # Set default team colors
            self.team_colors[1] = np.array([0, 0, 255])  # Red for team 1
            self.team_colors[2] = np.array([255, 0, 0])  # Blue for team 2

            # Create a dummy kmeans model with predefined centers
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
            self.kmeans.cluster_centers_ = np.array(
                [[0, 0, 255], [255, 0, 0]]  # Red  # Blue
            )
            self.kmeans.labels_ = np.array([0, 1])
            return

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Check if we have enough player colors for clustering
        if len(player_colors) < 2:
            print(
                "Warning: Not enough players detected for team assignment. Using default team colors."
            )
            # Set default team colors
            self.team_colors[1] = np.array([0, 0, 255])  # Red for team 1
            self.team_colors[2] = np.array([255, 0, 0])  # Blue for team 2

            # Create a dummy kmeans model with predefined centers
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
            self.kmeans.cluster_centers_ = np.array(
                [[0, 0, 255], [255, 0, 0]]  # Red  # Blue
            )
            self.kmeans.labels_ = np.array([0, 1])
            return

        # Convert to numpy array
        player_colors = np.array(player_colors)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        try:
            player_color = self.get_player_color(frame, player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1

            if player_id == 91:
                team_id = 1

            self.player_team_dict[player_id] = team_id
            return team_id
        except Exception as e:
            print(f"Error assigning team to player {player_id}: {str(e)}")
            # Default to team 1 if there's an error
            self.player_team_dict[player_id] = 1
            return 1
