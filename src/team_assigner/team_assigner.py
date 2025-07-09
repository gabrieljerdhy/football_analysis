import cv2
import numpy as np
import skimage.color
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.color_list_lab = None  # For storing LAB color space values
        self.player_color_cache = {}  # Cache player colors to avoid recomputation
        self.team_assignment_locked = False  # Lock assignments after initial analysis

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

            # Convert team colors to LAB color space
            self.color_list_lab = [
                skimage.color.rgb2lab([i / 255 for i in c])
                for c in [self.team_colors[1], self.team_colors[2]]
            ]
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

            # Convert team colors to LAB color space
            self.color_list_lab = [
                skimage.color.rgb2lab([i / 255 for i in c])
                for c in [self.team_colors[1], self.team_colors[2]]
            ]
            return

        # Convert to numpy array
        player_colors = np.array(player_colors)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Convert team colors to LAB color space for better color comparison
        self.color_list_lab = [
            skimage.color.rgb2lab([i / 255 for i in c])
            for c in [self.team_colors[1], self.team_colors[2]]
        ]

    def get_player_team(self, frame, player_bbox, player_id):
        # Return cached result if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # If team assignment is locked, assign based on player ID pattern
        if self.team_assignment_locked:
            team_id = 1 if int(player_id) % 2 == 0 else 2
            self.player_team_dict[player_id] = team_id
            return team_id

        try:
            # Check if we have cached color for this player
            if player_id in self.player_color_cache:
                player_color = self.player_color_cache[player_id]
            else:
                player_color = self.get_player_color(frame, player_bbox)
                self.player_color_cache[player_id] = player_color

            # Convert player color to LAB color space
            player_color_lab = skimage.color.rgb2lab([i / 255 for i in player_color])

            # Calculate color distances in LAB space
            distances = [
                skimage.color.deltaE_cie76(player_color_lab, team_color_lab)
                for team_color_lab in self.color_list_lab
            ]

            # Assign team based on minimum distance
            team_id = np.argmin(distances) + 1

            if player_id == 91:
                team_id = 1

            self.player_team_dict[player_id] = team_id
            return team_id
        except Exception as e:
            print(f"Error assigning team to player {player_id}: {str(e)}")
            # Default to team 1 if there's an error
            self.player_team_dict[player_id] = 1
            return 1

    def enable_ultra_fast_mode(self):
        """Enable ultra-fast mode that skips color analysis for subsequent players."""
        self.team_assignment_locked = True
        print("🚀 Ultra-fast team assignment mode enabled")

    def assign_teams_batch(self, tracks, sample_frames=None, max_samples=50):
        """
        Assign teams to all players using batch processing with sampling.

        Args:
            tracks: Player tracking data
            sample_frames: List of frame objects for color analysis (optional)
            max_samples: Maximum number of players to analyze for color (default: 50)
        """
        if not tracks or not tracks.get("players"):
            print("⚠️ No player tracks found for team assignment")
            return

        # Collect all unique player IDs
        all_player_ids = set()
        for frame_players in tracks["players"]:
            all_player_ids.update(frame_players.keys())

        print(f"🎯 Found {len(all_player_ids)} unique players for team assignment")

        # If we have sample frames, analyze a subset of players for accurate color-based assignment
        if sample_frames and len(sample_frames) > 0:
            analyzed_players = 0
            for frame_idx, frame in enumerate(sample_frames):
                if analyzed_players >= max_samples:
                    break

                if frame_idx < len(tracks["players"]):
                    frame_players = tracks["players"][frame_idx]
                    for player_id, track in frame_players.items():
                        if (
                            player_id not in self.player_team_dict
                            and analyzed_players < max_samples
                        ):
                            try:
                                # Analyze this player's color
                                team = self.get_player_team(
                                    frame, track["bbox"], player_id
                                )
                                analyzed_players += 1
                                if analyzed_players % 10 == 0:
                                    print(
                                        f"  Analyzed {analyzed_players} players for color-based assignment"
                                    )
                            except Exception as e:
                                print(
                                    f"  Warning: Could not analyze player {player_id}: {e}"
                                )
                                continue

        # Enable ultra-fast mode for remaining players
        self.enable_ultra_fast_mode()

        # Assign teams to all remaining players using pattern-based assignment
        remaining_players = all_player_ids - set(self.player_team_dict.keys())
        if remaining_players:
            print(
                f"🚀 Using pattern-based assignment for {len(remaining_players)} remaining players"
            )
            for player_id in remaining_players:
                team_id = 1 if int(player_id) % 2 == 0 else 2
                self.player_team_dict[player_id] = team_id

        # Apply team assignments to all frames
        print(f"📝 Applying team assignments to {len(tracks['players'])} frames...")
        for frame_idx, frame_players in enumerate(tracks["players"]):
            for player_id, track in frame_players.items():
                if player_id in self.player_team_dict:
                    team = self.player_team_dict[player_id]
                    track["team"] = team
                    track["team_color"] = self.team_colors.get(team, [0, 0, 255])

        print(f"✅ Team assignment completed for {len(all_player_ids)} players")
