from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self):
        # Dictionary for the team colors
        self.team_colors = {}

        # KMeans clustering model
        self.kmeans = None

        # Dictionary to store the team of each player
        self.players_team = {}

    # Function to get the clustering model
    def get_clustering_model(self, image):
        # Reshape the image to a 2D array
        image_2d = image.reshape(-1, 3)

        # Perform KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(image_2d)

        return kmeans

    # Function to get the color of the player given the bounding box
    def get_color(self, frame, bbox):
        # Get the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # Take the top half of the bounding box
        y2 = y1 + (y2 - y1) // 2

        # Get the region of interest
        roi = frame[int(y1):int(y2), int(x1):int(x2)]

        # Get the clustering model
        kmeans = self.get_clustering_model(roi)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the shape of the image
        clustered_image = labels.reshape(roi.shape[0], roi.shape[1])

        # Get the corners cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]

        # Get the non-player cluster
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Get the player cluster
        player_cluster = 1 if non_player_cluster == 0 else 0

        # Get the color of the player
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    # Function to assign the team colors
    def assign_team(self, frame, player_detections):

        player_colors = []

        for _, player_detection in player_detections.items():
            # Get the bounding box of the player
            bbox = player_detection['bbox']
            
            # Get the color of the player
            player_color = self.get_color(frame, bbox)

            # Add the color to the list of player colors
            player_colors.append(player_color)

        # Perform KMeans clustering with 2 clusters
        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(player_colors)

        # Assign the team colors
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    # Function to get the team of the player
    def get_player_team(self, frame, player_bbox, player_id):
        # If the player is already assigned a team, return the team
        if player_id in self.players_team:
            return self.players_team[player_id]
        
        # Get the color of the player
        player_color = self.get_color(frame, player_bbox)

        # Get the team of the player
        player_team = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Assign the team to the player
        self.players_team[player_id] = player_team

        return player_team