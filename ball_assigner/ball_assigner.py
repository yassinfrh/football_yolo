import sys
sys.path.append('../')
from utils import get_center_bbox, get_distance


class BallAssigner():
    def __init__(self):
        # Maximum distance for the ball assignment
        self.max_distance = 70

    # Function to assign the ball to the players
    def assign_ball(self, players, ball_bbox):
        # Get the center of the ball
        ball_pos = get_center_bbox(ball_bbox)

        # Minimum distance and player id for the ball assignment
        min_distance = 999999
        assigned_player = None

        # Loop over the players to assign the ball
        for player_id, player_info in players.items():
            # Get the bounding box of the player
            player_bbox = player_info['bbox']

            # Measure the distance between the player bottom left corner and the ball center and bottom right corner and the ball center
            bottom_left = (player_bbox[0], player_bbox[3])
            bottom_right = (player_bbox[2], player_bbox[3])
            distance = min(get_distance(bottom_left, ball_pos), get_distance(bottom_right, ball_pos))

            # Assign the ball to the player if the distance is less than the minimum distance
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player