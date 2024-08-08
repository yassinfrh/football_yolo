from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from ball_assigner import BallAssigner
import numpy as np

video_name = 'test (33)'

def main():
    # Read the input video
    input_frames = read_video('input_videos/' + video_name + '.mp4')

    # Initialize the tracker
    tracker = Tracker('models/yolov8x_best.pt')

    # Track the objects in the video
    tracks = tracker.get_tracks(input_frames, read_from_file=True, file_path='track_files/' + video_name + '.pkl')

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # Initialize the team assigner
    team_assigner = TeamAssigner()

    # Assign the team colors
    team_assigner.assign_team(input_frames[0], tracks['players'][0])

    # Loop over the frames to get the team of each player
    for i_frame, players in enumerate(tracks['players']):
        for player_id, player_info in players.items():
            # Get the bounding box of the player
            bbox = player_info['bbox']
            # Get the team of the player
            team = team_assigner.get_player_team(input_frames[i_frame], bbox, player_id)
            # Add the team to the player info
            player_info['team'] = team
            # Add the team color to the player info
            player_info['team_color'] = team_assigner.team_colors[team]

    # Initialize the ball assigner
    ball_assigner = BallAssigner()

    # Initialize the ball posession array
    ball_posession = []

    # Loop over the frames to assign the ball to the players
    for i_frame, players in enumerate(tracks['players']):
        # Get the ball bounding box
        ball_bbox = tracks['ball'][i_frame][1]['bbox']
        # Assign the ball to the players
        assigned_player = ball_assigner.assign_ball(players, ball_bbox)

        # If the assigned player is not None, add the ball to the player info and append the team to the ball posession array
        if assigned_player is not None:
            players[assigned_player]['has_ball'] = True
            ball_posession.append(players[assigned_player]['team'])
        # If the assigned player is None, append the last team to the ball posession array, if the array is not empty
        elif len(ball_posession) > 0:
            ball_posession.append(ball_posession[-1])
        # If the assigned player is None and the array is empty, append -1 to the ball posession array
        else:
            ball_posession.append(-1)

    # Convert to numpy array
    ball_posession = np.array(ball_posession)

    # Draw the indicators on the video
    output_frames = tracker.draw_annotations(input_frames, tracks, ball_posession)

    # Save the input video
    save_video(output_frames, 'output_videos/' + video_name + '.avi')

if __name__ == '__main__':
    main()