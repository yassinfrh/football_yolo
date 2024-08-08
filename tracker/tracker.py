from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
# Add the path for the previous directory to the system path
sys.path.append('../')
from utils import get_center_bbox, get_width_bbox


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # Function to interpolate the missing detections of the ball
    def interpolate_ball(self, ball_track):
        # Get the positions of the ball
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_track]

        # Convert to pandas DataFrame
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate the missing values
        ball_positions_df = ball_positions_df.interpolate()

        # Backfill the missing values for the first few frames if missing
        ball_positions_df = ball_positions_df.bfill()

        # Convert the DataFrame back to the original format
        ball_positions = [{1: {"bbox": x}} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions

    # Function for detecting objects in batches of frames
    def detect_objects(self, frames):
        # Batch size
        batch_size = 20
        # List to store the detected objects
        detections = []
        # Iterate over the frames in batches
        for i in range(0, len(frames), batch_size):
            # Perform detection on the batch of frames
            results = self.model.predict(frames[i:i+batch_size], device='0', conf=0.1)
            # Append the detected objects to the `detections` list
            detections += results
        
        return detections

    # Function to get the tracks of the players, referees, and ball
    def get_tracks(self, frames, read_from_file=False, file_path=None):

        # Check if the tracks are read from a file
        if read_from_file and file_path is not None and os.path.exists(file_path):
            # Load the tracks from the file
            with open(file_path, 'rb') as file:
                tracks = pickle.load(file)
            return tracks


        # Perform object detection
        detections = self.detect_objects(frames)

        # Dictionary to store the tracks
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Iterate over the detection frames to perform data processing
        for i_frame, detections_frame in enumerate(detections):

            # Get the class names of the detection
            class_names = detections_frame.names
            # Invert the key and value in the class names dictionary
            class_names_inv = {v: k for k, v in class_names.items()}

            # Convert the detections to Supervision format
            detections_sv = sv.Detections.from_ultralytics(detections_frame)

            # Convert goalkeeper to player
            for i_object, class_id in enumerate(detections_sv.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detections_sv.class_id[i_object] = class_names_inv['player']

            # Perform tracking
            detections_with_track = self.tracker.update_with_detections(detections_sv)


            # Append empty dictionaries to the tracks dictionary
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # Iterate over the detection with tracks to get the bounding boxes of players and referees
            for detection in detections_with_track:
                # Get the bounding box of the detection
                bbox = detection[0].tolist()
                # Get the class id of the detection
                class_id = detection[3]
                # Get the track id of the detection
                track_id = detection[4]

                # Add the bounding box to the tracks for players and referees
                if class_id == class_names_inv['player']:
                    tracks['players'][i_frame][track_id] = {"bbox": bbox}

                if class_id == class_names_inv['referee']:
                    tracks['referees'][i_frame][track_id] = {"bbox": bbox}

            # Iterate over the detection without tracks to get the bounding box of the ball
            for detection in detections_sv:
                # Get the bounding box of the detection
                bbox = detection[0].tolist()
                # Get the class id of the detection
                class_id = detection[3]

                # Add the bounding box to the tracks for the ball
                if class_id == class_names_inv['ball']:
                    tracks['ball'][i_frame][1] = {"bbox": bbox}

        # Save the tracks to a file if the file path is provided
        if file_path is not None:
            with open(file_path, 'wb') as file:
                pickle.dump(tracks, file)

        return tracks
    
    # Function to draw indicator
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Coordinates for the indicator
        x, _ = get_center_bbox(bbox)
        y = int(bbox[3]) # Bottom of the bounding box

        # Width of the bounding box
        width = get_width_bbox(bbox)

        # Draw the indicator
        cv2.ellipse(frame,
            center=(x, y),
            axes=(width,
            int(width*0.35)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )


        # Rectangle for the track id
        rectangle_width = 40
        rectangle_height = 20
        rect_x1 = int(x - rectangle_width // 2)
        rect_x2 = int(x + rectangle_width // 2)
        rect_y1 = int((y - rectangle_height//2) + 15)
        rect_y2 = int((y + rectangle_height//2) + 15)

        # Draw the track id if the track id is not None
        if track_id is not None:
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)

            # Text for the track id
            text_x1 = int(rect_x1 + 12)
            if track_id > 99:
                text_x1 -= 10

            cv2.putText(frame, 
                str(track_id),
                org=(text_x1, rect_y1 + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        return frame

    # Function to draw ball indicator
    def draw_triangle(self, frame, bbox, color):
        # Coordinates for the indicator
        x, _ = get_center_bbox(bbox)
        y = int(bbox[1]) # Top of the bounding box

        # Vertices of the triangle
        pt1 = (x, y)
        pt2 = (x - 10, y - 20)  
        pt3 = (x + 10, y - 20)
        triangle = np.array([pt1, pt2, pt3])

        # Draw the triangle
        cv2.drawContours(frame, [triangle], 0, color, -1)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)

        return frame

    # Function to draw ball posession statistics
    def draw_posession_stats(self, frame, i_frame, ball_posession):
        # Copy the frame
        frame = frame.copy()

        # Draw a larger semi-transparent rectangle for the ball posession statistics in the top right corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 300, frame.shape[0] - 200), (frame.shape[1], frame.shape[0] - 70), (255, 255, 255, 128), cv2.FILLED)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Take the array of posession until the current frame
        ball_posession = ball_posession[:i_frame+1]

        # Get the number of frames in which each team has posession
        team1_frames = len(np.where(ball_posession == 1)[0])
        team2_frames = len(np.where(ball_posession == 2)[0])

        # Calculate the percentage of posession for each team
        team1_percentage = team1_frames / (i_frame + 1) * 100
        team2_percentage = team2_frames / (i_frame + 1) * 100

        # Text for the ball posession statistics
        text1 = f"Team 1: {team1_percentage:.2f}%"
        text2 = f"Team 2: {team2_percentage:.2f}%"

        # Text for the title of the ball posession statistics
        title_text = "Ball Posession:"

        # Draw the text for the ball posession statistics
        cv2.putText(frame, 
            title_text,
            org=(frame.shape[1] - 290, frame.shape[0] - 170),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(frame, 
            text1,
            org=(frame.shape[1] - 290, frame.shape[0] - 135),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(frame, 
            text2,
            org=(frame.shape[1] - 290, frame.shape[0] - 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        return frame

    # Function to draw the indicators for players, referees, and ball
    def draw_annotations(self, frames, tracks, ball_posession):
        # Output frames
        output_frames = []

        # Iterate over the frames to draw the annotations
        for i_frame, frame in enumerate(frames):
            # Copy the frame
            frame = frame.copy()

            # Get the dictionaries of players, referees, and ball
            player_dict = tracks['players'][i_frame]
            referee_dict = tracks['referees'][i_frame]
            ball_dict = tracks['ball'][i_frame]

            # Draw the indicators for players
            for player_id, player_info in player_dict.items():
                # Get the team color
                color = player_info['team_color']
                frame = self.draw_ellipse(frame, player_info['bbox'], color, player_id)

                # Draw the indicator for the player with the ball
                if player_info.get('has_ball', False):
                    frame = self.draw_triangle(frame, player_info['bbox'], (0, 0, 255))

            # Draw the indicators for referees
            for _, referee_info in referee_dict.items():
                frame = self.draw_ellipse(frame, referee_info['bbox'], (255, 0, 0))

            # Draw the indicators for the ball
            for _, ball_info in ball_dict.items():
                frame = self.draw_triangle(frame, ball_info['bbox'], (0, 255, 255))

            # Draw the ball posession statistics
            frame = self.draw_posession_stats(frame, i_frame, ball_posession)

            # Add frame to the output frames
            output_frames.append(frame)

        return output_frames