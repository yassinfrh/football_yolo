import cv2

def read_video(video_path):
    # Capture video from file
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Read until video is completed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Append the frame to the `frames` list
        frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames

def save_video(output_frames, output_path):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))
    # Write the frames to the output video
    for frame in output_frames:
        out.write(frame)
    # Release the VideoWriter object
    out.release()