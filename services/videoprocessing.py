import cv2
import numpy as np
from pathlib import Path
import time  # For timing measurements
from utils.image_processing import draw_boxes

def process_video_frames(video_path, model):
    """
    Process each frame of the video using YOLOv5.

    Args:
        video_path (str): Path to the input video.
        model: Loaded YOLOv5 model.

    Returns:
        output_video_path (str): Path to the processed output video.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the output video path
    output_video_path = f"static/output_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Variables for timing measurements
    start_time = time.time()  # Start time for the entire video
    frame_times = []  # List to store processing time for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start time for the current frame
        frame_start_time = time.time()

        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 inference
        results = model(frame_rgb)

        # Draw bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame_rgb, results)

        # Convert back to BGR for saving
        frame_with_boxes_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(frame_with_boxes_bgr)

        # End time for the current frame
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)

        # Display the frame (optional)
        cv2.imshow("Processed Frame", frame_with_boxes_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate total and average processing time
    total_time = time.time() - start_time
    avg_time_per_frame = sum(frame_times) / len(frame_times) if frame_times else 0

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")
    print(f"Total frames processed: {len(frame_times)}")

    return output_video_path
