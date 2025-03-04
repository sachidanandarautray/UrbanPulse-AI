import torch
from pathlib import Path
import pathlib  # Import pathlib for the workaround


# Workaround for PosixPath issue on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from services.videoprocessing import process_video_frames

def load_yolov5_model(weights_path):
    """
    Load the YOLOv5 model from the given weights path.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

def process_video_with_yolo(video_path):
    """
    Process a video using YOLOv5 and save the output video.
    """
    # Load the YOLOv5 model
    model = load_yolov5_model("models/best.pt")  # Ensure this path is correct
    if model is None:
        raise Exception("Failed to load YOLOv5 modelwhywhen.")

    # Process the video frames
    output_video_path = process_video_frames(video_path, model)

    return output_video_path
