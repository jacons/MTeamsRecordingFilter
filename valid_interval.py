import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def process_frames(video_path: str, intervals: List[Tuple[int, int]]):
    """
    Processes frames from a video within specified intervals and visualizes them in grids.

    Args:
        video_path (str): Path to the video file.
        intervals (List[Tuple[int, int]]): List of (start_frame, end_frame) intervals to process.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {video_path}")

    # Process each interval
    for start_frame, end_frame in intervals:
        frames = extract_frames(cap, start_frame, end_frame)
        visualize_frames_in_grids(frames)

    # Release the video capture object
    cap.release()


def extract_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames from the video within a specific range.

    Args:
        cap (cv2.VideoCapture): Opened video capture object.
        start_frame (int): Starting frame number.
        end_frame (int): Ending frame number.

    Returns:
        List[Tuple[int, np.ndarray]]: List of (frame_index, frame_cropped).
    """
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}. Stopping early.")
            break
        # Crop the frame as required
        cropped_frame = frame[65:1080, 0:1675, :]
        frames.append((frame_idx, cropped_frame))

    return frames


def visualize_frames_in_grids(frames: List[Tuple[int, np.ndarray]], grid_size: int = 5):
    """
    Visualizes frames in grids, adding frame indices as labels.

    Args:
        frames (List[Tuple[int, np.ndarray]]): List of (frame_index, frame) tuples.
        grid_size (int): Number of rows/columns in the grid (default is 5x5).
    """
    total_frames = len(frames)
    frames_per_grid = grid_size ** 2

    for i in range(math.ceil(total_frames / frames_per_grid)):
        # Get the subset of frames for the current grid
        start_idx = i * frames_per_grid
        end_idx = start_idx + frames_per_grid
        grid_frames = frames[start_idx:end_idx]

        # Separate indices and frames
        frame_indices = [str(idx) for idx, _ in grid_frames]
        grid_images = [img for _, img in grid_frames]

        # Fill missing slots with placeholder frames if the grid is incomplete
        num_missing = frames_per_grid - len(grid_frames)
        if num_missing > 0:
            placeholder_frame = np.ones((5, 5, 3))  # Small white square
            grid_images.extend([placeholder_frame] * num_missing)
            frame_indices.extend([""] * num_missing)

        # Plot the frames in a grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 10))
        axes = axes.flatten()
        for ax, image, idx in zip(axes, grid_images, frame_indices):
            ax.imshow(image)
            ax.text(0, 40, idx, fontdict={'fontsize': 25, 'fontweight': 'bold', 'color': 'red'})
            ax.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

process_frames("src/videos/video_test.mp4", intervals=[(6280,6288)])
