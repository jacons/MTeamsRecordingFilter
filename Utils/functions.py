from datetime import datetime
from typing import Tuple, List

import cv2
from numpy import ndarray
from tqdm import tqdm
import subprocess

def time2frame(time: Tuple[str,str],fps:float):
    """
    Calculate start and end frame numbers based on start and end times
    :param time: (start, end)
    :param fps:
    :return:
    """
    origin_time = datetime.strptime("00:00:00", "%H:%M:%S")
    start_time = datetime.strptime(time[0], "%H:%M:%S")
    end_time = datetime.strptime(time[1], "%H:%M:%S")
    start_frame = int(fps * (start_time - origin_time).total_seconds())
    end_frame = int(fps * (end_time - origin_time).total_seconds())
    return start_frame, end_frame


def extract_audio(video_path:str, time: Tuple[str,str], output_audio_path:str):
    """
    Extract audio from a video file for the given time range.
    :param video_path: Path to the original video.
    :param time: Start/End time in HH:MM:SS format.
    :param output_audio_path: Path to save the extracted audio.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite existing files
        "-i", video_path,
        "-ss", time[0],
        "-to", time[1],
        "-vn",  # No video, audio only
        "-acodec", "copy",
        output_audio_path,
    ]
    subprocess.run(command, check=True)

def merge_video_audio(video_no_audio:str, audio_path: str, output_file: str):
    """
    Combine the processed video with extracted audio.
    :param video_no_audio: Path to the processed video (no audio).
    :param audio_path: Path to the extracted audio.
    :param output_file: Output file path with video and audio combined.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", video_no_audio,
        "-i", audio_path,
        "-c:v", "copy",  # Copy the video stream without re-encoding
        "-c:a", "aac",   # Ensure compatibility with most players
        output_file,
    ]
    subprocess.run(command, check=True)


def initialize_video_writer(output_file:str, frame:Tuple[int,int], fps:float):
    """
    :param output_file:
    :param frame: Tuple(height, width)
    :param fps:
    :return:
    """
    out = cv2.VideoWriter(
        filename=output_file,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(frame[1], frame[0])
    )
    if not out.isOpened():
        raise RuntimeError("Error: Failed to initialize video writer.")
    return out


def process_video(video_path:str, output_file:str, time:Tuple[str,str], crop_frame : Tuple[slice,slice]):

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {video_path}")

    # retrieve the fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # given the start/ end time, we calculate the start/end frame
    start_frame, end_frame = time2frame(time, fps)

    # Set the reader to a certain frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # from the slide, we obtain the  output frame dimensions
    height = crop_frame[0].stop - crop_frame[0].start
    width = crop_frame[1].stop - crop_frame[1].start
    # Initialize the output video object
    out = initialize_video_writer(output_file=output_file,
                                  frame=(height,width),
                                  fps=fps)

    total_frames = end_frame - start_frame
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    try:
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame = frame[crop_frame[0], crop_frame[1]]
            out.write(cropped_frame)
            pbar.update(1)
    finally:
        # Ensure resources are released even if an error occurs
        pbar.close()
        out.release()
        cap.release()
        cv2.destroyAllWindows()


def detect_face(frame: ndarray, face_classifier, show_box: bool = False,) -> bool:
    faces = face_classifier.detectMultiScale(image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(40, 40))
    if show_box:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return len(faces) != 0


def get_face_interval(video_path: str, time: Tuple[str, str]):


    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {video_path}")

    # retrieve the fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    # given the start/ end time, we calculate the start/end frame
    start_frame, end_frame = time2frame(time, fps)

    # Set the reader to a certain frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = end_frame - start_frame
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    face_classifier = cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml')

    print("Start on frame: ", start_frame)
    face_frame = []
    try:
        for current in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if detect_face(frame, face_classifier, show_box=False):
                face_frame.append(current + start_frame)

            # cv2.imshow("My Face Detection Project", frame)

            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            pbar.update(1)
    finally:
        # Ensure resources are released even if an error occurs
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

    print(face_frame)
    print(get_intervals(face_frame))

def get_intervals(sequence:List[int])->List[Tuple[int,int]]:
    prev, sx_bound  = sequence[0], sequence[0]
    intervals = []

    for curr in sequence[1:]:
        if curr - prev > 1:
            intervals.append((sx_bound, prev))
            sx_bound = curr
        prev = curr
    intervals.append((sx_bound, prev))

    return intervals

# print(get_intervals([1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,19,29,30,31,32,40]))