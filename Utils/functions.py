from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Callable, Union

import cv2
from numpy import ndarray
from tqdm import tqdm
import subprocess
from datetime import timedelta


def time2frame(time: Tuple[str, Union[str,None]], fps: float, tot_frames: float)-> Tuple[int, int]:
    """
    Calculate start and end frame numbers based on start and end times
    :param time: (start, end)
    :param fps:
    :param tot_frames:
    :return:
    """
    origin_time = datetime.strptime("00:00:00", "%H:%M:%S")

    start_time = datetime.strptime(time[0], "%H:%M:%S")
    start_frame = int(fps * (start_time - origin_time).total_seconds())

    if time[1] is None:
        end_frame = int(tot_frames)
    else:
        end_time = datetime.strptime(time[1], "%H:%M:%S")
        end_frame = int(fps * (end_time - origin_time).total_seconds())
    return start_frame, end_frame

def execute_ffmpeg(command: List[str]):
    """Run an ffmpeg command."""
    subprocess.run(command, check=True)

def extract_audio(video_path: str, time: Tuple[str, str], output_audio_path: str):
    """Extract audio from a video for a specific time range."""
    command = [
        "ffmpeg", "-y", "-i", video_path, "-ss", time[0], "-to", time[1],
        "-vn", "-acodec", "copy", output_audio_path
    ]
    execute_ffmpeg(command)

def merge_video_audio(video_no_audio: str, audio_path: str, output_file: str):
    """Merge a video without audio with an audio track."""
    command = [
        "ffmpeg", "-y", "-i", video_no_audio, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", output_file
    ]
    execute_ffmpeg(command)

def initialize_video_writer(output_file: str, frame: Tuple[int, int], fps: float)-> cv2.VideoWriter:
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

def get_intervals(sequence: List[int]) -> List[Tuple[int, int]]:
    # print(get_intervals([1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,19,29,30,31,32,40]))
    intervals = []
    if not sequence:
        return intervals

    prev, sx_bound = sequence[0], sequence[0]
    for curr in sequence[1:]:
        if curr - prev > 1:
            intervals.append((sx_bound, prev))
            sx_bound = curr
        prev = curr
    intervals.append((sx_bound, prev))
    return intervals

def frame_to_timestamp(fps_rate: float, frame_number: int) -> str:
    milliseconds = (frame_number / fps_rate) * 1000  # Tempo in ms
    timestamp = str(timedelta(milliseconds=milliseconds))
    return timestamp


def detect_face(frame: ndarray, classifiers) -> bool:

    to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flag = any(len(classifier.detectMultiScale(image=to_gray, scaleFactor=1.1, minNeighbors=5,minSize=(40, 40))>0)
               for classifier in classifiers)

    return flag

def process_frames(video_path: str, time: Tuple[str, Union[str, None]], fn:Callable):
    """General function to process frames within a specific time range."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame, end_frame = time2frame(time=time, fps=fps, tot_frames=tot_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = end_frame - start_frame

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        for n_frame in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            fn(frame=frame, n_frame=n_frame)
            pbar.update(1)
    cap.release()

def crop_video(video_path: str, output_file: str, time: Tuple[str, str],
               crop_frame: Tuple[slice, slice]):


    # from the slide, we obtain the  output frame dimensions
    height = crop_frame[0].stop - crop_frame[0].start
    width = crop_frame[1].stop - crop_frame[1].start

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out = initialize_video_writer(output_file=output_file, frame=(height, width), fps=fps)

    def crop_fn(frame:ndarray, n_frame:int):
        cropped_frame = frame[crop_frame[0], crop_frame[1]]
        out.write(cropped_frame)

    process_frames(video_path=video_path, time=time, fn=crop_fn)

    out.release()


def get_and_print_intervals(face_frames:List[int], fps:float, folder:Path):
    intervals = get_intervals(face_frames)

    Path(folder).mkdir(parents=True, exist_ok=True)
    with Path(folder / "frame_intervals.txt").open("w") as f:
        for line in intervals:
            f.write(f"{line}\n")

    with Path(folder / "time_intervals.txt").open("w") as f:
        for a, b in intervals:
            f.write(f"{frame_to_timestamp(fps, a)} {frame_to_timestamp(fps, b)}\n")

def detect_faces(video_path: str, time: Tuple[str, str], folder:Path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    classifiers = (
        cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier('./src/haarcascade_profileface.xml')
    )

    face_frames = []
    def detect_fn(frame: ndarray, n_frame:int):
        to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag = any(
            len(classifier.detectMultiScale(image=to_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))) > 0
            for classifier in classifiers)
        if flag:
            face_frames.append(n_frame)

    process_frames(video_path=video_path, time=time, fn=detect_fn)
    get_and_print_intervals(face_frames,fps, folder=folder)


def crop_detect(video_path: str, time: Tuple[str, Union[str,None]], crop_frame: Tuple[slice, slice],
                folder:Path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    classifiers = (
        cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier('./src/haarcascade_profileface.xml')
    )
    face_frames = []
    def crop_detect_fn(frame:ndarray, n_frame:int):
        cropped_frame = frame[crop_frame[0], crop_frame[1]]
        to_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        flag = any(
            len(classifier.detectMultiScale(image=to_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))) > 0
            for classifier in classifiers)
        if flag:
            face_frames.append(n_frame)

    process_frames(video_path=video_path, time=time, fn=crop_detect_fn)
    get_and_print_intervals(face_frames=face_frames,fps=fps, folder=folder)
