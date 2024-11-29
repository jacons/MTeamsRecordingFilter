import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Callable, Union

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import subprocess
from datetime import timedelta

face_tuple = namedtuple("face", ["frame", "boxes"])


def time2frame(time: Tuple[str, Union[str, None]], fps: float, tot_frames: float) -> Tuple[int, int]:
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


def initialize_video_writer(output_file: str, frame: Tuple[int, int], fps: float) -> cv2.VideoWriter:
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


def get_intervals(frame_sequence: List[face_tuple]) -> List[Tuple[int, int, float]]:
    # aaa = [face_tuple(1, 4), face_tuple(2, 7), face_tuple(3, 0), face_tuple(4, 9), face_tuple(5, 2), face_tuple(6, 6),
    #        face_tuple(7, 3), face_tuple(8, 9), face_tuple(10, 0), face_tuple(11, 0), face_tuple(12, 1),
    #        face_tuple(13, 8), face_tuple(14, 12), face_tuple(15, 4), face_tuple(17, 7), face_tuple(19, 2),
    #        face_tuple(29, 0), face_tuple(30, 1), face_tuple(31, 2), face_tuple(32, 6), face_tuple(40, 4)]
    # print(get_intervals(aaa))

    intervals = []
    if not frame_sequence:
        raise ValueError("Input frame sequence is empty.")

    prev, sx_bound = frame_sequence[0].frame, frame_sequence[0].frame
    id_prev, id_sx_bound = 0 ,0
    for id_curr, curr in enumerate(frame_sequence[1:],1):
        if curr.frame - prev > 1:

            intervals.append((sx_bound, prev, np.mean([i.boxes for i in frame_sequence[id_sx_bound:id_prev+1]])))
            sx_bound = curr.frame
            id_sx_bound = id_curr
        prev = curr.frame
        id_prev = id_curr

    # mean_faces = 0 if id_sx_bound == id_prev else np.mean([i.boxes for i in frame_sequence[id_sx_bound:id_prev]])
    intervals.append((sx_bound, prev, float(np.mean([i.boxes for i in frame_sequence[id_sx_bound:id_prev+1]]))))

    return intervals


def frame_to_timestamp(fps_rate: float, frame_number: int) -> timedelta:
    seconds = (frame_number / fps_rate)  # Tempo in ms
    return timedelta(seconds=seconds)


def detect_face(frame: ndarray, classifiers) -> bool:
    to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flag = any(len(classifier.detectMultiScale(image=to_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) > 0)
               for classifier in classifiers)

    return flag


def process_frames(video_path: str, time: Tuple[str, Union[str, None]], fn: Callable, debug: bool = False):
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
                print(f"Warning: Could not read frame {start_frame + n_frame}.")
                break
            output_frame = fn(frame=frame, n_frame=start_frame+n_frame)
            if debug:
                cv2.imshow('Frame Processing', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
                    break
            pbar.update(1)
    cap.release()
    cv2.destroyAllWindows()


def crop_video(video_path: str, output_file: str, time: Tuple[str, str],
               crop_frame: Tuple[slice, slice]):
    # from the slide, we obtain the  output frame dimensions
    height = crop_frame[0].stop - crop_frame[0].start
    width = crop_frame[1].stop - crop_frame[1].start

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out = initialize_video_writer(output_file=output_file, frame=(height, width), fps=fps)

    def crop_fn(frame: ndarray, n_frame: int):
        cropped_frame = frame[crop_frame[0], crop_frame[1], :]
        out.write(cropped_frame)
        return cropped_frame

    process_frames(video_path=video_path, time=time, fn=crop_fn)

    out.release()


def get_and_print_intervals(face_frames: List[face_tuple], fps: float, folder: Path):
    format_ = "%H:%M:%S.%f"
    intervals = get_intervals(frame_sequence=face_frames)

    Path(folder).mkdir(parents=True, exist_ok=True)

    logs = []
    for line in intervals:  # [from , to , mean of face revealed]
        start = frame_to_timestamp(fps, line[0])
        end = frame_to_timestamp(fps, line[1])
        logs.append(dict(
            frame_start=line[0],
            frame_end=line[1],
            frame_number=line[1] - line[0],
            time_start=str(start),
            time_end=str(end),
            time_delta=str(end - start),
            mean_face=line[2]
        ))

    with Path(folder / "intervals.txt").open("w") as f:
        f.write(json.dumps(logs, indent=4))


def detect_faces(video_path: str, time: Tuple[str, str], folder: Path,debug:bool=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    classifiers = (
        cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier('./src/haarcascade_profileface.xml')
    )

    face_frames = []
    def detect_fn(frame: ndarray, n_frame: int):
        to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag, boxes = False, 0

        for classifier in classifiers:
            boxes = classifier.detectMultiScale(image=to_gray, scaleFactor=1.4, minNeighbors=5, minSize=(40, 40))
            if debug:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            if len(boxes) > 0:
                flag = True
                break
        if flag:
            face_frames.append(face_tuple(frame=n_frame, boxes=len(boxes)))
        return frame

    process_frames(video_path=video_path, time=time, fn=detect_fn)
    get_and_print_intervals(face_frames=face_frames, fps=fps, folder=folder)


def crop_detect(video_path: str, time: Tuple[str, Union[str, None]], crop_frame: Tuple[slice, slice],
                folder: Path, debug: bool = False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    classifiers = (
        cv2.CascadeClassifier('./src/haarcascade/haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier('./src/haarcascade/haarcascade_profileface.xml')
    )
    face_frames = []
    def crop_detect_fn(frame: ndarray, n_frame: int):
        cropped_frame = frame[crop_frame[0], crop_frame[1], :]
        to_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        for classifier in classifiers:
            boxes = classifier.detectMultiScale(image=to_gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))
            if debug:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            if len(boxes) > 0:
                face_frames.append(face_tuple(frame=n_frame, boxes=len(boxes)))
                break
        return cropped_frame

    process_frames(video_path=video_path, time=time, fn=crop_detect_fn, debug=debug)
    get_and_print_intervals(face_frames=face_frames, fps=fps, folder=folder)
    cv2.destroyAllWindows()