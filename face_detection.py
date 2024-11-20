from pathlib import Path
from Utils.functions import detect_faces

# ---------- PARAMETERS ----------
video_path = "src/output/cropped_video.mp4"
folder = Path("./output/test1/")
start = "00:00:00"  # Specify start time in hh:mm:ss
end = "00:10:00"  # Specify end time in hh:mm:ss
# ---------- PARAMETERS ----------

detect_faces(video_path=video_path, time = (start,end), folder=folder)