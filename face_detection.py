import cv2

from Utils.functions import get_face_interval

# ---------- PARAMETERS ----------
video_path = "src/output/cropped_video.mp4"

start = "00:00:00"  # Specify start time in hh:mm:ss
end = "00:10:00"  # Specify end time in hh:mm:ss
# ---------- PARAMETERS ----------


# ---------- PARAMETERS ----------

get_face_interval(
    video_path=video_path,
    time = (start,end)
)