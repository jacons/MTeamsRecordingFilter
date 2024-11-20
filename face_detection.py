from Utils.functions import get_face_interval

# ---------- PARAMETERS ----------
video_path = "src/output/cropped_video.mp4"

start = "00:00:00"  # Specify start time in hh:mm:ss
end = "00:10:00"  # Specify end time in hh:mm:ss
# ---------- PARAMETERS ----------

get_face_interval(
    video_path=video_path,
    time = (start,end)
)

# intervals = [(34, 6284), (6315, 6318), (6320, 6320), (6322, 6322), (6371, 6373), (6375, 6375), (6377, 6402), (6490, 6503)]

