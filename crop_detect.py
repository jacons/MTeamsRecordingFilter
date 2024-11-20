from Utils.functions import crop_detect

# ---------- PARAMETERS ----------
video_path = "src/videos/video_test.mp4"

start = "00:00:00"  # Specify start time in hh:mm:ss
end = "00:10:00"  # Specify end time in hh:mm:ss

frame_width = 1675
frame_height = 1080 - 65
width_slice = slice(0, frame_width)
height_slice = slice(65, 1080)
# ---------- PARAMETERS ----------


# Run the video processing
crop_detect(video_path=video_path,
           time=(start,end),
           crop_frame=(height_slice,width_slice))

