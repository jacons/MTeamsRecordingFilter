from Utils.functions import process_video, extract_audio, merge_video_audio

# ---------- PARAMETERS ----------
video_path = "src/videos/video_test.mp4"
temp_audio_path = "src/tmp/audio_tmp.aac"
temp_video_path = "src/tmp/video_tmp.mp4"
output_file = "src/output/cropped_video.mp4"

start = "00:00:00"  # Specify start time in hh:mm:ss
end = "00:10:00"  # Specify end time in hh:mm:ss

frame_width = 1675
frame_height = 1080 - 65
width_slice = slice(0, frame_width)
height_slice = slice(65, 1080)
# ---------- PARAMETERS ----------


extract_audio(video_path=video_path,
              time=(start,end),
              output_audio_path=temp_audio_path)

# Run the video processing
process_video(video_path=video_path,
              output_file=temp_video_path,
              time=(start,end),
              crop_frame=(height_slice,width_slice))

# Merge the processed video with the extracted audio
merge_video_audio(video_no_audio=temp_video_path,
                  audio_path=temp_audio_path,
                  output_file=output_file)