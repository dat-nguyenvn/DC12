folder_names_to_process = [
    "DJI_20230719145427_0002_V_video5", "DJI_20230719145427_0002_V_video3", "DJI_0601_video2",
    "DJI_0207", "DJI_0119", "DJI_0133_video3", "DJI_20230719145427_0002_V_video4",
    "DJI_0601_video3", "DJI_0142_video1", "DJI_0161_video1",
    "vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V",
    "DJI_0601_video4", "DJI_0204_video2", "DJI_0204_video1",
    "DJI_0117_video4", "DJI_0117_video3", "DJI_0133_video1",'DJI_0601_video5','DJI_0117_video1',
    'DJI_0117_video2','DJI_0133_video1 ','DJI_0601_video6','DJI_20230719145427_0002_V_video1',
    'DJI_20230719145427_0002_V_video2','DJI_20230719145816_0003_V_video2','DJI_20230720075532_0007_V_video2',
] 
import os
from PIL import Image

def downsample_frames(video_names, input_root, output_root, target_size=(1280, 720)):
    for video_name in video_names:
        print(f"video_name: {video_name}")
        input_folder = os.path.join(input_root, video_name, "frames")
        output_folder = os.path.join(output_root, video_name, "frame2k")

        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"Warning: {input_folder} does not exist, skipping.")
            continue

        frame_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
        frame_files.sort()

        for frame_file in frame_files:
            input_path = os.path.join(input_folder, frame_file)
            output_path = os.path.join(output_folder, frame_file)

            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(target_size, Image.ANTIALIAS)
                    img_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

        print(f"Finished downsampling {video_name}")

# Example usage:
video_list = ['vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V', 'DJI_0117_video3','DJI_0117_video4',
              'DJI_0133_video3','DJI_0207','DJI_0204_video1','DJI_0204_video2','DJI_0119','DJI_0142_video1',
              'DJI_0161_video1','DJI_0601_video2','DJI_0601_video3','DJI_0601_video4','DJI_20230719145427_0002_V_video3',
              'DJI_20230719145427_0002_V_video4','DJI_20230719145427_0002_V_video5','DJI_0601_video5','DJI_0117_video1',
              'DJI_0117_video2','DJI_0601_video6','DJI_20230719145427_0002_V_video2','DJI_20230719145816_0003_V_video2'
               ]  # Replace with your actual list
input_folder = '/media/ah23975/Data/capture'
output_folder = '/media/ah23975/Data/hddata'

downsample_frames(video_list, input_folder, output_folder)
