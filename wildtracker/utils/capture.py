import os
#import imageio
import imageio.v3 as iio
import argparse

def extract_frames(video_path, output_folder,distance=50):
    # Create a folder with the same name as the video file
    video_name = os.path.basename(video_path)
    frame_folder = os.path.join(output_folder, video_name.split('.')[0],'cutoff_frames')
    os.makedirs(frame_folder, exist_ok=True)

    # Open the video file using imageio with the FFMPEG plugin
    #reader = imageio.get_reader(video_path, plugin="FFMPEG")

    # Initialize frame counter
    frame_count = 0

    # Read frames from the video and extract every 10th frame
    for i, frame in enumerate(iio.imiter(video_path,plugin="FFMPEG")):
        
        if frame_count % distance == 0:  # extract every x frame
            #frame_path = os.path.join(frame_folder, f"frame_{frame_count // 500:03d}.jpg")
            frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
            iio.imwrite(frame_path, frame)
            print(f"Saved frame: {frame_path}")
        frame_count += 1

    # Close the video file
    #reader.close()

def extract_all_frames(video_path, output_folder):
    # Create a folder with the same name as the video file
    video_name = os.path.basename(video_path)
    frame_folder = os.path.join(output_folder, video_name.split('.')[0],'frames')
    os.makedirs(frame_folder, exist_ok=True)

    # Open the video file using imageio with the FFMPEG plugin
    #reader = imageio.get_reader(video_path, plugin="FFMPEG")

    # Initialize frame counter
    frame_count = 0

    # Read frames from the video and extract every 10th frame
    for i, frame in enumerate(iio.imiter(video_path,plugin="FFMPEG")):
        frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
        iio.imwrite(frame_path, frame)
        print(f"Saved frame: {frame_path}")
        frame_count += 1

    # Close the video file
    #reader.close()

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('--video_folder',default='/data/captest/evaluation_data/', help='Path to the video file',)
    parser.add_argument('--output_folder', default='data/captest/cap_eval_data/', help='Path to the output folder (default: current directory)',)
    args = parser.parse_args()

    # List all video files in the folder
    video_files = [f for f in os.listdir(args.video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.wmv'))]

    # Extract frames from each video file
    for video_file in video_files:
        video_path = os.path.join(args.video_folder, video_file)
        extract_all_frames(video_path, args.output_folder)
        #extract_frames(video_path, args.output_folder)

if __name__ == '__main__':
    main()
