import argparse
import os
import time
import pandas as pd
from ultralytics import YOLO

def process_video_folder(model, video_folder, base_data_dir, output_dir):
    data = os.path.join(base_data_dir, video_folder, "frames")
    
    start_time = time.time()
    results = model.track(source=data, conf=0.1, iou=0.1, show=False, tracker="bytetrack.yaml", classes=[20, 22, 23], stream=True)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for {video_folder}: {elapsed_time:.5f} seconds")
    
    mot_results = []
    for frame_idx, frame in enumerate(results):
        frame_index = int(frame.path.split('frame_')[1].split('.jpg')[0])
        print(f"Processing frame: {frame.path}, index: {frame_idx}")
        
        for i in range(len(frame)):
            x1, y1, x2, y2 = frame.boxes.xyxy[i].tolist()
            w = x2 - x1
            h = y2 - y1
            track_id = None if frame.boxes.id is None else int(frame.boxes.id[i].tolist())
            mot_results.append([frame_idx + 1, track_id, x1, y1, w, h, -1, -1, -1, -1])
    
    mot_df = pd.DataFrame(mot_results, columns=["frame", "id", "x", "y", "w", "h", "confidence", "class_id", "abc", "bcd"])
    df_sorted = mot_df.sort_values(by="frame", key=lambda col: col.astype(int))
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{video_folder}.csv")
    df_sorted.to_csv(output_csv, index=False, header=False)
    print(f"Results for {video_folder} saved to {output_csv}")

def process_folder_by_index(model, base_data_dir, output_dir, index):
    folder_names_to_process = ['DJI_20230719145427_0002_V_video5', 'DJI_20230719145427_0002_V_video3', 'DJI_0601_video2', 
                               'DJI_0207', 'DJI_0119', 'DJI_0133_video3', 'DJI_20230719145427_0002_V_video4', 'DJI_0601_video3', 
                               'DJI_0142_video1', 'DJI_0161_video1', 'vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V', 
                               'DJI_0601_video4', 'DJI_0204_video2', 'DJI_0204_video1', 'DJI_0117_video4', 'DJI_0117_video3', 'DJI_0133_video1']
    
    if 0 <= index < len(folder_names_to_process):
        folder_name = folder_names_to_process[index]
        folder_path = os.path.join(base_data_dir, folder_name)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "frames")):
            print(f"Processing folder: {folder_name}")
            process_video_folder(model, folder_name, base_data_dir, output_dir)
        else:
            print(f"Folder '{folder_name}' does not exist or is missing frames directory.")
    else:
        print(f"Invalid index: {index}. Must be between 0 and {len(folder_names_to_process) - 1}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Tracking Script")
    parser.add_argument("--model", type=str, default="yolov8x-seg.pt", help="Path to the YOLO model file")
    parser.add_argument("--data_dir", type=str, default="/DC12/demo_data/eval/videos/", help="Base directory containing video frames")
    parser.add_argument("--output_dir", type=str, default="/DC12/demo_data/eval/output_csv/", help="Directory to save output CSVs")
    parser.add_argument("--index", type=int, required=True, help="Index of the folder to process")
    
    args = parser.parse_args()
    yolo_model = YOLO(args.model)
    process_folder_by_index(yolo_model, args.data_dir, args.output_dir, args.index)


import os
folders = [f for f in os.listdir("/home/ah23975/mypc/2024/DC12/demo_data/eval/videos") if os.path.isdir(os.path.join("/home/ah23975/mypc/2024/DC12/demo_data/eval/videos", f))]
