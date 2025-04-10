# from ultralytics import YOLO
# import os
# import cv2
# import pandas as pd
# import os
# import time
# # Load YOLOv8 model (you can use the pre-trained weights)
# from ultralytics import YOLO
# #data="/data/captest/eval/frames/"
# #data="/data/captest/cap_eval_data/vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V/frames/"
# #data="/data/captest/cap_eval_data/uav0000009_03358_v/"
# data="/DC12/demo_data/eval/videos/DJI_0117_video3/frames/"

# #vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V
# # Configure the tracking parameters and run the tracker
# model = YOLO("yolov8x-seg.pt")
# start_time = time.time()
# results = model.track(source=data, conf=0.1, iou=0.1, show=False,tracker="bytetrack.yaml",classes=[20,22,23],stream=True)
# #results = model.track(source=data, conf=0.1, iou=0.5, show=True,tracker="bytetrack.yaml",classes=[0,1,2,3,5,7])
# end_time = time.time()
# elapsed_time = end_time - start_time

# #text=text_image_size + text_fps
# print(f"Elapsed time: {elapsed_time:.5f} seconds")
# mot_results = []

# # Iterate over the frames in the tracking results
# for frame_idx, frame in enumerate(results):
#     frame_index = int(frame.path.split('frame_')[1].split('.jpg')[0])
    

#     print("frame.path",frame.path)
#     print("frame_index",frame_idx)
#     #obj=frame.boxes
#     # Iterate over each object in the frame
#     for i in range (0,len(frame)):
#         x1, y1, x2, y2= frame.boxes.xyxy[i].tolist()
#         w = x2 - x1
#         h = y2 - y1
#         #track_id=int(frame.boxes.id[i].tolist())
#         track_id = None if frame.boxes.id is None else int(frame.boxes.id[i].tolist())


#         # Append the data to the MOT result list
#         mot_results.append([frame_idx + 1, track_id, x1, y1, w, h, -1, -1,-1,-1])

# # Convert the list to a DataFrame
# mot_df = pd.DataFrame(mot_results, columns=["frame", "id", "x", "y", "w", "h", "confidence", "class_id","abc","bcd"])
# df_sorted = mot_df.sort_values(by="frame", key=lambda col: col.astype(int))
# # Save the DataFrame to a CSV file
# df_sorted.to_csv('/DC12/demo_data/eval/bytetrack/DJI_0117_video3.csv', index=False,header=False)



import argparse
import os
import time
import pandas as pd
from ultralytics import YOLO

def process_video_folder(model, video_folder, base_data_dir, output_dir):
    data = os.path.join(base_data_dir, video_folder, "frames")
    
    # Count total frames in the folder
    total_frames = len([f for f in os.listdir(data) if f.endswith('.jpg')])
    
    if total_frames == 0:
        print(f"❌ No frames found in {data}. Skipping...")
        return
    
    #start_time = time.time()  # Start time before processing

    # YOLO tracking process
    results = model.track(source=data, conf=0.1, iou=0.1, show=False, tracker="bytetrack.yaml", classes=[20, 22, 23], stream=True,imgsz=(2160,3840))
    #results = model.track(source=data, conf=0.1, iou=0.1, show=False, tracker="bytetrack.yaml", classes=[20, 22, 23], stream=True)
    
    #elapsed_time = end_time - start_time  # Total processing time
    #fps = total_frames / elapsed_time  # Calculate FPS
    
    

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
    
    # Convert results to DataFrame and sort by frame index
    mot_df = pd.DataFrame(mot_results, columns=["frame", "id", "x", "y", "w", "h", "confidence", "class_id", "abc", "bcd"])
    df_sorted = mot_df.sort_values(by="frame", key=lambda col: col.astype(int))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV
    output_csv = os.path.join(output_dir, f"{video_folder}.csv")
    df_sorted.to_csv(output_csv, index=False, header=False)
    
    print(f"📁 Results for {video_folder} saved to {output_csv}")
    #print(f"✅ Processed {total_frames} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS) for {video_folder}")

def process_folder_by_index(model, base_data_dir, output_dir, index):
    # folder_names_to_process = ['DJI_20230719145427_0002_V_video5', 'DJI_20230719145427_0002_V_video3', 'DJI_0601_video2', 
    #                            'DJI_0207', 'DJI_0119', 'DJI_0133_video3', 'DJI_20230719145427_0002_V_video4', 'DJI_0601_video3', 
    #                            'DJI_0142_video1', 'DJI_0161_video1', 'vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V', 
    #                            'DJI_0601_video4', 'DJI_0204_video2', 'DJI_0204_video1', 'DJI_0117_video4', 'DJI_0117_video3', 'DJI_0133_video1']
    folder_names_to_process = ['DJI_0117_video4']    
    if 0 <= index < len(folder_names_to_process):
        folder_name = folder_names_to_process[index]
        folder_path = os.path.join(base_data_dir, folder_name)
        
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "frames")):
            print(f"🔄 Processing folder: {folder_name}")
            process_video_folder(model, folder_name, base_data_dir, output_dir)
        else:
            print(f"⚠️ Folder '{folder_name}' does not exist or is missing 'frames' directory.")
    else:
        print(f"❌ Invalid index: {index}. Must be between 0 and {len(folder_names_to_process) - 1}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Tracking Script")
    parser.add_argument("--model", type=str, default="yolov8l-seg.pt", help="Path to the YOLO model file")
    parser.add_argument("--data_dir", type=str, default="/DC12/demo_data/eval/videos/", help="Base directory containing video frames")
    parser.add_argument("--output_dir", type=str, default="/DC12/demo_data/eval/output_csv/", help="Directory to save output CSVs")
    parser.add_argument("--index", type=int, required=True, help="Index of the folder to process")
    
    args = parser.parse_args()
    yolo_model = YOLO(args.model)
    start_time = time.time() 
    process_folder_by_index(yolo_model, args.data_dir, args.output_dir, args.index)
    end_time = time.time() 
    elapsed_time = end_time - start_time

    print(f"✅ Processed {elapsed_time:.2f}, fps = {(670/elapsed_time):.2f}")