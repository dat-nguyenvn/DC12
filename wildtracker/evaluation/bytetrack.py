from ultralytics import YOLO
import os
import cv2
import pandas as pd
import os

# Load YOLOv8 model (you can use the pre-trained weights)
from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolov8x.pt")
results = model.track(source="/data/captest/eval/frames/", conf=0.1, iou=0.5, show=True,tracker="bytetrack.yaml")
mot_results = []
print()
# Iterate over the frames in the tracking results
for frame_idx, frame in enumerate(results):
    frame_index = int(frame.path.split('frame_')[1].split('.jpg')[0])
    print("frame.path",frame.path)
    print("frame_index",frame_index)
    #obj=frame.boxes
    # Iterate over each object in the frame
    for i in range (0,len(frame)):
        x1, y1, x2, y2= frame.boxes.xyxy[i].tolist()
        w = x2 - x1
        h = y2 - y1
        track_id=int(frame.boxes.id[i].tolist())


        # Append the data to the MOT result list
        mot_results.append([frame_index + 1, track_id, x1, y1, w, h, -1, -1,-1,-1])

# Convert the list to a DataFrame
mot_df = pd.DataFrame(mot_results, columns=["frame", "id", "x", "y", "w", "h", "confidence", "class_id","abc","bcd"])
df_sorted = mot_df.sort_values(by="frame", key=lambda col: col.astype(int))
# Save the DataFrame to a CSV file
df_sorted.to_csv('bytetrack.csv', index=False,header=False)