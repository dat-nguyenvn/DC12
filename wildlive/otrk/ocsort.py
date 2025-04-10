import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sahi.utils.yolov8 import download_yolov8s_model, download_yolov8s_seg_model
import os
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from tqdm import tqdm
import json
import argparse
import torch
import pandas as pd
import time
from natsort import natsorted
import matplotlib  # ✅ Import matplotlib first
from boxmot.trackers.ocsort.ocsort import OcSort
matplotlib.use('Agg')
#from sort import Sort  # ✅ Import SORT tracker

from boxmot.tracker_zoo import create_tracker

def make_parser():
    parser = argparse.ArgumentParser("SORT Tracking with YOLOv8 + SAHI")
    parser.add_argument("-idx", "--index", type=int, default=0)
    parser.add_argument("-n", "--model", type=str, default="yolov8x-seg.pt", help="model name")
    parser.add_argument("--data_dir", default="/DC12/demo_data/eval/videos/", help="Path to video frames")
    parser.add_argument("--output_dir", type=str, default="/DC12/demo_data/eval", help="Output directory")
    parser.add_argument("--save_result", action="store_true", help="Save results?")
    parser.add_argument("--conf", default=0.3, type=float, help="Detection confidence threshold")
    parser.add_argument("--fps", default=30, type=int, help="Frame rate (fps)")
    return parser


def detect_objects_sahi(img, model_name, conf=0.3):
    """
    Runs YOLOv8 + SAHI detection on an image.
    """
    detection_model_seg = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_name,
        confidence_threshold=conf,
        device="cuda",  # or "cpu"
    )

    try:
        result = get_sliced_prediction(
            img,
            detection_model_seg,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        detections = []
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            conf = obj.score.value
            class_id = obj.category.id
            detections.append([x1, y1, x2, y2, conf,class_id])

        return np.array(detections)

    except Exception as e:
        print(f"❌ Error in get_sliced_prediction: {e}")
        return []


def track_objects_from_folder(tracker, folder_name, image_folder, output_dir, model):
    """
    Runs SAHI detection + SORT tracking on images from a folder.
    """
    image_folder = os.path.join(image_folder, "frames")
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    image_files = natsorted(image_files)

    tracking_results = []
    total_frames = len(image_files)
    start_time = time.time()

    for frame_id, img_name in tqdm(enumerate(image_files), total=total_frames, desc="Processing Frames"):
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Warning: Could not read {img_name}. Skipping...")
            continue

        detections = detect_objects_sahi(image, model_name=model)  # Run SAHI detection

        # Convert detections to SORT format: [x1, y1, x2, y2, score]
        dets = []
        for det in detections:
            x1, y1, x2, y2, conf,class_id = det
            dets.append([x1, y1, x2, y2, conf,class_id])

        dets = np.array(dets)
        print(dets.shape)
        # Update SORT tracker
        if len(dets) > 0:
            tracked_objects = tracker.update(dets,image)
        else:
            tracked_objects = []

        # Save tracking results
        for track in tracked_objects:
            #print("track",track.shape)
            #print(track)
            x1, y1, x2, y2, track_id ,confi,cls,det_ind= track.astype(int)
            w = x2 - x1
            h = y2 - y1
            tracking_results.append([frame_id, track_id, x1, y1, w, h])

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = total_frames / elapsed_time
    print(f"Frame {frame_id}, FPS: {fps:.2f}")

    speed_output_dir = os.path.join(output_dir, "speed")
    gpu_model=torch.cuda.get_device_name()
    os.makedirs(speed_output_dir, exist_ok=True)
    speed_csv = os.path.join(speed_output_dir, f"{folder_name}.csv")
    

    speed_data = {
        "folder_name": [folder_name],
        "gpu":[gpu_model],
        "total_frames": [total_frames],
        "elapsed_time": [elapsed_time],
        "fps": [fps]
    }

    speed_df = pd.DataFrame(speed_data)
    speed_df.to_csv(speed_csv, index=True,header=True)

    # Save tracking results
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{folder_name}.csv")
    df = pd.DataFrame(tracking_results, columns=["frame", "track_id", "x1", "y1", "w", "h"])
    df.to_csv(output_csv, index=False, header=False)
    print(f"✅ Tracking results saved: {output_csv}")


if __name__ == "__main__":
    folder_names_to_process = [
        "DJI_20230719145427_0002_V_video5", "DJI_20230719145427_0002_V_video3", "DJI_0601_video2",
        "DJI_0207", "DJI_0119", "DJI_0133_video3", "DJI_20230719145427_0002_V_video4",
        "DJI_0601_video3", "DJI_0142_video1", "DJI_0161_video1",
        "vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V",
        "DJI_0601_video4", "DJI_0204_video2", "DJI_0204_video1",
        "DJI_0117_video4", "DJI_0117_video3", "DJI_0133_video1",'DJI_0601_video5','DJI_0117_video1',
        'DJI_0117_video2','DJI_0133_video1 ','DJI_0601_video6','DJI_20230719145427_0002_V_video1',
        'DJI_20230719145427_0002_V_video2','DJI_20230719145816_0003_V_video2','DJI_20230720075532_0007_V_video2',
    ] #0->25

    args = make_parser().parse_args()

    if 0 <= args.index < len(folder_names_to_process):
        folder_name = folder_names_to_process[args.index]
        folder_path = os.path.join(args.data_dir, folder_name)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "frames")):
            print(f"Processing folder: {folder_name}")

            # ✅ Replace BYTETracker with SORT
            tracker = create_tracker(tracker_type='ocsort',reid_weights='lmbn_n_cuhk03_d.pt',tracker_config='/home/boxmot/boxmot/configs/ocsort.yaml') #
            tracker_method = tracker #Sort()  

            track_objects_from_folder(tracker_method, folder_name, folder_path, args.output_dir, args.model)
