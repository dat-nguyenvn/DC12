import cv2
import numpy as np

import matplotlib.pyplot as plt
from random import randint
from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)
import os
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from tqdm import tqdm 
import json
import argparse
from yolox.tracker.byte_tracker import BYTETracker

import torch
import pandas as pd

import time 
from natsort import natsorted 


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-idx", "--index", type=int, default=0)
    parser.add_argument("-n", "--model", type=str, default="yolov8x-seg.pt", help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--data_dir", default="/DC12/demo_data/eval/videos/", help="path to images or video"
    )
    parser.add_argument("--output_dir", type=str, default="/DC12/demo_data/eval", help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

#tracker = BYTETracker(frame_rate=30)  # Adjust frame rate


def detect_objects_sahi(img,model_name,conf=0.1):

    # input_fordel_path=input_path
    yolov8_seg_model_path = model_name
    # im = read_image(input_fordel_path+"frame_0.jpg")
    # h = im.shape[0]
    # w = im.shape[1]

    detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_seg_model_path,
    confidence_threshold=conf,
    device="cuda", # or 'cuda:0'
    )
    # result2 = get_prediction(img, detection_model_seg, full_shape=(2160,3840))
    # result2.export_visuals(export_dir="demo_data/yoloonly.jpg")
    try:
        result = get_sliced_prediction(
            img,
            detection_model_seg,
            slice_height = 640,
            slice_width = 640,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2)
    
        detections = []
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            conf = obj.score.value
            class_id = obj.category.id
            detections.append([x1, y1, x2, y2, conf, class_id])
        
        return np.array(detections)

    except Exception as e:
        print(f"❌ Error in get_sliced_prediction: {e}")
        return []

def track_objects_from_folder(tracker,name,image_folder, output_dir,model):
    """
    Runs SAHI detection + ByteTrack tracking on images from a folder.
    """
    # Get sorted list of images
    image_folder=os.path.join(image_folder, "frames")



    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    image_files=natsorted(image_files)
    tracking_results = []
    total_frame=len(image_files)
    start_time = time.time()
    for frame_id, img_name in tqdm(enumerate(image_files), total=len(image_files), desc="Processing Frames"):
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Warning: Could not read {img_name}. Skipping...")
            continue

        
        detections = detect_objects_sahi(image,model_name=model)  # Run SAHI detection
        img_info = (image.shape[0], image.shape[1])  # (height, width)
        img_size = (image.shape[0], image.shape[1]) 
        # Convert detections to ByteTrack format
        if len(detections) > 0:
            tracked_objects = tracker.update(torch.tensor(detections),img_info,img_size)
        else:
            tracked_objects = []

        # Save tracking results
        for track in tracked_objects:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            w = x2 - x1
            h = y2 - y1
            tracking_results.append([frame_id, track_id, x1, y1, w, h])

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = total_frame / elapsed_time
    print(f"Frame {frame_id}, FPS: {fps:.2f}")
    
    speed_output_dir = os.path.join(output_dir, "speed")
    gpu_model=torch.cuda.get_device_name()
    os.makedirs(speed_output_dir, exist_ok=True)
    speed_csv = os.path.join(speed_output_dir, f"{folder_name}.csv")
    

    speed_data = {
        "folder_name": [folder_name],
        "gpu":[gpu_model],
        "total_frames": [total_frame],
        "elapsed_time": [elapsed_time],
        "fps": [fps]
    }
    speed_df = pd.DataFrame(speed_data)
    speed_df.to_csv(speed_csv, index=True,header=True)

    # Save tracking results as CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{folder_name}.csv")
    df = pd.DataFrame(tracking_results, columns=["frame", "track_id", "x1", "y1", "w", "h"])
    df.to_csv(output_csv, index=False,header=False)
    print(f"✅ Tracking results saved: {output_csv}")

if __name__ == "__main__":
    # folder_names_to_process = ['DJI_20230719145427_0002_V_video5', 'DJI_20230719145427_0002_V_video3', 'DJI_0601_video2', 
    #                            'DJI_0207', 'DJI_0119', 'DJI_0133_video3', 'DJI_20230719145427_0002_V_video4', 'DJI_0601_video3', 
    #                            'DJI_0142_video1', 'DJI_0161_video1', 'vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V', 
    #                            'DJI_0601_video4', 'DJI_0204_video2', 'DJI_0204_video1', 'DJI_0117_video4', 'DJI_0117_video3', 'DJI_0133_video1']
    
    folder_names_to_process = [
        "DJI_20230719145427_0002_V_video5", "DJI_20230719145427_0002_V_video3", "DJI_0601_video2",
        "DJI_0207", "DJI_0119", "DJI_0133_video3", "DJI_20230719145427_0002_V_video4",
        "DJI_0601_video3", "DJI_0142_video1", "DJI_0161_video1",
        "vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V",
        "DJI_0601_video4", "DJI_0204_video2", "DJI_0204_video1",
        "DJI_0117_video4", "DJI_0117_video3", "DJI_0133_video1",'DJI_0601_video5','DJI_0117_video1',
        'DJI_0117_video2','DJI_0133_video1 ','DJI_0601_video6','DJI_20230719145427_0002_V_video1',
        'DJI_20230719145427_0002_V_video2','DJI_20230719145816_0003_V_video2','DJI_20230720075532_0007_V_video2']
    args = make_parser().parse_args()

    if 0 <= args.index < len(folder_names_to_process):
        folder_name = folder_names_to_process[args.index]
        folder_path = os.path.join(args.data_dir, folder_name)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "frames")):
            #im_folder=os.path.join(folder_path, "frames")
            print(f"Processing folder: {folder_name}")
            tracker_method = BYTETracker(args)  # Adjust frame rate
            track_objects_from_folder(tracker_method,folder_name,folder_path, args.output_dir,args.model)
   
    # Example usage
    # image_folder = "/DC12/demo_data/eval/videos/vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V/frames"  # Change to your folder
    # output_dir = "/DC12/demo_data/eval/sahi-byte/yolov8x-seg"
    # track_objects_from_folder(image_folder, output_dir)