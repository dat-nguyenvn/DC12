import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import random
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import time
import cv2
import queue
import threading
from shapely.geometry import Polygon, Point
from collections import Counter
from ultralytics import YOLO
from utils_DC12.object import Animal,Zoo
# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future
from cotracker.models.build_cotracker import build_cotracker
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
def make_parse():
    parser=argparse.ArgumentParser("DC12 Demo !")
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        #default='/home/src/yolo/DJI_20230607090521_0006_V.mp4',
        #default='/home/src/yolo/trimvideo.mp4',
        #default='/home/src/yolo/lowresolution.mp4',
        #default='/home/src/yolo/DJI_20230607092838_0003_V.mp4',
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default='./checkpoints/cotracker2.pth',
        #default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--save",
        default=False,
        help="Save frame and track points",
    )    
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=20,
        help="Compute dense and grid tracks starting from this frame",
    )

    return parser
def _process_step(window_frames, is_first_step, grid_size, grid_query_frame,quetest,model):
    video_chunk = (
        torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return model(
        video_chunk,
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        #queries=quetest[None],
    )
def main(args):
    queries= torch.tensor([[0,300.,300.],
                        [0,400.,400.],
                        [0,500.,500.],
                        [0,450.,450.],
                        [0,250.,250.]]) 
    queries=queries.cuda()
    is_first_step=True
    
    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
        #model = build_cotracker(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to('cuda')
    window_frames = []
    dummy_input = torch.randn(16, 3, 384, 512)
    pred_tracks, pred_visibility = _process_step(
        dummy_input,
        is_first_step,            # if queries_new.shape[0]>200:
    #     num_rows_to_keep = queries_new.size(0) - 20 
    #     indices_to_keep = random.sample(range(queries_new.size(0)), num_rows_to_keep)
    #     queries_new=queries_new[indices_to_keep]
        grid_size=args.grid_size,
        #grid_query_frame= queries[None],
        grid_query_frame= args.grid_query_frame,
        quetest=queries,
        model=model,
    )
    window_frames = []    
    #predictor = CoTrackerPredictor(checkpoint)

    #All inputs should be resized to 384x512
    dummy_input = torch.randn(1, 8, 3, 384, 512, device="cuda")

    # We take a video and queried points as input
    input_names = ["input_video", "input_queries"]
    output_names = ["output_tracks", "output_feature", "output_visib", "output_metadata"]

    # Video length is also dynamic
    dynamic_axes_dict = {
        'input_video': {
            0: 'batch_size',
            1: 'video_len'
        },
        'input_queries': {
            0: 'batch_size',
            1: 'video_len'
        },
    } 
    
    torch.onnx.export(model,
                    dummy_input,
                    "cotracker.onnx",
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes_dict,
                    export_params=True,
                    )
    print("end")
if __name__ == "__main__":
    args=make_parse().parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Total time",end_time-start_time)