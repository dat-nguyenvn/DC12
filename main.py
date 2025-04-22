import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
import time
import random
import argparse
import time
import vpi
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import json
import natsort
import imageio
import random
import pandas as pd
import csv
import argparse

from sahi.utils.yolov8 import (download_yolov8s_model, download_yolov8s_seg_model)
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from natsort import natsorted
from wildlive.generate_data_dynamic_detect import *
from wildlive.ultilkenya import *
from wildlive.utils import *
from wildlive.instance_segmentation import *
from wildlive.instance_segmentation.first_frame import filter_init_detection

from wildlive.points_selection import *
from wildlive.visualization import *
from wildlive.matching import matching_module,calculate_average_distance,unique_id_not_in_matched_box,center_of_box_detected
from wildlive.reconstruct import reconstruct_process
from wildlive.utils.utils import compute_centroid, check_box_overlap,need_add_id_and_point,need_increase_point_for_id,load_config
from wildlive.utils.save import save_in_step
from wildlive.videosource import rtsp_stream,input_folder,videosourceprovider
from wildlive.ultilkenya import filter_dict_main_overlap_box,draw_window_and_center
from wildlive.evaluation.generate_predict import generate_to_mot_format
from wildlive.utils.window_detect import *
#from jetson_utils import videoSource, videoOutput
#import jetson.utils

parser = argparse.ArgumentParser(description="DC12 parses arguments with defaults.")


# Add only the config file argument
parser.add_argument("--config", type=str, default='./config/default.yaml', help="Path to the configuration file (YAML).")
args = parser.parse_args()
config = load_config(args.config)

n_window=config['n_window']
ecl_dis_match=config['ecl_dis_match']
point_not_inmask=config['point_not_inmask']
if config['length_run']==0:
    abcd = sorted(
        [f for f in os.listdir(config['input_fordel_path']) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    #image_files = natsorted(image_files)
    length_run=len(abcd)-2
else:
    length_run=config['length_run'] 
thesshold_area_each_animal=config['thesshold_area_each_animal']  #gan thi 90000 =300*300
save_video_dir=config['save_video_dir'] #'/home/src/yolo/ultralytics/demo_data/demo_test.mp4'
save_folder=config['save_folder'] #'/home/src/yolo/ultralytics/demo_data/demo2/'
save_window_path=config['save_window_path'] #'/home/src/yolo/ultralytics/demo_data/window/'
input_fordel_path=config['input_fordel_path']
model =YOLO(config['model_detection']) # YOLO('yolov8n-seg.pt')
sahimodel= config['model_detection'].split(".")[0]+".pt"
save_folder_small='./demo_data/small/'
save_mot=config['save_mot']
video_name = [part for part in config['input_fordel_path'].split("/") if part.startswith("DJI")][0]
trim_model_name=config['model_detection'].split(".")[0]

#DJI_0133_video1 : con huou
#DJI_0601_video1
#DJI_0117_video4 herd
#DJI_20230720075532_0007_V_video2 : clear 4 zebra
#input_fordel_path="./demo_data/jenna/"
#top view DJI_20230719145427_0002_V_video2 
#inputsource= rtsp_stream(rtsp_link="rtsp://192.168.144.25:8554/main.264")
#output = jetson.utils.videoOutput("rtsp://192.168.144.23:1234/output")

#inputsource= input_folder(args.input_fordel_path)

#inputsource= rtsp_stream(rtsp_link="rtsp://aaa:aaa@192.168.137.195:8554/streaming/live/1")
print("config['input_fordel_path']",config['input_fordel_path'])
inputsource= videosourceprovider(config['input_fordel_path'])

w,h,c=inputsource.frame_size()
im=inputsource.get_RGBframe_numpy()
idFrame=inputsource.index()



result_main=init_detection(im, sahimodel)
trackpoint_list_tuple,id_list_intrack,history_point_inmask,list_dict_info_main,show_image = process_first_frame(result_main)
list_dict_info_main,trackpoint_list_tuple,id_list_intrack,history_point_inmask=process_boxes_complete_step_init(list_dict_info_main,id_list_intrack,trackpoint_list_tuple,w,h,history_point_inmask)

show_image=visual_image().draw_info_from_main_dict(show_image,list_dict_info_main)

selector = StrategySelector()
strategy = selector.get_strategy(n_window)
center_window_list,border_center_point,salient_center_point=generate_centers().generate_tile_centers_border_and_salient(w,h)
predict_mot=[]
remove_dict={}
speed=[]
with vpi.Backend.CPU:
    frame = vpi.asimage(im, vpi.Format.RGB8).convert(vpi.Format.U8)
points_np = np.array(trackpoint_list_tuple, dtype=np.float32)
curFeatures = vpi.asarray(points_np)
with vpi.Backend.CUDA:
    optflow = vpi.OpticalFlowPyrLK(frame, curFeatures,5)

while True:
    start_time = time.time()
    line_mot=generate_to_mot_format(idFrame,list_dict_info_main,id_list_intrack)
    predict_mot.append(line_mot)
    print(idFrame)
    prevFeatures = curFeatures
    cvFrame=inputsource.get_frame()
    if idFrame >= length_run:
        print("Video ended.")
        break
    idFrame =inputsource.index()
    with vpi.Backend.CUDA:
        frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures, status = optflow(frame)
    list_of_tuples = [tuple(map(int, row)) for row in curFeatures.cpu().tolist()]
    rgb_image = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)

    idx_list_need_remove_status=check_live_info().check_and_find_remove_list(sta=status,history_point=history_point_inmask,threshold_point_not_inmask=point_not_inmask,cur_point= curFeatures.cpu(),ID_list=id_list_intrack,dict_inside=list_dict_info_main)

    row_at_the_edge=np.where(np.any(curFeatures.cpu() < 5, axis=1))[0].tolist()

    idx_list_need_remove=list(set(idx_list_need_remove_status + row_at_the_edge))
    if len(idx_list_need_remove)>0:
        curFeatures, status,id_list_intrack,history_point_inmask=remove_intrack().apply_remove(idx_list_need_remove,curFeatures,status,id_list_intrack,history_point_inmask)       
        with vpi.Backend.CUDA:
            optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)

    list_dict_info_main=reconstruct_process(show_image,list_dict_info_main,curFeatures.cpu(),id_list_intrack,w,h)

    center_crop,window_color=strategy.execute(idFrame,center_window_list,border_center_point,salient_center_point,[tuple(map(int, row)) for row in curFeatures.cpu().tolist()])
    
    
    window_detection=crop_window(cvFrame,center_crop)

    all_out_detector=model.predict(window_detection,show_boxes=True, save_crop=False ,show_labels=False,show_conf=False,save=False, classes=[20,22,23] ,conf=0.1,imgsz=(640,640))  # [20,22,23][0,1,2,3,5,7]

    history_point_inmask = [x + 1 for x in history_point_inmask]
    
    for m in range (0,n_window):
        box_id2=None
        out_detector=[all_out_detector[m]]
        if out_detector[0].masks!=None:            
            poatabel=matching_module().poa_table(curFeatures.cpu(),out_detector,id_list_intrack,center_crop[m])
            ioutable=matching_module().iou_table(list_dict_info_main,curFeatures.cpu(),out_detector,id_list_intrack,center_crop[m])
            ioutable=matching_module().accumulate_lists(poatabel,ioutable)

            box_id2=matching_module().matching1(id_list_intrack,ioutable)
            list_dict_info_main=update().step_accumulate(list_dict_info_main,out_detector,id_list_intrack,[tuple(map(int, row)) for row in curFeatures.cpu().tolist()],box_id2,center_crop[m],nwin=n_window)
            
            
            save_window=visual_image().draw_all_on_window(out_detector,box_id2,list_dict_info_main,curFeatures.cpu(),center_crop[m],id_list_intrack)

            name="frame_"+str(idFrame)+".jpg"
            name_txt="frame_"+str(idFrame)+".txt"
            file_path = os.path.join(save_window_path, name)
            file_path_txt = os.path.join(save_window_path, name_txt)


            history_point_inmask=check_live_info().find_point_not_in_mask(id_list_intrack,curFeatures.cpu(),box_id2,out_detector,center_crop[m],history_point=history_point_inmask) # remove points in belong to same object
            dict_id_need_increase_point=need_increase_point_for_id(id_list_intrack,box_id2)
            if len(dict_id_need_increase_point)>0:
                list_dict_info_main,curFeatures,id_list_intrack,history_point_inmask=add_points().apply_add_process_need_more_points(dict_id_need_increase_point,show_image,list_dict_info_main,box_id2,out_detector,center_crop[m],curFeatures.cpu(),id_list_intrack,history_point_inmask)
                with vpi.Backend.CUDA:
                    curFeatures=vpi.asarray(curFeatures)
                    optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)

            list_dict_info_main=update().step_update_detected_bbox_to_main_dict(list_dict_info_main,out_detector,id_list_intrack,[tuple(map(int, row)) for row in curFeatures.cpu().tolist()],box_id2,center_crop[m])
        if need_add_id_and_point(box_id2):
            list_dict_info_main,curFeatures,id_list_intrack,history_point_inmask=add_points().apply_add_process_new_id(show_image,w,h,list_dict_info_main,box_id2,out_detector,center_crop[m],curFeatures.cpu(),id_list_intrack,history_point_inmask,thesshold_area_each_animal)
            with vpi.Backend.CUDA:
                curFeatures=vpi.asarray(curFeatures)
                optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)


    ##### END #####
    ################command ###########
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps=int(1/elapsed_time)
    speed.append(1/elapsed_time)
    text_fps='FPS :' + str(fps)
    text_image_size= 'Resolution: '+ str(w) + 'x' + str(h) 
    print(f"Elapsed time: {elapsed_time:.5f} seconds")
    show_end_image=visual_image().visual_bounding_box_of_dict(list_dict_info_main,rgb_image,id_list_intrack)
    show_end_image=visual_image().draw_pixels_with_colors(show_end_image,curFeatures.cpu(),id_list_intrack,list_dict_info_main)
    small_text_image=visual_image().visual_bounding_box_of_dict(list_dict_info_main,rgb_image,id_list_intrack,fontscale=1)
    small_text_image=visual_image().draw_pixels_with_colors(small_text_image,curFeatures.cpu(),id_list_intrack,list_dict_info_main,radius=5)
    show_end_image=visual_image().add_text_with_background(show_end_image,text_fps)
    show_end_image=visual_image().add_text_with_background(show_end_image,text_image_size, position=(w-400,10) )
    name="frame_"+str(idFrame)+".jpg"
    file_path = os.path.join(save_folder, name)
    plt.imsave(file_path, show_end_image)
    ################command ###########

flattened_data = [item for sublist in predict_mot for item in sublist]
csv_filename = save_mot +video_name+'.csv'
print("csv directory",csv_filename)
os.makedirs(save_mot, exist_ok=True)

mot_df = pd.DataFrame(flattened_data, columns=["frame", "id", "x", "y", "w", "h", "confidence", "class_id","abc","bcd"])
mot_df.to_csv(csv_filename, index=False,header=False)

average = sum(speed) / len(speed)

speed_data = {
    "folder_name": [video_name],
    "gpu":[torch.cuda.get_device_name()],
    "total_frames": [length_run],
    "elapsed_time": [elapsed_time],
    "fps": [average]
}

speed_output_dir = os.path.join(save_mot, "speed")
os.makedirs(speed_output_dir, exist_ok=True)
speed_csv = os.path.join(speed_output_dir, f"{video_name}.csv")
speed_df = pd.DataFrame(speed_data)
speed_df.to_csv(speed_csv, index=True,header=True)

print("Average speed",average)
generate_video().create_video_from_images(save_folder,save_video_dir)
print(f"Video saved: {save_video_dir}")




