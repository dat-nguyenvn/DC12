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
import matplotlib.pyplot as plt
import os
import json
import natsort
import imageio
import random

import argparse

from sahi.utils.yolov8 import (download_yolov8s_model, download_yolov8s_seg_model)
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

from wildtracker.generate_data_dynamic_detect import *
from wildtracker.ultilkenya import *
from wildtracker.utils import *
from wildtracker.instance_segmentation import *
from wildtracker.points_selection import *
from wildtracker.visualization import *
from wildtracker.matching import matching_module,calculate_average_distance,unique_id_not_in_matched_box,center_of_box_detected
from wildtracker.reconstruct import reconstruct_process
from wildtracker.utils.utils import compute_centroid, check_box_overlap,need_add_id_and_point,need_increase_point_for_id


parser = argparse.ArgumentParser(description="DC12 parses arguments with defaults.")

# Add arguments with default values
parser.add_argument("--input_fordel_path", type=str, default='/home/src/data/captest/capture/DJI_20230720075532_0007_V_video2/frames/', help="Your name (default: 'User')")
parser.add_argument("--save_folder", type=str, default='/home/src/yolo/ultralytics/demo_data/demo2/', help="Your age (default: 18)")
parser.add_argument("--save_video_dir", type=str, default='/home/src/yolo/ultralytics/demo_data/demo_test.mp4', help="Your age (default: 18)")
parser.add_argument("--save_window_path", type=str, default='/home/src/yolo/ultralytics/demo_data/window/', help="Your age (default: 18)")
parser.add_argument("--model_detection", type=str, default='yolov8x-seg.pt', help="[yolov8n-seg.pt, yolov8x-seg.pt,yolov8m-seg.pt, yolov8n-seg.engine]")
parser.add_argument("--length_run", type=int, default=100, help="Your age (default: 18)")
parser.add_argument("--point_not_inmask", type=int, default=200, help="Your age (default: 18)")
parser.add_argument("--ecl_dis_match", type=int, default=10, help="Your age (default: 18)")
parser.add_argument("--thesshold_area_each_animal", type=int, default=1000, help="Your age (default: 18)")


# Parse the arguments
args = parser.parse_args()


ecl_dis_match=args.ecl_dis_match
point_not_inmask=args.point_not_inmask
length_run=args.length_run#
thesshold_area_each_animal=args.thesshold_area_each_animal  #gan thi 90000 =300*300
save_video_dir=args.save_video_dir #'/home/src/yolo/ultralytics/demo_data/demo_test.mp4'
save_folder=args.save_folder #'/home/src/yolo/ultralytics/demo_data/demo2/'
save_window_path=args.save_window_path #'/home/src/yolo/ultralytics/demo_data/window/'
input_fordel_path=args.input_fordel_path #'/home/src/data/captest/capture/DJI_0117_video4/frames/'
model =YOLO(args.model_detection) # YOLO('yolov8n-seg.pt')
#DJI_0133_video1 : con huou
#DJI_0601_video1
#DJI_0117_video4 herd
#input_fordel_path="/home/src/yolo/ultralytics/demo_data/jenna/"


result_main=init_detection(input_fordel_path)
trackpoint_list_tuple,id_list_intrack,history_point_inmask,list_dict_info_main,show_image = process_first_frame(result_main)
list_dict_info_main=process_boxes_complete_step_init(list_dict_info_main,id_list_intrack,trackpoint_list_tuple)

show_image=visual_image().draw_info_from_main_dict(show_image,list_dict_info_main)
plt.imshow(show_image)
plt.show()
im = read_image(input_fordel_path+"frame_0.jpg")
h = im.shape[0]
w = im.shape[1]
center_window_list,border_center_point,salient_center_point=generate_centers().generate_tile_centers_border_and_salient(w,h)

# im_win='/home/src/yolo/ultralytics/demo_data/demo2/frame_29.jpg'
# im_win=cv2.imread(im_win)
# im_win = cv2.cvtColor(im_win, cv2.COLOR_BGR2RGB)
# for k in border_center_point:

#     win_vi=visual_image().draw_window(im_win,k,color=(255, 255, 0))
    


# for k in salient_center_point:

#     win_vi=visual_image().draw_window(im_win,k,color=(0, 255, 0))
#     plt.imsave('window.jpg', win_vi)

# plt.imshow(win_vi)
# plt.show()

idFrame=0
remove_dict={}
speed=[]
with vpi.Backend.CPU:
    frame = vpi.asimage(im, vpi.Format.BGR8).convert(vpi.Format.U8)
points_np = np.array(trackpoint_list_tuple, dtype=np.float32)
curFeatures = vpi.asarray(points_np)
with vpi.Backend.CUDA:
    optflow = vpi.OpticalFlowPyrLK(frame, curFeatures,5)

#start_time = time.time()

while True:
    start_time = time.time()
    print(idFrame)
    prevFeatures = curFeatures
    #if idFrame >= len(images_list)-1:
    if idFrame >= length_run:
        print("Video ended.")
        break
    idFrame += 1
    path=input_fordel_path+"frame_"+str(idFrame)+".jpg"
    cvFrame=cv2.imread(path)
    with vpi.Backend.CUDA:
        frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures, status = optflow(frame)
    list_of_tuples = [tuple(map(int, row)) for row in curFeatures.cpu().tolist()]
    rgb_image = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
    #show_image=visualize_points_on_image(image_np=rgb_image,points=list_of_tuples)
    # plt.imshow(show_image)
    # plt.show()



    idx_list_need_remove_status=check_live_info().check_and_find_remove_list(sta=status,history_point=history_point_inmask,threshold_point_not_inmask=point_not_inmask)
    row_at_the_edge=np.where(np.any(curFeatures.cpu() < 5, axis=1))[0].tolist()
    if row_at_the_edge!=[]:
        print("row_at_the_edge",row_at_the_edge)
    row_at_the_edge=[]    
    idx_list_need_remove=list(set(idx_list_need_remove_status + row_at_the_edge))
    if len(idx_list_need_remove)>0:
        curFeatures, status,id_list_intrack,history_point_inmask=remove_intrack().apply_remove(idx_list_need_remove,curFeatures,status,id_list_intrack,history_point_inmask)
        with vpi.Backend.CUDA:
            optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)

    #list_dict_info_main=filter_dict_main_overlap_box(list_dict_info_main)
    list_dict_info_main=reconstruct_process(show_image,list_dict_info_main,curFeatures.cpu(),id_list_intrack)

    #list_dict_info_main=reconstruct_process(show_image,list_dict_info_main,curFeatures.cpu(),id_list_intrack)

    # show_image=visual_image().draw_info_from_main_dict(rgb_image,list_dict_info_main)
    # plt.imshow(show_image)
    # plt.show()

    center_crop,window_color=strategy_pick_window(idFrame,center_window_list,border_center_point,salient_center_point,list_of_tuples)
    window_detection=crop_window(cvFrame,center_crop)
    out_detector=model.predict(window_detection,show_boxes=True, save_crop=False ,show_labels=False,show_conf=False,save=False, classes=[20,22,23],conf=0.3,imgsz=(640,640))

    history_point_inmask = [x + 1 for x in history_point_inmask]
    #print("out_detector[0]",out_detector[0])
    box_id2=None
    if out_detector[0].masks!=None:
        #print("out_detector[0].masks",out_detector[0].masks.xy)   #list array n points,and 2 (x,y)
        #print("out_detector[0]", out_detector[0].boxes.xywh.cpu().numpy().shape)
        history_point_inmask=update().history_point_mask(points=curFeatures.cpu(),history_points_in_mask=history_point_inmask,yolo_detector=out_detector,center_window=center_crop)
        poatabel=matching_module().poa_table(curFeatures.cpu(),out_detector,id_list_intrack,center_crop)
        

        box_id=matching_module().matching1(id_list_intrack,poatabel)
        box_id2=matching_module().matching2(box_id,id_list_intrack,out_detector,center_crop,curFeatures.cpu(),threshold_ecl_dis_match=ecl_dis_match)
        print("outttttttttt .boxes.conf.cpu().numpy()[idx]",out_detector[0].boxes.conf.cpu().numpy().shape)


        list_dict_info_main=update().step_accumulate(list_dict_info_main,out_detector,id_list_intrack,list_of_tuples,box_id2,center_crop)
        list_dict_info_main=update().update_bounding_box_based_on_eq4(list_dict_info_main,out_detector,id_list_intrack,list_of_tuples,box_id2,center_crop)

        save_window=visual_image().draw_all_on_window(out_detector,box_id2,list_dict_info_main,curFeatures.cpu(),center_crop,id_list_intrack)
        # plt.imshow(save_window)
        # plt.show()

        name="frame_"+str(idFrame)+".jpg"
        name_txt="frame_"+str(idFrame)+".txt"
        file_path = os.path.join(save_window_path, name)
        plt.imsave(file_path, save_window)
        file_path_txt = os.path.join(save_window_path, name_txt)
        save_2d_list_to_txt(poatabel,file_path_txt)


        idx_list_need_remove=matching_module().filter_poa_table(id_list_intrack,poatabel)

        if len(idx_list_need_remove)>0:
            curFeatures, status,id_list_intrack,history_point_inmask=remove_intrack().apply_remove(idx_list_need_remove,curFeatures,status,id_list_intrack,history_point_inmask)
            with vpi.Backend.CUDA:
                optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)

        dict_id_need_increase_point=need_increase_point_for_id(id_list_intrack,box_id2)
        if len(dict_id_need_increase_point)>0:
            list_dict_info_main,curFeatures,id_list_intrack,history_point_inmask=add_points().apply_add_process_need_more_points(dict_id_need_increase_point,show_image,list_dict_info_main,box_id2,out_detector,center_crop,curFeatures.cpu(),id_list_intrack,history_point_inmask)
            
            
            with vpi.Backend.CUDA:
                curFeatures=vpi.asarray(curFeatures)
                optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)


    if need_add_id_and_point(box_id2):
        list_dict_info_main,curFeatures,id_list_intrack,history_point_inmask=add_points().apply_add_process_new_id(show_image,list_dict_info_main,box_id2,out_detector,center_crop,curFeatures.cpu(),id_list_intrack,history_point_inmask,thesshold_area_each_animal)
        
        # print("cur_feature", curFeatures.cpu().shape)
        # print("tracking_list",len(id_list_intrack))
        # print("id_list_intrack",id_list_intrack)
        # print("set track ing list ",set(id_list_intrack))
        
        with vpi.Backend.CUDA:
            curFeatures=vpi.asarray(curFeatures)
            optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 5)


    ##### END #####


    end_time = time.time()
    elapsed_time = end_time - start_time
    fps=int(1/elapsed_time)
    speed.append(fps)
    text_fps='FPS :' + str(fps)
    text_image_size= 'Resolution: '+ str(w) + 'x' + str(h) 
    #text=text_image_size + text_fps
    print(f"Elapsed time: {elapsed_time:.5f} seconds")

    show_end_image=visual_image().visual_bounding_box_of_dict(list_dict_info_main,rgb_image,id_list_intrack)
    show_end_image=visual_image().draw_pixels_with_colors(rgb_image,curFeatures.cpu(),id_list_intrack,list_dict_info_main)
    show_end_image=visual_image().draw_window(rgb_image,center_crop,color=window_color)
    show_end_image=visual_image().add_text_with_background(show_end_image,text_fps)
    show_end_image=visual_image().add_text_with_background(show_end_image,text_image_size, position=(w-1500,10) )

    # plt.imshow(show_end_image)
    # plt.show()    

    name="frame_"+str(idFrame)+".jpg"
    
    file_path = os.path.join(save_folder, name)
    plt.imsave(file_path, show_end_image)

average = sum(speed) / len(speed)
print("average",average)
generate_video().create_video_from_images(save_folder,save_video_dir)



    # rgb_image = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
    # list_of_tuples = [tuple(map(int, row)) for row in curFeatures.cpu().tolist()]
    # show_image=visualize_points_on_image(image_np=rgb_image,points=list_of_tuples)
    # list_polygon=numpy_to_shapely_polygons(out_detector[0].masks.xy)

    # for one_polygon in list_polygon:
    #     #print("one_polygon",one_polygon)
    #     one_polygon=convert_polygon_to_big_frame(one_polygon,center_crop)
    #     show_image=visualize_shapely_polygon_on_image(show_image,one_polygon)
    







