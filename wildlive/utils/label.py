import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
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

import json
from wildlive.utils.crop import crop_bbox
from wildlive.points_selection.apply_method import apply_processing
from wildlive.visualization.visual import visual_image
from wildlive.utils.convert import convert_process,convert_list_dict_to_dict
from wildlive.utils.utils import generate_high_contrast_colors

from natsort import natsorted 


def init_detection(img,model_name,output_folder,name):

    # input_fordel_path=input_path
    yolov8_seg_model_path = model_name
    # im = read_image(input_fordel_path+"frame_0.jpg")
    # h = im.shape[0]
    # w = im.shape[1]

    detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_seg_model_path,
    confidence_threshold=0.1,
    device="cuda", # or 'cuda:0'
    )
    # result2 = get_prediction(img, detection_model_seg, full_shape=(2160,3840))
    # result2.export_visuals(export_dir="demo_data/yoloonly.jpg")

    result = get_sliced_prediction(
        img,
        detection_model_seg,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2)
    
    result.export_visuals(export_dir="demo_data/"+output_folder+"/visual/"+name)
    #rgbarray=np.array(result.image)
    object_prediction_list = result.object_prediction_list
    result_coco_format=result.to_coco_annotations()
    #result_coco_format=convert_list_dict_to_dict(result_coco_format)
    #print("result_coco_format",result_coco_format)
    result_coco_format=convert_xlabel(result_coco_format,name)
    return result_coco_format
def convert_xlabel(sahi_data,image_name):
    xlabeling_data = {
    "version": "2.5.1",
    "flags": {},
    "shapes": [],
    "imagePath": image_name,
    "imageData": None,
    "imageHeight": 2160,
    "imageWidth": 3840,
    "description":"" 
    }

    for idx, entry in enumerate(sahi_data):
        x, y, w, h = entry["bbox"]
        label = entry["category_name"]
        
        shape = {
            "kie_linking": [],
            "label": label,
            "score": entry.get("score", None),
            "points": [
                [x, y],  
                [x + w, y],  
                [x + w, y + h],  
                [x, y + h]  
            ],
            "group_id": idx + 1,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {}
        }
        
        xlabeling_data["shapes"].append(shape)

    return xlabeling_data

def main(folder_name):
    
    #folder_name='DJI_0117_video1'
    folder_path='/DC12/demo_data/' + folder_name +'/frames/'
    
    out_path="/DC12/demo_data/" +folder_name +"/label/"
    #visual_path="/data/captest/capture/"+folder_name+"/visual_sahi/"

    os.makedirs(out_path, exist_ok=True)
    #os.makedirs(visual_path, exist_ok=True)
    model_path='yolo11x.pt'
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    images = natsorted(images)
    for image in images:
        #print("visual_image_path",visual_image_path)
        image_path = os.path.join(folder_path, image)
        img_bgr=cv2.imread(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data=init_detection(img,model_path,folder_name,image)
        new_filename = os.path.splitext(image)[0] + ".json"
        filename=os.path.join(out_path,new_filename)
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        visual_image_path = os.path.join(out_path, image)
        cv2.imwrite(visual_image_path, img_bgr)
        print("visual_image_path",visual_image_path)
    return None

if __name__ == "__main__":
    folderlist=[
    #"DJI_0117_video5",
    "nature",
    ]
    for k in folderlist:
        main(k)