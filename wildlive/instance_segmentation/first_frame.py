import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
import matplotlib.pyplot as plt
from random import randint
from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict


from wildlive.utils.crop import crop_bbox
from wildlive.points_selection.apply_method import apply_processing
from wildlive.visualization.visual import visual_image
from wildlive.utils.convert import convert_process,convert_list_dict_to_dict
from wildlive.utils.utils import generate_high_contrast_colors
from wildlive.ultilkenya import check_live_info
from wildlive.utils.remove import remove_intrack

def init_detection(img,model_name):

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
    
    result.export_visuals(export_dir="demo_data/yolo+sahi.jpg")
    rgbarray=np.array(result.image)
    object_prediction_list = result.object_prediction_list
    return result

def filter_init_detection(result):
    object_prediction_list = result.object_prediction_list
    print("aaaaaaaaaaasss",result)

    # filtered_detections = []
    
    # for detection in object_prediction_list:
    #     # Check if bounding box or segmentation mask exists for each detection
    #     # Assuming 'bbox' represents the bounding box and 'mask' represents the segmentation mask
    #     bbox = detection.get('bbox', None)
    #     #mask = detection.get('mask', None)
        
    #     # You can adjust the conditions based on your output format
    #     if len(bbox) ==4:
    #         # Only append detections with both bbox and mask
    #         filtered_detections.append(detection)
    
    return None
    


def process_first_frame(result):

    id=1
    id_list_intrack=[]
    trackpoint_LK=[]


    rgbarray=np.array(result.image)
    object_prediction_list = result.object_prediction_list

    count=0
    remove_detect_list=[]
    for ob1 in object_prediction_list:
        #print("count",count)
        croped=crop_bbox(rgbarray,[ob1.bbox.minx,ob1.bbox.miny,ob1.bbox.maxx,ob1.bbox.maxy])
        coords=ob1.mask.segmentation[0]
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

        polygon_in_window = Polygon(points) 

        # # start slide here
        # transformed_polygon=convert_process().transform_polygon_window_to_box(polygon_in_window,ob1.bbox.minx,ob1.bbox.miny)
        # show_image=visual_image().visualize_shapely_polygon_on_image(croped,transformed_polygon)
        # plt.imshow(show_image)
        # plt.show()
        # # end slide
   
        

        #points_one_object=apply_processing('harris',croped,polygon,ob1.bbox.minx,ob1.bbox.miny)
        points_one_object=apply_processing('harris',croped,polygon_in_window,ob1.bbox.minx,ob1.bbox.miny)
        show_image=visual_image().visualize_points_on_image(image_np=croped,points=points_one_object)
        
        #visual_image().show_image(show_image)



        # slide here
        # show_image=harris_selection_method().visual_result(croped)
        # plt.imshow(show_image)
        # plt.show()
        # end slide
        print("aasss!!!!!!",ob1.bbox)
        if len(points_one_object)>0 :
            point_sample=points_one_object
            #print("point_sample",point_sample)
            id_list_intrack += [id]*len(point_sample)
            original_points=convert_process().convert_points_box_to_full_frame(point_sample,ob1.bbox.minx,ob1.bbox.miny)
            #print("original_points",original_points)
            trackpoint_LK+=original_points
            #show_image=visual_image().visualize_points_on_image(rgbarray,original_points)

            id+=1
        else:
            remove_detect_list.append(count)

        count+=1

    list_dict_info=result.to_coco_annotations()
    dict_filtered = [list_dict_info[i] for i in range(len(list_dict_info)) if i not in remove_detect_list]

    history_points_tracked_list=[0] * len(id_list_intrack)

    show_image=visual_image().visualize_points_on_image(rgbarray,trackpoint_LK)
    
    plt.imshow(show_image)
    plt.show()
    print("Done process_first_frame")
    return trackpoint_LK,id_list_intrack,history_points_tracked_list,dict_filtered,show_image


def dict_id_center(tracking_list, in_points):
    """
    Calculate the centers of points grouped by their tracking IDs.

    Parameters:
    - tracking_list: A list of IDs corresponding to each point.
    - points: A list of tuples, where each tuple contains (x, y) coordinates of a point.

    Returns:
    - A dictionary where keys are IDs and values are the centers of the points corresponding to each ID.
    """
    # Initialize an empty dictionary to store centers
    centers_dict = {}
    point_of_id_dict= {}

    # Iterate over unique IDs in the tracking list
    unique_ids = set(tracking_list)
    for id_ in unique_ids:
        #print("len points",len(in_points))
        #print("len(tracking_list)",len(tracking_list))

        # Find all points corresponding to the current ID
        id_points = [in_points[i] for i in range(len(tracking_list)) if tracking_list[i] == id_]
        
        # Calculate the mean (center) of the points
        center = np.mean(id_points, axis=0)
        
        # Store the center in the dictionary
        centers_dict[id_] = tuple(center)
        point_of_id_dict[id_]= id_points
        #print("id_points", id_points)

    return centers_dict,point_of_id_dict


def process_boxes_complete_step_init(list_dict_info,tracking_list,points,wid,hei,history_point):

    '''
    result: out put of sahi
    
    #output is list of dict 
       {1: {'image_id': None,
    'bbox': [447.6741943359375,
    309.5724792480469,
    48.0478515625,
    32.496734619140625],
    'score': 0.885761022567749,
    'category_id': 2,
    'category_name': 'car',
    'segmentation': [],
    'iscrowd': 0,
    'area': 1561}  }

            new_id_dict= {'image_id': newid,
                        'bbox': bbox, # [x,y,w,h]
                        'score': 0, #float
                        'category_id': None, #int
                        'category_name': None, #str
                        'segmentation': [],
                        'iscrowd': 0,
                        'area': 0,
                        'ori_bbox': bbox, #list [x,y,w,h],
                        'ori_center_points': groupid_center, #(x,y)
                        'color': (randint(0, 255),randint(0, 255),randint(0, 255)),
                        'lastest_point':None,
                        'disappear_step':None ,   
                        'points_step_t':list_tuple_five_points
                        drift:(0,0)
    '''



    id_center_dict,point_of_id_dict=dict_id_center(tracking_list,points)
    print("list_dict_info",list_dict_info)
    
    for idx,value in enumerate(list_dict_info):
        

        value['image_id']=idx+1
        value['bbox']=value['bbox']
        value['ori_center_points']=id_center_dict[idx+1]
        value['ori_bbox']=value['bbox']
        value['color']=generate_high_contrast_colors()
        value['lastest_point']=None
        value['disappear_step']=None
        value['points_step_t']=point_of_id_dict[idx+1]
        value['drift']=(0,0)
        if value['score']<0.8:
            value['visible']=False
        if value['score']>0.8:
            value['visible']=True
        value['image width']=wid
        value['image height']=hei
    
    #print("len(list_dict_info)",len(list_dict_info))
    list_dict_info=convert_list_dict_to_dict(list_dict_info)
    rm_list=check_live_info().check_main_dict_by_id(list_dict_info)
    list_dict_info_out=remove_intrack().remove_key_in_dict(list_dict_info,rm_list)
    out_point,out_id_list,out_history=remove_intrack().apply_remove_first_step(rm_list,points,tracking_list,history_point)



    #print(list_dict_info.keys())
    #print("list_dict_info aaaaaaaaaaa",list_dict_info[2].keys())
    
    
    return list_dict_info_out,out_point,out_id_list,out_history