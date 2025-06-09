import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
import matplotlib.pyplot as plt
from random import randint


from wildlive.utils.convert import convert_process
from wildlive.utils.utils import compute_centroid,generate_high_contrast_colors,is_not_box_at_edge
from wildlive.instance_segmentation.first_frame import dict_id_center

animals_categories = {
    20: "elephant",
    22: "zebra",
    23: "giraffe"
}
class update():
    def history_point_mask(self,points,history_points_in_mask,yolo_detector,center_window):
        
        r=yolo_detector[0]
        #result = []
        if len(yolo_detector[0].masks.xy)>0:
            for idx, pixel in enumerate(points):
                point = Point(pixel)

                for mask_count in range (len(r.masks.xy)):
                    mask = Polygon(r.masks.xy[mask_count])
                    #mask_buffer = mask.buffer(distance=-2)
                    mask=convert_process().convert_polygon_window_to_full_frame(mask,center_window)
                    if mask.contains(point):
                        history_points_in_mask[idx]=0

                        # result.append(idx)

        # if len(result) != len(set(result)):
        #     result=remove_duplicates(result)
        #     print("Duplicates in history_points_tracked")
        
        # else:
        #     for i in range(len(history_points_in_mask)):  # Iterate over each index in the history list
        #         if i in result:  # If the index is in the specified indices list
        #             history_points_in_mask[i] = 0  # Set the value to 0
        #         else:
        #             history_points_in_mask[i] += 1

        return history_points_in_mask
    def update_list_dict_info(self,list_dict_info,newid, bbox,groupid_center,list_tuple_five_points,conf,wid,hei,class_num,threshold_conf=0.8):
        #use for new ID
        #dummy=box_at_edge(bbox,wid,hei)
        class_category=animals_categories.get(class_num)
        if conf>threshold_conf:
            new_id_dict= {'image_id': newid,
                            'bbox': bbox, # [x,y,w,h]
                            'score': conf, #float
                            'category_id': class_num, #int
                            'category_name': class_category, #str
                            'segmentation': [],
                            'iscrowd': 0,
                            'area': 0,
                            'ori_bbox': bbox, #list [x,y,w,h],
                            'ori_center_points': groupid_center, #(x,y)
                            'color':generate_high_contrast_colors(),
                            'lastest_point':None,
                            'disappear_step':None ,   
                            'points_step_t':list_tuple_five_points,
                            'visible':True, 
                            'drift':(0,0),
                            'image width': wid,
                            'image height': hei
                            
                            }
        #if conf<threshold_conf: 
        else:
            new_id_dict= {'image_id': newid,
                'bbox': bbox, # [x,y,w,h]
                'score': conf, #float
                'category_id': class_num, #int
                'category_name': class_category, #str
                'segmentation': [],
                'iscrowd': 0,
                'area': 0,
                'ori_bbox': bbox, #list [x,y,w,h],
                'ori_center_points': groupid_center, #(x,y)
                'color': generate_high_contrast_colors(),
                'lastest_point':None,
                'disappear_step':None ,   
                'points_step_t':list_tuple_five_points,
                'visible':False ,
                'drift':(0,0),
                'image width': wid,
                'image height': hei
                }

        #list_dict_info.append(new_id_dict) #backup
        list_dict_info[newid]=new_id_dict
        return list_dict_info


    def step_accumulate(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,nwin,window_size,threshold_box_conf=0.5):
        if nwin == 1 or nwin == 2:
            athres=2

        if nwin == 4 or nwin == 8:
            athres= 4
        if nwin == 16 or nwin == 24 :
            athres= 6

        for idx,value in enumerate(match_box_id):
            #print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_box_conf:# and is_not_box_at_edge(yolo_detector[0].boxes.xywh.cpu().numpy()[idx]):
                #tuc la deteced box matched, va co co   nf cao thi minh se update box
                box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
                #print("box_yolo",box_yolo.shape)
                box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)

                box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(window_size,window_size))[0]
                #print("box_yolo ************* ",box_yolo)

                dict_inside[value]['visible']=True
                




                # dict_inside[value]['bbox']=box_yolo
                # dict_inside[value]['ori_bbox']=box_yolo
                # dict_inside[value]['ori_center_points']=self.update_points_group_center(value,tracking_list,points)
                dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]

            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]<threshold_box_conf:
                dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
                if dict_inside[value]['score']>=athres:
                    dict_inside[value]['visible']=True
                elif dict_inside[value]['score']<athres:  
                    dict_inside[value]['visible']=False
          


        dict_inside=self.decrease_score_step(dict_inside)

        return dict_inside


    def step_update_detected_bbox_to_main_dict(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.8):
        for idx,value in enumerate(match_box_id):
            #print("yolo_detector[0]",yolo_detector[0])
            #print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_box_conf:
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
                if is_not_box_at_edge(box_yolo):
                #if True:
                    box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)
                    box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]
                    dict_inside[value]['bbox']=box_yolo
                    dict_inside[value]['ori_bbox']=box_yolo
                    id_center_dict,point_of_id_dict=dict_id_center(tracking_list,points)

                    #dict_inside[value]['ori_center_points']=self.update_points_group_center(value,tracking_list,points)
                    dict_inside[value]['ori_center_points']=id_center_dict[value]
        

        return dict_inside

    def update_points_group_center(self,id_,tracking_list,points):

        id_points = [points[i] for i in range(len(tracking_list)) if tracking_list[i] == id_]
        
        # Calculate the mean (center) of the points
        center = np.mean(id_points, axis=0)
        
        # Store the center in the dictionary
        return tuple(center)
    def decrease_score_step(self,inside_dict):
        for key, value in inside_dict.items():
            if 'score' in value:  # Check if 'score' exists
                value['score'] -= 0.01

        return inside_dict

    def predict_box_based_equ4(self,driftx,drifty,x_topleft_previous,y_topleft_previous,x_detect,y_detect,tau=4):



        # x_pre_topleft=(driftx +x_topleft_previous +tau*x_detect)/(1+tau)
        # y_pre_topleft=(drifty +y_topleft_previous +tau*y_detect)/(1+tau)
        x_pre_topleft=x_detect
        y_pre_topleft=y_detect

        return (x_pre_topleft,y_pre_topleft)

    def update_bounding_box_based_on_eq4(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.5):
        for idx,value in enumerate(match_box_id):
            #print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]<threshold_box_conf:
                
                drix=dict_inside[value]['drift'][0]
                driy=dict_inside[value]['drift'][1]
                x_pre_top=dict_inside[value]['bbox'][0]
                y_pre_top=dict_inside[value]['bbox'][1]
                box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
               
                box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)

                box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]
   
                predict_basedeq4=self.predict_box_based_equ4(drix,driy,x_pre_top,y_pre_top,box_yolo[0],box_yolo[1])
                
                if type(dict_inside[value]['bbox']) is tuple:
                    #dict_inside[value]['ori_bbox']=list(dict_inside[value]['bbox'])
                    dict_inside[value]['bbox']=list(dict_inside[value]['bbox'])
                    
                else:
                    #print("dict_inside[value]['bbox']",dict_inside[value]['bbox'])
                    new_box=dict_inside[value]['bbox']
                    new_box[0]=predict_basedeq4[0]
                    new_box[1]=predict_basedeq4[1]
                    dict_inside[value]['bbox']=new_box
                    #dict_inside[value]['ori_bbox']=new_box


        return dict_inside


def update_bounding_box(center_points_t0, points_t1, box_t):
    """
    Update the bounding box based on points at time step t and t+1.

    Parameters:
    - points_t: List of points [(x1, y1), (x2, y2), ...] at time t.
    - points_t1: List of points [(x1, y1), (x2, y2), ...] at time t+1.
    - box_t: Original bounding box (x_center, y_center, w, h) at time t.

    Returns:
    - new_box: Updated bounding box [x_center, y_center, w, h] at time t+1.
    """
    # Get original bounding box center and size
    x_center_t, y_center_t, w, h = box_t
    
    # Compute centroids of points at time t and time t+1
    centroid_t = center_points_t0
    centroid_t1 = compute_centroid(points_t1)
    
    # Compute shift in centroid position
    shift_x = centroid_t1[0] - centroid_t[0]
    shift_y = centroid_t1[1] - centroid_t[1]
    
    # Update bounding box center with the shift
    new_x_center = x_center_t + shift_x
    new_y_center = y_center_t + shift_y
    
    # Return the new bounding box with the same width and height
    return [new_x_center, new_y_center, w, h],(shift_x,shift_y)
    
class CheckInfoDict():
    """
    A class to filter and process a dictionary of tracking information.
    """

    def __init__(self):
        """
        Initializes the CheckInfoDict class.
        """
        pass

    def _calculate_modified_iou(self,box1, box2):
        # Extract top-left corner and width/height for both boxes
        x1_1, y1_1, w1, h1 = box1  # Box 1
        x1_2, y1_2, w2, h2 = box2  # Box 2
        
        # Calculate the bottom-right corner for both boxes
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        
        # Calculate the area of intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # If there's no intersection, return IoU as 0
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate the area of each box
        area_1 = w1 * h1
        area_2 = w2 * h2

        # Calculate the minimum area of the two boxes
        min_area = min(area_1, area_2)

        # Calculate the Modified IoU
        modified_iou = inter_area / min_area
        return modified_iou


    def _calculate_iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        Boxes are in [x, y, w, h] format.
        """
        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Determine the coordinates of the intersection rectangle
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # Calculate the area of the intersection rectangle
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        # Calculate the area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calculate the union area
        union_area = box1_area + box2_area - intersection_area

        # Handle the case where union_area is zero to avoid division by zero
        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _filter_by_id(self, data_dict, id_list_intrack):
        """
        Filter 1: Filters the main dictionary to include only items whose keys
        (image_ids) are present in the id_list_intrack.

        Args:
            data_dict (dict): The main dictionary containing tracking info.
            id_list_intrack (list): A list of image_ids that are currently in track.

        Returns:
            dict: A new dictionary containing only the filtered items.
        """
        # Using a dictionary comprehension to create a new filtered dictionary
        filtered_data = {k: data_dict[k] for k in id_list_intrack if k in data_dict}
        return filtered_data

    def _filter_overlapping_boxes(self, data_dict, iou_threshold=0.95):
        """
        Filter 2: Identifies and removes entries from the dictionary where bounding
        boxes overlap by more than a specified IoU threshold.
        When overlap occurs, the entry with the lower 'score' is removed.

        Args:
            data_dict (dict): The dictionary of tracking information (e.g., output from filter1).
                              This dictionary will be modified in-place if items are removed.
            iou_threshold (float): The Intersection over Union (IoU) threshold.
                                   If IoU > this, one of the overlapping boxes is removed.

        Returns:
            dict: A new dictionary with overlapping boxes removed.
        """
        # Create a list of (id, bbox, score) tuples to easily iterate and compare
        # Using list() to create a copy of items to avoid issues while modifying dict
        items_to_check = list(data_dict.items())
        ids_to_remove = set()

        # Iterate through all unique pairs of items
        for i in range(len(items_to_check)):
            id_i, info_i = items_to_check[i]
            bbox_i = info_i['bbox']
            score_i = info_i['score']

            if id_i in ids_to_remove: # Skip if this item is already marked for removal
                continue

            for j in range(i + 1, len(items_to_check)):
                id_j, info_j = items_to_check[j]
                bbox_j = info_j['bbox']
                score_j = info_j['score']

                if id_j in ids_to_remove: # Skip if this item is already marked for removal
                    continue

                iou = self._calculate_modified_iou(bbox_i, bbox_j)

                if iou > iou_threshold:
                    # Overlap detected, decide which one to remove
                    if score_i < score_j and score_i <0.4 :
                        ids_to_remove.add(id_i)
                        break # No need to compare id_i with other boxes if it's being removed
                    if score_j <= score_i and score_j <0.4 : # score_j <= score_i (including equal scores, remove id_j)
                        ids_to_remove.add(id_j)

        # Create a new dictionary without the marked-for-removal IDs
        filtered_data = {k: v for k, v in data_dict.items() if k not in ids_to_remove}
        return filtered_data,ids_to_remove

    def apply(self, list_dict_info_main, id_list_intrack):
        """
        Applies filter1 and filter2 to the tracking information.

        Args:
            list_dict_info_main (dict): The main dictionary where all tracking data is stored.
            id_list_intrack (list): A list of image_ids that are currently in track.

        Returns:
            dict: The final dictionary after applying both filters.
        """
        print(f"Initial items in list_dict_info_main: {len(list_dict_info_main)}")
        print(f"IDs in id_list_intrack: {len(id_list_intrack)}")

        # --- Apply Filter 1 ---
        # Filters to keep only items whose IDs are in id_list_intrack
        print("\nApplying Filter 1: Filtering by in-track IDs...")
        data_after_filter1 = self._filter_by_id(list_dict_info_main, id_list_intrack)
        print(f"Items after Filter 1: {len(data_after_filter1)}")

        # --- Apply Filter 2 ---
        # Removes overlapping bounding boxes, keeping the one with the higher score
        print("\nApplying Filter 2: Removing overlapping boxes (IoU > 0.5)...")
        final_filtered_data,ids_to_remove = self._filter_overlapping_boxes(data_after_filter1, iou_threshold=0.5)
        print(f"Items after Filter 2: {len(final_filtered_data)}")

        return final_filtered_data,ids_to_remove