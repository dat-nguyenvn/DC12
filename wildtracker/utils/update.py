import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
import matplotlib.pyplot as plt
from random import randint


from wildtracker.utils.convert import convert_process




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
    def update_list_dict_info(self,list_dict_info,newid, bbox,groupid_center,list_tuple_five_points,conf,threshold_conf=0.8):
        #use for new ID

        if conf>threshold_conf:
            new_id_dict= {'image_id': newid,
                            'bbox': bbox, # [x,y,w,h]
                            'score': conf, #float
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
                            'points_step_t':list_tuple_five_points,
                            'visible':True, 
                            'drift':(0,0)
                            
                            }
        else: 
            new_id_dict= {'image_id': newid,
                'bbox': bbox, # [x,y,w,h]
                'score': conf, #float
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
                'points_step_t':list_tuple_five_points,
                'visible':False ,
                'drift':(0,0)
                }

        #list_dict_info.append(new_id_dict) #backup
        list_dict_info[newid]=new_id_dict
        return list_dict_info


    def step_accumulate(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.8):
        for idx,value in enumerate(match_box_id):
            print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_box_conf:# and is_not_box_at_edge(yolo_detector[0].boxes.xywh.cpu().numpy()[idx]):
                #tuc la deteced box matched, va co co   nf cao thi minh se update box
                box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
                print("box_yolo",box_yolo.shape)
                box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)

                box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]
                print("box_yolo ************* ",box_yolo)
                dict_inside[value]['visible']=True
                




                dict_inside[value]['bbox']=box_yolo
                dict_inside[value]['ori_bbox']=box_yolo
                dict_inside[value]['ori_center_points']=self.update_points_group_center(value,tracking_list,points)
                dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
            if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]<threshold_box_conf:
                dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
                if dict_inside[value]['score']>0.8:
                    dict_inside[value]['visible']=True

                

            

            


        dict_inside=self.decrease_score_step(dict_inside)

        return dict_inside


    # def step_box_update(dict_inside,out_yolo,match_box_id,threshold_box_conf=0.8):
    #     idx,value in enumerate(
    #     dict_inside=update_bbox()

    #     return dict_inside

    def update_points_group_center(self,id_,tracking_list,points):

        id_points = [points[i] for i in range(len(tracking_list)) if tracking_list[i] == id_]
        
        # Calculate the mean (center) of the points
        center = np.mean(id_points, axis=0)
        
        # Store the center in the dictionary
        return center
    def decrease_score_step(self,inside_dict):
        for key, value in inside_dict.items():
            if 'score' in value:  # Check if 'score' exists
                value['score'] -= 0.01

        return inside_dict

    def predict_box_based_equ4(self,driftx,drifty,x_topleft_previous,y_topleft_previous,x_detect,y_detect,tau=4):



        x_pre_topleft=driftx +x_topleft_previous +tau*x_detect
        y_pre_topleft=drifty +y_topleft_previous +tau*y_detect
        return (x_pre_topleft,y_pre_topleft)

    def update_bounding_box_based_on_eq4(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.5):
        for idx,value in enumerate(match_box_id):
            print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
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
                    dict_inside[value]['bbox']=list(dict_inside[value]['bbox'])
                else:
                    print("dict_inside[value]['bbox']",dict_inside[value]['bbox'])
                    new_box=dict_inside[value]['bbox']
                    new_box[0]=predict_basedeq4[0]
                    new_box[1]=predict_basedeq4[1]
                    dict_inside[value]['bbox']=new_box


        return dict_inside
