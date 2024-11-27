import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
from wildtracker.visualization.visual import visual_image
from wildtracker.utils.convert import convert_process
from wildtracker.points_selection.harris import harris_selection_method
from wildtracker.utils.utils import compute_centroid, check_box_overlap,need_add_id_and_point
from wildtracker.utils.update import update
class add_points():
    def add_to_feature(self,featurecpuin,new_points):
        # if isinstance(featurecpuin, vpi.Array):
        #     with featurecpuin.lock_cpu() as feacpu:
        #         copy_fea = feacpu

        print("featurecpuin",type(featurecpuin))
        points_array = np.array(new_points)
        print("points_array",points_array.shape)
        print("featurecpu",featurecpuin.shape)
        combined_array = np.vstack((featurecpuin, points_array))
        print("combined_array",combined_array.shape)
        combined_array = combined_array.astype(np.float32)
        #cur_Features = vpi.asarray(combined_array)
        return combined_array
    
    def add_to_history(self,history,number_points_add):
        history += [0] * number_points_add
        return history
    
    def add_to_id_intrack(self,trackinglist,newid,number_points_add):

        trackinglist+=[newid]*number_points_add
        print("insdie tracking list",trackinglist)

        return trackinglist
    
    def getnumpyimage_from_yolo(self,yolo_detector,box_target):
        window_np=yolo_detector[0].orig_img
        box=yolo_detector[0].boxes.xywh[box_target]
        x_center, y_center, width, height = box.cpu().numpy()
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Ensure the coordinates are within the image boundaries
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, window_np.shape[0])  # width of the image
        y_max = min(y_max, window_np.shape[1])  

        cropped_image = window_np[y_min:y_max, x_min:x_max]
        #yolo_detector[0].masks
        mask = Polygon(yolo_detector[0].masks.xy[box_target])

        return cropped_image,mask,x_min,y_min,x_center,y_center,width,height


    def apply_add_process_new_id(self,rgb_image,dict_inside,matched_box,yolo_detector,centerwindow,featurecpu,trackinglist,history,thesshold_area_of_animal,threshold_conf=0.3):
        
        for idx,value in enumerate(matched_box):
            if value==0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_conf :
                conf=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
                image_np,polygon,minx,miny,xcen,ycen,wid,hei=self.getnumpyimage_from_yolo(yolo_detector,idx)
                converted_box=convert_process().convert_bounding_boxes_to_big_frame(np.array([[minx, miny, wid, hei]]),centerwindow,(640,640))
                dummy= check_box_overlap(converted_box[0],dict_inside)
                five_points=harris_selection_method().filter_some_points(image_np,polygon,minx,miny)

                if len(five_points)>0 and wid * hei>thesshold_area_of_animal and dummy:
                    five_points_in_window=convert_process().convert_points_box_to_full_frame(five_points,minx,miny)
                    five_points_in_full_frame=convert_process().convert_point_window_to_full_frame(five_points_in_window,centerwindow)
                    convert_minx_miny=convert_process().convert_point_window_to_full_frame([(minx,miny)],centerwindow)
                    number_id_exist=len(dict_inside)
                    newid=number_id_exist+1

                    print("1 five_points_in_full_frame",five_points_in_full_frame)
                    centroid=compute_centroid(five_points_in_full_frame)

                    featurecpu=self.add_to_feature(featurecpu,five_points_in_full_frame)
                    history=self.add_to_history(history,5)
                    trackinglist=self.add_to_id_intrack(trackinglist,newid,5)
                    dict_inside=update().update_list_dict_info(dict_inside,newid,[convert_minx_miny[0][0],convert_minx_miny[0][1],wid,hei],centroid,five_points_in_full_frame,conf)



        return dict_inside,featurecpu,trackinglist,history

    def apply_add_process_need_more_points(self,dictid_need_increase_point,rgb_image,dict_inside,matched_box,yolo_detector,centerwindow,featurecpu,trackinglist,history):

        for idx,value in enumerate(matched_box):
            if value!=0:
                if value in dictid_need_increase_point.keys():

                    image_np,polygon,minx,miny,xcen,ycen,wid,hei=self.getnumpyimage_from_yolo(yolo_detector,idx)
                    
                    print("dictid_need_increase_point[value]^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",dictid_need_increase_point[value])
                    five_points=harris_selection_method().filter_some_points(image_np,polygon,minx,miny,number_point_per_animal=dictid_need_increase_point[value])
                    if len(five_points)>0:
                        five_points_in_window=convert_process().convert_points_box_to_full_frame(five_points,minx,miny)
                        five_points_in_full_frame=convert_process().convert_point_window_to_full_frame(five_points_in_window,centerwindow)
                        convert_minx_miny=convert_process().convert_point_window_to_full_frame([(minx,miny)],centerwindow)
                        #number_id_exist=len(dict_inside)
                        id_need_add=value

                        print("1 five_points_in_full_frame",five_points_in_full_frame)
                        #centroid=compute_centroid(five_points_in_full_frame)

                        featurecpu=self.add_to_feature(featurecpu,five_points_in_full_frame)
                        history=self.add_to_history(history,dictid_need_increase_point[value])
                        trackinglist=self.add_to_id_intrack(trackinglist,id_need_add,dictid_need_increase_point[value])
                        #dict_inside=update().update_list_dict_info(dict_inside,id_need_add,(convert_minx_miny[0][0],convert_minx_miny[0][1],wid,hei),centroid,five_points_in_full_frame)



        return dict_inside,featurecpu,trackinglist,history
