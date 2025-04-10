import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
import matplotlib.pyplot as plt
from random import randint


from wildlive.utils.convert import convert_process
from wildlive.ultilkenya import iou



class matching_module():
    def poa(self,points, polygon):
        #points: list of tuple
        #polygon : shapely polygon
        count = 0

        # Iterate over each point from the Harris corner detector
        for point in points:
            # Create a Shapely Point object
            shapely_point = Point(point)

            # Check if the point is inside the polygon
            if polygon.contains(shapely_point):
                count += 1
        poa_value=count/len(points)


        return poa_value,count

    def poa_table(self,points,yolo_detector,tracking_list,center_window):
        #poa table: row ~ mask /
        # each row : value poa of group ID

        unique_values = set(tracking_list)
        
        # Get the number of unique values
        number_unique_id = len(unique_values)
        #print("number_unique_id",number_unique_id)


        #poa_table is list of list , each row is list belong to a mask
        poa_table=[]
        for r in yolo_detector:
            # TODO something: visula polygon detected on cropped frame
            # orig_img=r.orig_img
            # print("type orig_img",type(orig_img))
            # point_list_of_tuples = [tuple(map(int, row)) for row in points]
            # print("point_list_of_tuples",point_list_of_tuples)
            # show_image=visualize_shapely_polygon_on_image(orig_img,point_list_of_tuples)
            # plt.imshow(show_image)
            # plt.show()

            if len(r.masks.xy)>0:
                for mask_count in range (len(r.masks.xy)):
                    mask = Polygon(r.masks.xy[mask_count])
                    #mask_buffer = mask.buffer(distance=-5)
                    mask=convert_process().convert_polygon_window_to_full_frame(mask,center_window)

                    poa_per_mask=[]
                    #mask to shapely

                    for unique in unique_values:
                        indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
                        point_of_one_id=[tuple(points[i]) for i in indices_of_unique]
                        poa_value, count= self.poa(point_of_one_id,mask)
                        #print("poa_value",poa_value)
                        poa_per_mask.append(poa_value)
                        #TODO: kill point not in mask 
                        #need_remove(indices_of_unique)

                    #print("poa_per_mask",poa_per_mask)
                    poa_table.append(poa_per_mask)

            #     mask_poa.append(poa)
            # poa_table.append(mask_poa)

        return poa_table

    def iou_table(self,dict_inside,points,yolo_detector,tracking_list,center_window):
        #poa table: row ~ mask /
        # each row : value poa of group ID

        unique_values = set(tracking_list)
        
        # Get the number of unique values
        number_unique_id = len(unique_values)
        #print("number_unique_id",number_unique_id)


        #poa_table is list of list , each row is list belong to a mask
        iou_table=[]
        for r in yolo_detector:
            # TODO something: visula polygon detected on cropped frame
            # orig_img=r.orig_img
            # print("type orig_img",type(orig_img))
            # point_list_of_tuples = [tuple(map(int, row)) for row in points]
            # print("point_list_of_tuples",point_list_of_tuples)
            # show_image=visualize_shapely_polygon_on_image(orig_img,point_list_of_tuples)
            # plt.imshow(show_image)
            # plt.show()

            if len(r.masks.xy)>0:
                for box_count in range (r.boxes.xyxy.cpu().numpy().shape[0]):
                    #mask = Polygon(r.masks.xy[mask_count])
                    box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[box_count]
                    box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)
                    box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]



                    iou_per_box=[]
                    #mask to shapely

                    for unique in unique_values:
                        indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
                        point_of_one_id=[tuple(points[i]) for i in indices_of_unique]
                        pre_box=dict_inside[unique]['bbox']
                        iou_value=iou(pre_box,box_yolo) 

                        
                        #print("poa_value",poa_value)
                        iou_per_box.append(iou_value)
     

                    #print("poa_per_mask",poa_per_mask)
                    iou_table.append(iou_per_box)

            #     mask_poa.append(poa)
            # poa_table.append(mask_poa)

        return iou_table
    

    def matching3 (self,tracking_list,iou_table,match_box_id,iou_thresshold=0.7):
        #match iou thresshold
        unique_values = list(set(tracking_list))
        number_bbox=len(match_box_id)



        return match_box_id
    
    def matching1 (self,tracking_list,poa_table): # First association
        #poa_table=self.poa_table
        unique_values = list(set(tracking_list))
        number_bbox=len(poa_table)
        #print("number_bbox",number_bbox)
        match_box_id=[0]*number_bbox  # 0 mean unmatch to any id; init step
        for idx_mask,poa_per_mask in enumerate(poa_table):
            if poa_per_mask:
                #print("poa_table",poa_table)
                #print("poa_per_mask",poa_per_mask)
                best_match_index=poa_per_mask.index(max(poa_per_mask))
                #print("best_match_index",best_match_index)

                #todo: threshold 0.4 ? + .index(max(poa_per_mask))\
                #  la tim vi tri dau tien cua gia tri max, \
                # neu co 2 index = nhau co loi nhe
                #todo check heare
                if poa_per_mask[best_match_index]>=0.1 : 
                    match_box_id[idx_mask]=unique_values[best_match_index]


        #print("match_box_id",match_box_id)
        return match_box_id
    
    def accumulate_lists(self,list1, list2):
        """
        Add values at corresponding positions in two lists of lists.
        
        :param list1: First list of lists.
        :param list2: Second list of lists.
        :return: A new list of lists with summed values.
        """
        if len(list1) != len(list2) or any(len(row1) != len(row2) for row1, row2 in zip(list1, list2)):
            raise ValueError("Both lists must have the same dimensions.")
        
        # Add corresponding elements
        return [[v1 + v2 for v1, v2 in zip(row1, row2)] for row1, row2 in zip(list1, list2)]

    def filter_poa_table (self,tracking_list,poa_table): # remove points in belong to same object
        #poa_table=self.poa_table
        unique_values = list(set(tracking_list))
        number_bbox=len(poa_table)
        #print("number_bbox",number_bbox)
        match_box_id=[0]*number_bbox  # 0 mean unmatch to any id; init step
        for idx_mask,poa_per_mask in enumerate(poa_table):
            #indexes = [i for i, value in enumerate(poa_per_mask) if value == max(poa_per_mask)]

            indexes = [i for i, value in enumerate(poa_per_mask) if value ==1]
            if len(indexes) >1:
                id_list=[unique_values[i] for i in indexes]
                id_list.remove(min(id_list))

                remove_indexes=[i for i, value in enumerate(tracking_list) if value in id_list]
                return remove_indexes
        #     #best_match_index=poa_per_mask.index(max(poa_per_mask))
        #     #print("best_match_index",best_match_index)

        #     #todo: threshold 0.4 ? + .index(max(poa_per_mask))\
        #     #  la tim vi tri dau tien cua gia tri max, \
        #     # neu co 2 index = nhau co loi nhe
        #     #todo check heare
        #     if poa_per_mask[best_match_index]>=0.4 : 
        #         match_box_id[idx_mask]=unique_values[best_match_index]

            else:
                return []
        # print("match_box_id",match_box_id)
        



    def matching2(self,match_box_id,tracking_list,yolo_detector,centercrop,featurecpu,threshold_ecl_dis_match=50):
        #unmatched_bbox_list_idx=find_indices_unmatched1(match_box_id)
        #unique_values = set(tracking_list)
        exist_id=unique_id_not_in_matched_box(match_box_id,tracking_list)
        exist_id=list(set(tracking_list))
        list_tuple_center_of_box_position_in_window=center_of_box_detected(yolo_detector)
        list_tuple_center_of_box_position_in_full_frame=convert_process().convert_point_window_to_full_frame(list_tuple_center_of_box_position_in_window,centercrop)
        points = [tuple(map(int, row)) for row in featurecpu.tolist()]
      
        #[0,5]
        match_box_id_after_matching1=match_box_id
        for idx,value in enumerate(match_box_id_after_matching1):
            if value==0:
                average_ecl_distances_of_groupid_to_center_box= [] #for one box
                centerbox=list_tuple_center_of_box_position_in_full_frame[idx]

                for unique in exist_id:
                    #if unique not in match_box_id:
                    indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
                    point_of_one_id=[tuple(points[i]) for i in indices_of_unique]
                    id_to_center_distance=calculate_average_distance(point_of_one_id,centerbox)
                    average_ecl_distances_of_groupid_to_center_box.append(id_to_center_distance)
                    if max(average_ecl_distances_of_groupid_to_center_box)<threshold_ecl_dis_match:
                        best_match_index=average_ecl_distances_of_groupid_to_center_box.index(max(average_ecl_distances_of_groupid_to_center_box))
                        
                
                        match_box_id[idx]=exist_id[best_match_index]
        
    




        return match_box_id
    



def calculate_average_distance(points, center_box):
    """
    Calculate the average Euclidean distance of all points to the center_box.

    Parameters:
    - points: List of tuples [(x1, y1), (x2, y2), ...].
    - center_box: Tuple (x_center, y_center) representing the center of the box.

    Returns:
    - average_distance: The average Euclidean distance of the points to the center_box.
    """
    total_distance = 0
    for point in points:
        x, y = point
        x_center, y_center = center_box
        distance = np.sqrt((x_center - x)**2 + (y_center - y)**2)
        total_distance += distance
    
    # Calculate the average distance
    average_distance = total_distance / len(points) if points else 0
    
    return average_distance


def unique_id_not_in_matched_box(matched_box, b):
    # Convert list a to a set for faster lookup
    #set_a = set(matched_box)
    
    # Use set to remove duplicates from b and filter out values present in set_a
    result = [value for value in set(b) if value not in matched_box]
    
    return result

def center_of_box_detected(yolo_detector):
    for r in yolo_detector: 
        center=r.boxes.xywh.cpu().numpy() #numpy (n,4) : n is number of box, 4 ls xywh : center of box + w+h
        tuple_list = [tuple(row[:2]) for row in center] 
    return tuple_list  #list of tuple (x,y) center of bboxes
