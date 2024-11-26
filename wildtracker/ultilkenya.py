'''
#TODO: 
# class : convert 
def polygon_window_to_full_frame
def box_window_to_full_frame
def points_full_frame_to window (chua co)
def points_box_to window
def points_window_to full_frame


class remove
def remove feature.cpu
def remove status
def remove id_list_intrack
def remove_history_old point
def process (remove all)

class visual_image
def show
def draw_points_on_np_array
def draw_polygon_on_np_array
def draw_box_on_np_array
def save_image

class animal:
id
class : zebra
bbox: x,y,w,h
polygon: polygon
color:()
points in full frame : [()] list of tuple
visible: yes/no
last_point




class zoo():
dict ['id']= animal
def update

class add_point (coban la append)
def add_feature.cpu
def add_id_list_track
def add_history old_point
def add to Zoo

def process : run all add above

class point_info
position (x,y)
id: 
history:
status_lk

class update:
    def update zoo (last point,)
    def history_point_in mask


class live_manager_points:
    def check_len 
        feature.cpu = status= id_list_intrack=history
        return feature.cpu = status= id_list_intrack=history

    def  check a ID of class Animal: add_point, box, polygon, if need

class matching_module
    def poa
    def matching 1
    def matching 2
    process run all
        return ID_mask coresspond detector


class adding_process
    check which process will run
    def add new id track
        update func
    def add_new_point
        add_point 

    def process 


live_animal_ocl 
killed_animal

def:
check a ID of class Animal: add_point, box, polygon, if need

def reconstruct_bbox
def generate_video

def crop_box

def main:
    init detection
    def process_first_frame 
        return feature.cpu = id_list_intrack=history

    visual all (point, box,polygon)
    show

    init opticalflow
    while True: 
        live_manager_points_check
        pre_process_in_step
        run optflow
            check status 
            remove point
            update func
            

        run detection
        process out_detector if need (exist object)
            update history_point_in mask
            matching_module
        
        check adding process    
        adding_process

        recontruct box
        visual all  

        
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon, Point,MultiPolygon
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
from random import randint

from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image


from abc import ABC, abstractmethod

from wildtracker.generate_data_dynamic_detect import *
from wildtracker.utils.crop import crop_bbox
from wildtracker.utils.convert import convert_process


# def crop_bbox(image_np, bbox):
#     """
#     Crops the image using the bounding box coordinates.

#     Parameters:
#     - image_np: NumPy array of the image (H, W, C)
#     - bbox: list or tuple of bounding box [minx, miny, maxx, maxy]

#     Returns:
#     - Cropped image as a NumPy array
#     """
#     minx, miny, maxx, maxy = bbox
    
#     # Crop the image using array slicing
#     cropped_image = image_np[miny:maxy, minx:maxx]

#     return cropped_image





# class selected_point_method(ABC):
#     @abstractmethod
#     def process(self, cropped_image_numpy):
#         pass
#     def visual_result(self,cropped_image_numpy):
#         result_image = cropped_image_numpy.copy()
#         corners_sorted=self.process(cropped_image_numpy)
#         for x, y in corners_sorted:
#             cv2.circle(result_image, (x, y), 3, (0, 0, 255), 2)  # Red circles for corners
         
#         return result_image
#     def transform_polygon_window_to_box(self,polygon, xmin, ymin):
#         """
#         Transforms a polygon from the original frame to the cropped frame by adjusting coordinates.
        
#         Parameters:
#         - polygon (shapely.geometry.Polygon): The polygon in the original frame.
#         - xmin (int or float): The x-coordinate of the top-left corner of the cropped region.
#         - ymin (int or float): The y-coordinate of the top-left corner of the cropped region.

#         Returns:
#         - transformed_polygon (shapely.geometry.Polygon): The polygon transformed to the cropped frame.
#         """
#         # Get the coordinates of the polygon
#         #print("polygon in trasform",polygon)
#         if isinstance(polygon, MultiPolygon):
#             polygon=polygon.geoms[0]
#             original_coords = list(polygon.exterior.coords)
#             cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]
#             transformed_polygon = Polygon(cropped_coords)
            
#         else:
#             original_coords = list(polygon.exterior.coords)

#             # Transform each point in the polygon to the cropped frame
#             cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]

#             # Create a new polygon with the transformed coordinates
#             transformed_polygon = Polygon(cropped_coords)

#         return transformed_polygon
#     def filter_points_inside_polygon(self,points, polygon):
#         """
#         Filters the list of points to return only those that are inside the given polygon.
        
#         Parameters:
#         - points (list of tuples): List of (x, y) coordinates representing the points.
#         - polygon (shapely.geometry.Polygon): A Shapely Polygon object.

#         Returns:
#         - inside_points (list of tuples): List of (x, y) coordinates that are inside the polygon.
#         """
#         inside_points = []

#         # Iterate over each point and check if it is inside the polygon
#         for point in points:
#             shapely_point = Point(point)
#             if polygon.contains(shapely_point):
#                 inside_points.append(point)
#         #print("inside_points",len(inside_points))
#         return inside_points

#     def filter_some_points(self,cropped_image_numpy,polygon,xmin,ymin,margin=-5,number_point_per_animal=5):
#         '''
#         xmin,ymin : topleft position of window
#         polygon in window
#         xmin,ymin topleft box
#         '''
#         corners_sorted=self.process(cropped_image_numpy)
#         #filter in polygon
#         polygon=polygon.buffer(distance=margin)
#         transformed_polygon=self.transform_polygon_window_to_box(polygon,xmin,ymin)
#         points_inside_polygon=self.filter_points_inside_polygon(corners_sorted,transformed_polygon)
#         try:
#             five_points= random.sample(points_inside_polygon,number_point_per_animal)
#             return five_points
#         except ValueError as e:
#             print(f"Error: {e} Can not pick point in the mask")
#             return []
        



# class harris_selection_method(selected_point_method):
#     def process(self,cropped_image_numpy,blockSize=5, ksize=3, k=0.04, threshold=0.2):

#         gray_image = cv2.cvtColor(cropped_image_numpy, cv2.COLOR_BGR2GRAY)

#         # Convert to float32 for cornerHarris
#         gray_image = np.float32(gray_image)

#         # Apply Harris corner detection
#         dst = cv2.cornerHarris(gray_image, blockSize, ksize, k)

#         # Dilate the result to mark the corners (optional for visualization)
#         dst = cv2.dilate(dst, None)

#         # Copy the original image for visualization
#         result_image = cropped_image_numpy.copy()

#         # Find coordinates of corners by applying the threshold
#         corners = np.argwhere(dst > threshold * dst.max())

#         # Create a list of (x, y, confidence) tuples
#         corners_with_confidence = [(int(x), int(y), dst[y, x]) for y, x in corners]

#         # Sort the corners by confidence (Harris response) from highest to lowest
#         corners_with_confidence.sort(key=lambda x: x[2], reverse=True)

#         # Extract only the (x, y) coordinates, sorted by confidence
#         corners_sorted = [(x, y) for x, y, _ in corners_with_confidence]

#         for x, y in corners_sorted:
#             cv2.circle(result_image, (x, y), 3, (0, 0, 255), 2)  # Red circles for corners
 
#         return corners_sorted


#todo here #############
class orb_selection_method(selected_point_method):
    def process(self,cropped_image_numpy):
        return None

#end todo here ~~~~~~~~~~~~

# def apply_processing(method, cropped_image,polygon,minx,miny):
#     methods = {
#         'harris': harris_selection_method(),
#         'edge_detection': harris_selection_method(),
#         'blur': harris_selection_method()
#     }
#     return methods[method].filter_some_points(cropped_image_numpy=cropped_image, polygon= polygon,xmin=minx,ymin=miny )



# class visual_image():
#     def visualize_shapely_polygon_on_image(self,image_np, shapely_polygon, color=(0, 255, 0), thickness=2):
#         """
#         Visualizes a Shapely polygon on an image by drawing its outline.

#         Parameters:
#         - image_np (np.array): Input image in numpy array format (BGR).
#         - shapely_polygon (shapely.geometry.Polygon): Shapely polygon object.
#         - color (tuple): Color of the polygon outline in BGR format (default is green).
#         - thickness (int): Thickness of the polygon outline (default is 2).

#         Returns:
#         - result_image (np.array): Image with the polygon drawn on it.
#         """
#         # Extract the polygon points and convert to a numpy array of type int32

#         polygon_points = np.array(shapely_polygon.exterior.coords, dtype=np.int32)
        
#         # Reshape the array to the correct format for polylines
#         polygon_points = polygon_points.reshape((-1, 1, 2))

#         # Draw the polygon outline on the image
#         result_image = image_np.copy()
#         cv2.polylines(result_image, [polygon_points], isClosed=True, color=color, thickness=thickness)

#         return result_image
#     def show_image(self,numpy_image):
#         #numpy_image : numpy 3d array 
#         # rgb color
#         plt.imshow(numpy_image)
#         plt.show()

#     def visualize_points_on_image(self,image_np, points, color=(0, 0, 255), radius=5, thickness=-1):
#         """
#         Visualizes points on an image by drawing circles at each point.

#         Parameters:
#         - image_np (np.array): Input image in numpy array format (BGR).
#         - points (list of tuples): List of (x, y) coordinates to visualize.
#         - color (tuple): Color of the points (default is red in BGR format).
#         - radius (int): Radius of the circles representing the points (default is 5).
#         - thickness (int): Thickness of the circle outline. Use -1 for filled circles (default is -1).

#         Returns:
#         - result_image (np.array): Image with the points drawn on it.
#         """
#         # Create a copy of the original image to draw on
#         copy_image = image_np.copy()

#         # Iterate over each point and draw a circle
#         for point in points:
#             #print("*****point",point)
#             #print("copy_image",copy_image.shape)
#             cv2.circle(copy_image, (int(point[0]),int(point[1])), radius, color, thickness)

#         return copy_image
    

#     def draw_bounding_boxes_on_image(self,image, bboxes, color=(0, 255, 0), thickness=2):
#         """
#         Draw bounding boxes on the image.

#         Parameters:
#         - image: The original image (as a NumPy array).
#         - bboxes: List of bounding boxes in the format [(x, y, w, h), ...]. x,y center of box 
#         - color: Color of the bounding box in BGR format.
#         - thickness: Thickness of the bounding box lines.
        
#         Returns:
#         - Image with bounding boxes drawn.
#         """
#         for (x, y, w, h) in bboxes:
#             #print("(x, y, w, h) ",(x, y, w, h) )
#             # Calculate the top-left corner from center (x, y)
#             top_left_x = int(x-w/2)
#             top_left_y = int(y-h/2)
#             bottom_right_x = int(x + w/2)
#             bottom_right_y = int(y + h/2)

#             # Draw the rectangle on the image
#             cv2.rectangle(image, (top_left_x, top_left_y), 
#                         (bottom_right_x, bottom_right_y), 
#                         color, thickness)
        
#         return image    
    
#     def draw_info_from_main_dict(self,np_image,data_list):
#         """
#         Draw bounding boxes on the image based on the data_list.
        
#         Parameters:
#         - image: NumPy array representing the image (BGR format).
#         - data_list: main dict dict of dictionaries with 'bbox' information. Each dictionary contains:
#         'bbox': [x_top_left, y_toplet, width, height].
#         """
#         if type(data_list) is not dict:
#             for item in data_list:
                
#                 bbox = item['bbox']
#                 x_top_left, y_top_left, w, h = bbox
                
#                 # Convert bbox from center (x, y) to top-left (x1, y1) for OpenCV's rectangle function
#                 x1 = int(x_top_left)
#                 y1 = int(y_top_left)
#                 x2 = int(x_top_left + w )
#                 y2 = int(y_top_left + h )
                
#                 # Draw rectangle on the image (blue color with thickness 2)
#                 cv2.rectangle(np_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                
#                 # Optionally, display the category name at the top of the bounding box
#                 category_name = item.get('category_name', 'Unknown')
#                 category_name='animal'
#                 cv2.putText(np_image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             fontScale=0.5, color=(255, 0, 0), thickness=1)      


#         if type(data_list) is dict:
#             for key, item in data_list.items():
                
#                 bbox = item['bbox']
#                 x_top_left, y_top_left, w, h = bbox
                
#                 # Convert bbox from center (x, y) to top-left (x1, y1) for OpenCV's rectangle function
#                 x1 = int(x_top_left)
#                 y1 = int(y_top_left)
#                 x2 = int(x_top_left + w )
#                 y2 = int(y_top_left + h )
                
#                 # Draw rectangle on the image (blue color with thickness 2)
#                 cv2.rectangle(np_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                
#                 # Optionally, display the category name at the top of the bounding box
#                 category_name = item.get('category_name', 'Unknown')
#                 category_name='animal'
#                 cv2.putText(np_image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             fontScale=0.5, color=(255, 0, 0), thickness=1)   

#         return np_image

#     def draw_one_id_on_image(self,image, bbox, image_id, color,conf='0',thickness=10):
#         """
#         Draw a bounding box with a label on the image.
#         Args:
#         - image (np.array): Image to draw on
#         - bbox (list): Bounding box [x, y, w, h] x,y, topleft
#         - image_id (int): The label or ID to display on the bounding box
#         - color (tuple): Color for the bounding box (BGR format)
#         """
#         # Unpack bbox coordinates
#         x, y, w, h = bbox
#         x=int(x)
#         y=int(y)
#         w=int(w)
#         h=int(h)


#         # Draw rectangle (bbox)
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
#         # Text settings
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 2
#         font_thickness = 10
#         text = f'ID: {image_id} ; conf: {conf:.2f}'
        
#         # Calculate text size and position
#         text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#         text_x = x
#         text_y = y - 10 if y - 10 > 0 else y + text_size[1] + 10
        
#         # Draw text (image_id)
#         cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)
#         return image
#     def visual_bounding_box_of_dict(self,dict_inside,np_image,tracking_list):

#         unique_values = set(tracking_list)
#         #print("unique_values in reconstruct ",unique_values)
        
#         for unique in unique_values:
#             '''
#             for item in dict_inside:
#                 if item['image_id'] == unique:
#                     #for obj in dict_inside:
#                     out_img=self.draw_one_id_on_image(np_image, item['bbox'], item['image_id'], item['color'])
#             '''
#             if unique in dict_inside: 
#                 item = dict_inside[unique]
#                 if item['visible']==True:
#                     out_img=self.draw_one_id_on_image(np_image, item['bbox'], item['image_id'], item['color'],item['score'])
#                 elif item['visible']==False:
#                     out_img=self.draw_one_id_on_image(np_image, item['bbox'], ' ' , (211, 211, 211),item['score'])



#         return out_img
#     def draw_pixels_with_colors(self,image, featurecpu, id_list_intrack, new_id_dict_list):
#         """
#         Draws pixels on the image with colors corresponding to their IDs.
        
#         Args:
#             image (np.ndarray): The input image on which to draw pixels.
#             featurecpunumpy array (nx2): representing pixel positions.
#             id_list_intrack (list of int): A list of IDs corresponding to each pixel in pixel_positions.
#             new_id_dict_list (list of dict): A list of dictionaries containing 'image_id' and 'color' fields.
        
#         Returns:
#             np.ndarray: The image with colored pixels.
#         """
#         pixel_positions=convert_process().convert_featurecpu_to_list_tuple(featurecpu)
#         # Create a dictionary to map image_id to color from the list of dictionaries
#         id_to_color = {obj['image_id']: obj['color'] for obj in new_id_dict_list.values()}

#         # Iterate through the pixel positions and corresponding IDs
#         for i, (x, y) in enumerate(pixel_positions):
#             image_id = id_list_intrack[i]  # Get the ID for this pixel position
#             color = id_to_color.get(image_id, (0, 0, 0))  # Get the color for this ID, default to black if not found
            
#             # Draw the pixel by setting the color at the pixel location
#             cv2.circle(image, (x,y), 10, color, -1)
#         return image

#     def draw_all_on_window(self,yolo_output,box_matched,dict_inside,points,window_center,tracking_list):
#         np_img = yolo_output[0].orig_img
#         points_in_window=convert_process().convert_points_from_big_to_window(points,window_center)
#         np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
#         #check if have time 
#         for dummy,value in enumerate(box_matched):
#             #color_id=find_color_by_image_id(dict_inside,value)  #backup
#             if value!=0:

#                 color_id=dict_inside[value]['color']
#                 box=yolo_output[0].boxes.xywh.cpu().numpy()[dummy]

#                 np_img=self.visualize_shapely_polygon_on_image(np_img, Polygon(yolo_output[0].masks.xy[dummy]),color_id,5)
#                 np_img=self.draw_bounding_boxes_on_image(np_img,[box],color_id,1)
#                 box=convert_process().convert_center_to_topleft(box[0],box[1],box[2],box[3])
#                 np_img=self.draw_one_id_on_image(np_img,box,value,color_id,5)
#             if value==0:

#                 color_id=(0,0,0)
#                 box=yolo_output[0].boxes.xywh.cpu().numpy()[dummy]

#                 np_img=self.visualize_shapely_polygon_on_image(np_img, Polygon(yolo_output[0].masks.xy[dummy]),color_id,5)
#                 np_img=self.draw_bounding_boxes_on_image(np_img,[box],color_id,1)
#                 box=convert_process().convert_center_to_topleft(box[0],box[1],box[2],box[3])
#                 np_img=self.draw_one_id_on_image(np_img,box,value,color_id,5)                

#         for idx,point in enumerate(points_in_window):
            
#             value=tracking_list[idx]
#             #color_id=find_color_by_image_id(dict_inside,value) #backup
#             color_id=dict_inside[value]['color']
#             np_img=self.visualize_points_on_image(np_img,[point],color_id,radius=8)

#         #print("np_img************",np_img.shape)

        
        
#         return np_img



#     # def draw_window(self,image, center, size=(640,640)):
#     #     """
#     #     Draws a rectangle on the image given the center and size.
        
#     #     Args:
#     #         image (np.ndarray): The input image on which to draw the rectangle.
#     #         center (tuple): The (x, y) coordinates of the center of the rectangle.
#     #         size (tuple): The (width, height) of the rectangle.
        
#     #     Returns:
#     #         np.ndarray: The image with the drawn rectangle.
#     #     """
#     #     # Unpack center and size
#     #     center_x, center_y = center
#     #     width, height = size

#     #     # Calculate top-left and bottom-right corners
#     #     top_left = (center_x - width // 2, center_y - height // 2)
#     #     bottom_right = (center_x + width // 2, center_y + height // 2)

#     #     # Draw the rectangle on the image
#     #     cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=15)  # Red color

#     #     return image  


#     def draw_window(self,image, center, size=(640,640), dash_length=25, spacing=30,color=(255, 0, 0)):
#         """
#         Draws a dashed rectangle on the image given the center and size.
        
#         Args:
#             image (np.ndarray): The input image on which to draw the dashed rectangle.
#             center (tuple): The (x, y) coordinates of the center of the rectangle.
#             size (tuple): The (width, height) of the rectangle.
#             dash_length (int): The length of each dash.
#             spacing (int): The spacing between dashes.
        
#         Returns:
#             np.ndarray: The image with the drawn dashed rectangle.
#         """
#         # Unpack center and size
#         center_x, center_y = center
#         width, height = size

#         # Calculate top-left and bottom-right corners
#         top_left = (center_x - width // 2, center_y - height // 2)
#         bottom_right = (center_x + width // 2, center_y + height // 2)

#         # Get the coordinates of the rectangle's corners
#         points = [
#             (top_left[0], top_left[1]),                        # Top-left
#             (bottom_right[0], top_left[1]),                    # Top-right
#             (bottom_right[0], bottom_right[1]),                # Bottom-right
#             (top_left[0], bottom_right[1]),                    # Bottom-left
#             (top_left[0], top_left[1])                         # Close the rectangle
#         ]

#         # Draw dashes along each edge
#         for i in range(4):
#             start_point = points[i]
#             end_point = points[i + 1]
            
#             # Calculate the total length of the current edge
#             edge_length = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))

#             # Calculate the number of dashes and spacing
#             num_dashes = edge_length // (dash_length + spacing)

#             for j in range(num_dashes):
#                 # Calculate the starting point of the dash
#                 dash_start_x = int(start_point[0] + j * (dash_length + spacing) * (end_point[0] - start_point[0]) / edge_length)
#                 dash_start_y = int(start_point[1] + j * (dash_length + spacing) * (end_point[1] - start_point[1]) / edge_length)
                
#                 # Calculate the ending point of the dash
#                 dash_end_x = int(dash_start_x + (dash_length * (end_point[0] - start_point[0]) / edge_length))
#                 dash_end_y = int(dash_start_y + (dash_length * (end_point[1] - start_point[1]) / edge_length))
                
#                 # Draw the dash
#                 cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color=color, thickness=15)

#         return image


#     def add_text_with_background(self,image: np.ndarray, text: str, position=(10, 10), 
#                                 font_scale=4, font_thickness=5, 
#                                 text_color=(255, 255, 255), background_color=(0, 0, 0)) -> np.ndarray:
#         """
#         Adds text with a tightly fitted background rectangle to an image.
        
#         Parameters:
#             image (np.ndarray): Input image as a NumPy array (BGR format for OpenCV).
#             text (str): Text to display.
#             position (tuple): Top-left position for the text and rectangle.
#             font_scale (int): Scale of the font.
#             font_thickness (int): Thickness of the font.
#             text_color (tuple): Color of the text in (B, G, R).
#             background_color (tuple): Color of the rectangle background in (B, G, R).
            
#         Returns:
#             np.ndarray: Image with text and background rectangle added.
#         """
#         # Copy image to avoid modifying the original
#         output_image = image.copy()

#         # Font for the text
#         font = cv2.FONT_HERSHEY_SIMPLEX
        
#         # Get text size
#         (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

#         # Define rectangle coordinates with minimal padding
#         top_left = position
#         bottom_right = (top_left[0] + text_width + 4, top_left[1] + text_height + 8)  # Minimal padding

#         # Draw background rectangle
#         cv2.rectangle(output_image, top_left, bottom_right, background_color, -1)

#         # Define text position inside the rectangle
#         text_position = (top_left[0] + 2, top_left[1] + text_height + 4)

#         # Put text on top of the rectangle
#         cv2.putText(output_image, text, text_position, font, font_scale, text_color, font_thickness)
        
#         return output_image

def find_color_by_image_id(data, image_id_to_find):
    """
    Find the color associated with a specific image_id in a list of dictionaries.
    
    Parameters:
    - data (list of dict): List containing dictionaries, each with an 'image_id' and 'color' key.
    - image_id_to_find (int): The image_id to search for.
    
    Returns:
    - tuple or None: The color (as an RGB tuple) if found, otherwise None.
    """
    for entry in data:
        if entry.get('image_id') == image_id_to_find:
            return entry.get('color')
    return None

# class convert_process():
#     def transform_polygon_window_to_box(self,polygon, xmin, ymin):
#         """
#         Transforms a polygon from the original frame to the cropped frame by adjusting coordinates.
        
#         Parameters:
#         - polygon (shapely.geometry.Polygon): The polygon in the original frame.
#         - xmin (int or float): The x-coordinate of the top-left corner of the cropped region.
#         - ymin (int or float): The y-coordinate of the top-left corner of the cropped region.

#         Returns:
#         - transformed_polygon (shapely.geometry.Polygon): The polygon transformed to the cropped frame.
#         """
#         # Get the coordinates of the polygon
#         #print("polygon in trasform",polygon)
#         if isinstance(polygon, MultiPolygon):
#             polygon=polygon.geoms[0]
#             original_coords = list(polygon.exterior.coords)
#             cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]
#             transformed_polygon = Polygon(cropped_coords)
            
#         else:
#             original_coords = list(polygon.exterior.coords)

#             # Transform each point in the polygon to the cropped frame
#             cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]

#             # Create a new polygon with the transformed coordinates
#             transformed_polygon = Polygon(cropped_coords)

#         return transformed_polygon
#     def convert_points_box_to_full_frame(self,cropped_points, xmin, ymin):
#         """
#         Converts points from the cropped frame back to the original image frame.

#         Parameters:
#         - cropped_points (list of tuples): List of (x, y) coordinates in the cropped frame.
#         - xmin (int or float): x-coordinate of the top-left corner of the bounding box in the original frame.
#         - ymin (int or float): y-coordinate of the top-left corner of the bounding box in the original frame.

#         Returns:
#         - original_points (list of tuples): List of (x, y) coordinates in the original frame.
#         """
#         original_points = [(x + xmin, y + ymin) for (x, y) in cropped_points]
#         return original_points    
#     def convert_featurecpu_to_list_tuple(self,featurecpu):
#         list_of_tuples = [tuple(map(int, row)) for row in featurecpu.tolist()]
#         return list_of_tuples
#     def convert_polygon_window_to_full_frame(self,cropped_polygon,center_point,frame_size=640):
#         center_x, center_y = center_point
#         top_left_x = center_x - frame_size//2
#         top_left_y = center_y - frame_size//2
#         # Extract the coordinates of the polygon
#         cropped_coords = list(cropped_polygon.exterior.coords)
        
#         # Add the top-left offset to each vertex
#         big_frame_coords = [(x + top_left_x, y + top_left_y) for x, y in cropped_coords]
        
#         # Create a new polygon in the big frame using the transformed coordinates
#         big_frame_polygon = Polygon(big_frame_coords)
        
#         return big_frame_polygon
    
    
#     def convert_point_window_to_full_frame(self,points_in_window, window_center, window_size=640):
#         # Extract center coordinates
#         center_x, center_y = window_center
        
#         # Calculate the top-left corner of the window
        
#         top_left_x = center_x - window_size // 2
#         top_left_y = center_y - window_size // 2
        
#         # Create a list to hold the converted positions
#         points_in_full_frame = []
        
#         # Convert each position in the window to the larger frame
#         for (x, y) in points_in_window:
#             # Adjust the position based on the top-left corner
#             big_x = top_left_x + x
#             big_y = top_left_y + y
#             points_in_full_frame.append((int(big_x), int(big_y)))
        
#         return points_in_full_frame
        

#     def convert_bounding_boxes_to_big_frame(self,bboxes, window_center, window_size):
#         """
#         Convert bounding boxes from a smaller window to a larger frame.
        
#         Parameters:
#         - bboxes: np.ndarray of shape (n, 4) with each row as [x, y, w, h]
#         - window_center: Tuple[int, int], the center (x, y) of the window in the big frame
#         - window_size: Tuple[int, int], size of the window (width, height)
#         - big_frame_size: Tuple[int, int], size of the big frame (width, height)

#         Returns:
#         - List of converted bounding boxes in the big frame as [(x, y, w, h), ...]
#         """
#         window_center_x, window_center_y = window_center
#         window_width, window_height = window_size
        
        
#         # Calculate scaling factors

#         top_left_x=window_center_x - window_width/2
#         top_left_y=window_center_y - window_height/2

#         big_frame_bboxes = []
        
#         for box in bboxes:
#             x, y, w, h = box

#             # Convert to big frame coordinates
#             big_frame_x = top_left_x + x
#             big_frame_y = top_left_y + y
            
#             # Scale the width and height
#             big_w = w 
#             big_h = h
            
#             # Append the new bounding box to the list
#             big_frame_bboxes.append((big_frame_x, big_frame_y, big_w, big_h))

#         return big_frame_bboxes
#     def convert_points_from_big_to_window(self,points, center, window_size=(640, 640)):
#         #keep point in window only
#         # Calculate half-width and half-height of the window
#         half_width, half_height = window_size[0] // 2, window_size[1] // 2
        
#         # Calculate the bounds of the small window
#         x_min = center[0] - half_width
#         x_max = center[0] + half_width
#         y_min = center[1] - half_height
#         y_max = center[1] + half_height
        
#         # # Filter points that are inside the small window
#         # mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
#         #     (points[:, 1] >= y_min) & (points[:, 1] < y_max)
#         # cropped_points = points[mask]
        
#         # Translate the points to the small window's coordinate system
#         translated_points = points - np.array([x_min, y_min])
        
#         # Convert to list of tuples
#         return [tuple(point) for point in translated_points]
#     def convert_center_to_topleft(self,x_center, y_center, w, h):
#         x_topleft = x_center - w / 2
#         y_topleft = y_center - h / 2
#         return [x_topleft, y_topleft, w, h]


#     def convert_xywh_to_top_left(self,xywh):
#         """
#         Convert bounding box format from (x_center, y_center, width, height)
#         to (x_top_left, y_top_left, width, height) for a single NumPy array.

#         Parameters:
#             xywh (np.ndarray): A NumPy array of shape [1, 4] representing (x_center, y_center, width, height).
        
#         Returns:
#             np.ndarray: A NumPy array of shape [1, 4] representing (x_top_left, y_top_left, width, height).
#         """
#         x_center, y_center, width, height = xywh
#         x_top_left = x_center - (width / 2)
#         y_top_left = y_center - (height / 2)
#         return np.array([x_top_left, y_top_left, width, height])

# def init_detection(input_path):
#     input_fordel_path=input_path
#     yolov8_seg_model_path = "yolov8x-seg.pt"
#     im = read_image(input_fordel_path+"frame_0.jpg")
#     h = im.shape[0]
#     w = im.shape[1]

#     detection_model_seg = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=yolov8_seg_model_path,
#     confidence_threshold=0.1,
#     device="cuda", # or 'cuda:0'
#     )


#     result = get_sliced_prediction(
#         im,
#         detection_model_seg,
#         slice_height = 640,
#         slice_width = 640,
#         overlap_height_ratio = 0.2,
#         overlap_width_ratio = 0.2)
    
#     result.export_visuals(export_dir="demo_data/yolo+sahi.jpg")
#     rgbarray=np.array(result.image)
#     object_prediction_list = result.object_prediction_list
#     return result

# def process_first_frame(result):

#     id=1
#     id_list_intrack=[]
#     trackpoint_LK=[]


#     rgbarray=np.array(result.image)
#     object_prediction_list = result.object_prediction_list

#     count=0
#     remove_detect_list=[]
#     for ob1 in object_prediction_list:
#         #print("count",count)
#         croped=crop_bbox(rgbarray,[ob1.bbox.minx,ob1.bbox.miny,ob1.bbox.maxx,ob1.bbox.maxy])
#         coords=ob1.mask.segmentation[0]
#         points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

#         polygon_in_window = Polygon(points) 

#         # # start slide here
#         # transformed_polygon=convert_process().transform_polygon_window_to_box(polygon_in_window,ob1.bbox.minx,ob1.bbox.miny)
#         # show_image=visual_image().visualize_shapely_polygon_on_image(croped,transformed_polygon)
#         # plt.imshow(show_image)
#         # plt.show()
#         # # end slide
   
        

#         #points_one_object=apply_processing('harris',croped,polygon,ob1.bbox.minx,ob1.bbox.miny)
#         points_one_object=apply_processing('harris',croped,polygon_in_window,ob1.bbox.minx,ob1.bbox.miny)
#         show_image=visual_image().visualize_points_on_image(image_np=croped,points=points_one_object)
        
#         #visual_image().show_image(show_image)



#         # slide here
#         # show_image=harris_selection_method().visual_result(croped)
#         # plt.imshow(show_image)
#         # plt.show()
#         # end slide
#         if len(points_one_object)>0:
#             point_sample=points_one_object
#             #print("point_sample",point_sample)
#             id_list_intrack += [id]*len(point_sample)
#             original_points=convert_process().convert_points_box_to_full_frame(point_sample,ob1.bbox.minx,ob1.bbox.miny)
#             #print("original_points",original_points)
#             trackpoint_LK+=original_points
#             #show_image=visual_image().visualize_points_on_image(rgbarray,original_points)

#             id+=1
#         else:
#             remove_detect_list.append(count)

#         count+=1

#     list_dict_info=result.to_coco_annotations()
#     dict_filtered = [list_dict_info[i] for i in range(len(list_dict_info)) if i not in remove_detect_list]

#     history_points_tracked_list=[0] * len(id_list_intrack)

#     show_image=visual_image().visualize_points_on_image(rgbarray,trackpoint_LK)
    
#     plt.imshow(show_image)
#     plt.show()
#     print("Done process_first_frame")
#     return trackpoint_LK,id_list_intrack,history_points_tracked_list,dict_filtered,show_image

# class generate_centers():
#     def generate_tile_centers(self,frame_width, frame_height, tile_width=640, tile_height=640, overlap_width=10, overlap_height=0):
#         centers = []
        
#         # Calculate the number of tiles needed along width and height
#         num_tiles_x = (frame_width + tile_width - overlap_width - 1) // (tile_width - overlap_width)
#         num_tiles_y = (frame_height + tile_height - overlap_height - 1) // (tile_height - overlap_height)
        
#         # Calculate actual step sizes (this will help distribute overlap in the center)
#         step_x = (frame_width - tile_width) // (num_tiles_x - 1) if num_tiles_x > 1 else frame_width
#         step_y = (frame_height - tile_height) // (num_tiles_y - 1) if num_tiles_y > 1 else frame_height
        
#         # Calculate tile centers based on the number of tiles
#         for j in range(num_tiles_y):
#             for i in range(num_tiles_x):
#                 # Calculate the center of the tile
#                 x_center = tile_width // 2 + i * step_x
#                 y_center = tile_height // 2 + j * step_y
#                 centers.append((x_center, y_center))

#         return centers

#     def generate_tile_centers_border_and_salient(self,frame_width, frame_height, tile_width=640, tile_height=640, overlap_width=10, overlap_height=0):
#         centers = []
#         border_centers = []
#         salient_centers = []
        
#         # Calculate the number of tiles needed along width and height
#         num_tiles_x = (frame_width + tile_width - overlap_width - 1) // (tile_width - overlap_width)
#         num_tiles_y = (frame_height + tile_height - overlap_height - 1) // (tile_height - overlap_height)
        
#         # Calculate actual step sizes (this will help distribute overlap in the center)
#         step_x = (frame_width - tile_width) // (num_tiles_x - 1) if num_tiles_x > 1 else frame_width
#         step_y = (frame_height - tile_height) // (num_tiles_y - 1) if num_tiles_y > 1 else frame_height
        
#         # Calculate tile centers and categorize them
#         for j in range(num_tiles_y):
#             for i in range(num_tiles_x):
#                 # Calculate the center of the tile
#                 x_center = tile_width // 2 + i * step_x
#                 y_center = tile_height // 2 + j * step_y
#                 centers.append((x_center, y_center))
                
#                 # Determine if the center is at an edge
#                 if i == 0 or i == num_tiles_x - 1 or j == 0 or j == num_tiles_y - 1:
#                     border_centers.append((x_center, y_center))
#                 else:
#                     salient_centers.append((x_center, y_center))

#         return centers, border_centers, salient_centers


# def strategy_pick_window(step,list_all_center,border_centers,salient_centers,current_track_points ):
#     big_list=[border_centers,salient_centers,current_track_points]
#     current_list = big_list[step % 3]

#     if step%3==0: #border
#         center=current_list[step // 3 % len(current_list)]
#         win_color=(255, 255, 0)
#     elif step%3==1:  #salient
#         center=current_list[step // 3 % len(current_list)]
#         win_color=(0, 255, 0)
#     else : #step%3==2:
#         center =random.choice(current_track_points)
#         win_color=(255, 0, 0)

#     #center=random.choice(list_all_center)
#     # center=random.choice(current_track_points)
#     # win_color=(255, 0, 0)
#     return center,win_color



# def crop_window(full_frame, center,window_size=640):
#     #center : tuple (x,y)
#     center_x=center[0]
#     center_y=center[1]
#     # Get the dimensions of the full frame
#     frame_height, frame_width = full_frame.shape[:2]
    
#     # Half of the window size to help with calculations
#     half_window_size = window_size // 2

#     # Calculate the top-left corner (x, y) of the window
#     top_left_x = max(center_x - half_window_size, 0)
#     top_left_y = max(center_y - half_window_size, 0)

#     # Calculate the bottom-right corner, ensuring we don't go out of frame bounds
#     bottom_right_x = min(center_x + half_window_size, frame_width)
#     bottom_right_y = min(center_y + half_window_size, frame_height)

#     # Crop the window from the full frame
#     cropped_window = full_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

#     return cropped_window

# class update():
#     def history_point_mask(self,points,history_points_in_mask,yolo_detector,center_window):
        
#         r=yolo_detector[0]
#         #result = []
#         if len(yolo_detector[0].masks.xy)>0:
#             for idx, pixel in enumerate(points):
#                 point = Point(pixel)

#                 for mask_count in range (len(r.masks.xy)):
#                     mask = Polygon(r.masks.xy[mask_count])
#                     #mask_buffer = mask.buffer(distance=-2)
#                     mask=convert_process().convert_polygon_window_to_full_frame(mask,center_window)
#                     if mask.contains(point):
#                         history_points_in_mask[idx]=0

#                         # result.append(idx)

#         # if len(result) != len(set(result)):
#         #     result=remove_duplicates(result)
#         #     print("Duplicates in history_points_tracked")
        
#         # else:
#         #     for i in range(len(history_points_in_mask)):  # Iterate over each index in the history list
#         #         if i in result:  # If the index is in the specified indices list
#         #             history_points_in_mask[i] = 0  # Set the value to 0
#         #         else:
#         #             history_points_in_mask[i] += 1

#         return history_points_in_mask
#     def update_list_dict_info(self,list_dict_info,newid, bbox,groupid_center,list_tuple_five_points,conf,threshold_conf=0.8):
#         #use for new ID

#         if conf>threshold_conf:
#             new_id_dict= {'image_id': newid,
#                             'bbox': bbox, # [x,y,w,h]
#                             'score': conf, #float
#                             'category_id': None, #int
#                             'category_name': None, #str
#                             'segmentation': [],
#                             'iscrowd': 0,
#                             'area': 0,
#                             'ori_bbox': bbox, #list [x,y,w,h],
#                             'ori_center_points': groupid_center, #(x,y)
#                             'color': (randint(0, 255),randint(0, 255),randint(0, 255)),
#                             'lastest_point':None,
#                             'disappear_step':None ,   
#                             'points_step_t':list_tuple_five_points,
#                             'visible':True, 
#                             'drift':(0,0)
                            
#                             }
#         else: 
#             new_id_dict= {'image_id': newid,
#                 'bbox': bbox, # [x,y,w,h]
#                 'score': conf, #float
#                 'category_id': None, #int
#                 'category_name': None, #str
#                 'segmentation': [],
#                 'iscrowd': 0,
#                 'area': 0,
#                 'ori_bbox': bbox, #list [x,y,w,h],
#                 'ori_center_points': groupid_center, #(x,y)
#                 'color': (randint(0, 255),randint(0, 255),randint(0, 255)),
#                 'lastest_point':None,
#                 'disappear_step':None ,   
#                 'points_step_t':list_tuple_five_points,
#                 'visible':False ,
#                 'drift':(0,0)
#                 }

#         #list_dict_info.append(new_id_dict) #backup
#         list_dict_info[newid]=new_id_dict
#         return list_dict_info


#     def step_accumulate(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.8):
#         for idx,value in enumerate(match_box_id):
#             print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
#             if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_box_conf:# and is_not_box_at_edge(yolo_detector[0].boxes.xywh.cpu().numpy()[idx]):
#                 #tuc la deteced box matched, va co co   nf cao thi minh se update box
#                 box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
#                 print("box_yolo",box_yolo.shape)
#                 box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)

#                 box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]
#                 print("box_yolo ************* ",box_yolo)
#                 dict_inside[value]['visible']=True
                




#                 dict_inside[value]['bbox']=box_yolo
#                 dict_inside[value]['ori_bbox']=box_yolo
#                 dict_inside[value]['ori_center_points']=self.update_points_group_center(value,tracking_list,points)
#                 dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
#             if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]<threshold_box_conf:
#                 dict_inside[value]['score']+=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
#                 if dict_inside[value]['score']>0.8:
#                     dict_inside[value]['visible']=True

                

            

            


#         dict_inside=self.decrease_score_step(dict_inside)

#         return dict_inside


#     # def step_box_update(dict_inside,out_yolo,match_box_id,threshold_box_conf=0.8):
#     #     idx,value in enumerate(
#     #     dict_inside=update_bbox()

#     #     return dict_inside

#     def update_points_group_center(self,id_,tracking_list,points):

#         id_points = [points[i] for i in range(len(tracking_list)) if tracking_list[i] == id_]
        
#         # Calculate the mean (center) of the points
#         center = np.mean(id_points, axis=0)
        
#         # Store the center in the dictionary
#         return center
#     def decrease_score_step(self,inside_dict):
#         for key, value in inside_dict.items():
#             if 'score' in value:  # Check if 'score' exists
#                 value['score'] -= 0.01

#         return inside_dict

#     def predict_box_based_equ4(self,driftx,drifty,x_topleft_previous,y_topleft_previous,x_detect,y_detect,tau=4):



#         x_pre_topleft=driftx +x_topleft_previous +tau*x_detect
#         y_pre_topleft=drifty +y_topleft_previous +tau*y_detect
#         return (x_pre_topleft,y_pre_topleft)

#     def update_bounding_box_based_on_eq4(self,dict_inside,yolo_detector,tracking_list,points,match_box_id,center_window,threshold_box_conf=0.5):
#         for idx,value in enumerate(match_box_id):
#             print("yolo_detector[0].boxes.conf.cpu().numpy()[idx]",yolo_detector[0].boxes.conf.cpu().numpy()[idx])
#             if value!=0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]<threshold_box_conf:
                
#                 drix=dict_inside[value]['drift'][0]
#                 driy=dict_inside[value]['drift'][1]
#                 x_pre_top=dict_inside[value]['bbox'][0]
#                 y_pre_top=dict_inside[value]['bbox'][1]
#                 box_yolo=yolo_detector[0].boxes.xywh.cpu().numpy()[idx]
               
#                 box_yolo=convert_process().convert_xywh_to_top_left(box_yolo)

#                 box_yolo=convert_process().convert_bounding_boxes_to_big_frame(box_yolo.reshape(1, 4),center_window,(640,640))[0]
   
#                 predict_basedeq4=self.predict_box_based_equ4(drix,driy,x_pre_top,y_pre_top,box_yolo[0],box_yolo[1])
                
#                 if type(dict_inside[value]['bbox']) is tuple:
#                     dict_inside[value]['bbox']=list(dict_inside[value]['bbox'])
#                 else:
#                     print("dict_inside[value]['bbox']",dict_inside[value]['bbox'])
#                     new_box=dict_inside[value]['bbox']
#                     new_box[0]=predict_basedeq4[0]
#                     new_box[1]=predict_basedeq4[1]
#                     dict_inside[value]['bbox']=new_box


#         return dict_inside


# def is_not_box_at_edge(bbox, window_size=(640,640)):
#     # Extract bounding box parameters from the input array
#     x_center, y_center, w, h = bbox
#     window_width,window_height=window_size
#     # Calculate the top-left and bottom-right coordinates
#     x_top_left = x_center - w / 2
#     y_top_left = y_center - h / 2
#     x_bottom_right = x_center + w / 2
#     y_bottom_right = y_center + h / 2
    
#     # Check if any corner is at the edge of the detection window
#     if x_top_left <= 5 or y_top_left <= 5 or x_bottom_right >= window_width-5 or y_bottom_right >= window_height-5:
#         return False
#     return True

# class matching_module():
#     def poa(self,points, polygon):
#         #points: list of tuple
#         #polygon : shapely polygon
#         count = 0

#         # Iterate over each point from the Harris corner detector
#         for point in points:
#             # Create a Shapely Point object
#             shapely_point = Point(point)

#             # Check if the point is inside the polygon
#             if polygon.contains(shapely_point):
#                 count += 1
#         poa_value=count/len(points)


#         return poa_value,count

#     def poa_table(self,points,yolo_detector,tracking_list,center_window):
#         #poa table: row ~ mask /
#         # each row : value poa of group ID

#         unique_values = set(tracking_list)
        
#         # Get the number of unique values
#         number_unique_id = len(unique_values)
#         print("number_unique_id",number_unique_id)


#         #poa_table is list of list , each row is list belong to a mask
#         poa_table=[]
#         for r in yolo_detector:
#             # TODO something: visula polygon detected on cropped frame
#             # orig_img=r.orig_img
#             # print("type orig_img",type(orig_img))
#             # point_list_of_tuples = [tuple(map(int, row)) for row in points]
#             # print("point_list_of_tuples",point_list_of_tuples)
#             # show_image=visualize_shapely_polygon_on_image(orig_img,point_list_of_tuples)
#             # plt.imshow(show_image)
#             # plt.show()

#             if len(r.masks.xy)>0:
#                 for mask_count in range (len(r.masks.xy)):
#                     mask = Polygon(r.masks.xy[mask_count])
#                     #mask_buffer = mask.buffer(distance=-5)
#                     mask=convert_process().convert_polygon_window_to_full_frame(mask,center_window)

#                     poa_per_mask=[]
#                     #mask to shapely

#                     for unique in unique_values:
#                         indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
#                         point_of_one_id=[tuple(points[i]) for i in indices_of_unique]
#                         poa_value, count= self.poa(point_of_one_id,mask)
#                         #print("poa_value",poa_value)
#                         poa_per_mask.append(poa_value)

#                     #print("poa_per_mask",poa_per_mask)
#                     poa_table.append(poa_per_mask)

#             #     mask_poa.append(poa)
#             # poa_table.append(mask_poa)

#         return poa_table
    
#     def matching1 (self,tracking_list,poa_table): # First association
#         #poa_table=self.poa_table
#         unique_values = list(set(tracking_list))
#         number_bbox=len(poa_table)
#         #print("number_bbox",number_bbox)
#         match_box_id=[0]*number_bbox  # 0 mean unmatch to any id; init step
#         for idx_mask,poa_per_mask in enumerate(poa_table):
#             best_match_index=poa_per_mask.index(max(poa_per_mask))
#             #print("best_match_index",best_match_index)

#             #todo: threshold 0.4 ? + .index(max(poa_per_mask))\
#             #  la tim vi tri dau tien cua gia tri max, \
#             # neu co 2 index = nhau co loi nhe
#             #todo check heare
#             if poa_per_mask[best_match_index]>=0.2 : 
#                 match_box_id[idx_mask]=unique_values[best_match_index]


#         print("match_box_id",match_box_id)
#         return match_box_id

#     def filter_poa_table (self,tracking_list,poa_table): # remove points in belong to same object
#         #poa_table=self.poa_table
#         unique_values = list(set(tracking_list))
#         number_bbox=len(poa_table)
#         #print("number_bbox",number_bbox)
#         match_box_id=[0]*number_bbox  # 0 mean unmatch to any id; init step
#         for idx_mask,poa_per_mask in enumerate(poa_table):
#             #indexes = [i for i, value in enumerate(poa_per_mask) if value == max(poa_per_mask)]

#             indexes = [i for i, value in enumerate(poa_per_mask) if value ==1]
#             if len(indexes) >1:
#                 id_list=[unique_values[i] for i in indexes]
#                 id_list.remove(min(id_list))

#                 remove_indexes=[i for i, value in enumerate(tracking_list) if value in id_list]
#                 return remove_indexes
#         #     #best_match_index=poa_per_mask.index(max(poa_per_mask))
#         #     #print("best_match_index",best_match_index)

#         #     #todo: threshold 0.4 ? + .index(max(poa_per_mask))\
#         #     #  la tim vi tri dau tien cua gia tri max, \
#         #     # neu co 2 index = nhau co loi nhe
#         #     #todo check heare
#         #     if poa_per_mask[best_match_index]>=0.4 : 
#         #         match_box_id[idx_mask]=unique_values[best_match_index]

#             else:
#                 return []
#         # print("match_box_id",match_box_id)
        



#     def matching2(self,match_box_id,tracking_list,yolo_detector,centercrop,featurecpu,threshold_ecl_dis_match=50):
#         #unmatched_bbox_list_idx=find_indices_unmatched1(match_box_id)
#         #unique_values = set(tracking_list)
#         exist_id=unique_id_not_in_matched_box(match_box_id,tracking_list)
#         exist_id=list(set(tracking_list))
#         list_tuple_center_of_box_position_in_window=center_of_box_detected(yolo_detector)
#         list_tuple_center_of_box_position_in_full_frame=convert_process().convert_point_window_to_full_frame(list_tuple_center_of_box_position_in_window,centercrop)
#         points = [tuple(map(int, row)) for row in featurecpu.tolist()]
      
#         #[0,5]
#         match_box_id_after_matching1=match_box_id
#         for idx,value in enumerate(match_box_id_after_matching1):
#             if value==0:
#                 average_ecl_distances_of_groupid_to_center_box= [] #for one box
#                 centerbox=list_tuple_center_of_box_position_in_full_frame[idx]

#                 for unique in exist_id:
#                     #if unique not in match_box_id:
#                     indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
#                     point_of_one_id=[tuple(points[i]) for i in indices_of_unique]
#                     id_to_center_distance=calculate_average_distance(point_of_one_id,centerbox)
#                     average_ecl_distances_of_groupid_to_center_box.append(id_to_center_distance)
#                     if max(average_ecl_distances_of_groupid_to_center_box)<threshold_ecl_dis_match:
#                         best_match_index=average_ecl_distances_of_groupid_to_center_box.index(max(average_ecl_distances_of_groupid_to_center_box))
                        
                
#                         match_box_id[idx]=exist_id[best_match_index]
        
    




#         return match_box_id

def find_indices_unmatched1(input_list):
    # Use list comprehension to find indices of value 0
    indices = [index for index, value in enumerate(input_list) if value == 0]
    return indices

# def center_of_box_detected(yolo_detector):
#     for r in yolo_detector: 
#         center=r.boxes.xywh.cpu().numpy() #numpy (n,4) : n is number of box, 4 ls xywh : center of box + w+h
#         tuple_list = [tuple(row[:2]) for row in center] 
#     return tuple_list  #list of tuple (x,y) center of bboxes


class check_live_info():
    def check_status(self,sta,idx_list_need_remove):
        if np.any(sta.cpu() == 1):
            with sta.lock_cpu() as status_cpu:
                rows_to_remove = np.where(status_cpu == 1)[0]
                for row in rows_to_remove:
                    idx_list_need_remove.append(int(row))
                #idx_list_need_remove.append(int(rows_to_remove))

        return idx_list_need_remove
    def check_history(self,history_point,idx_list_need_remove,threshold_point_not_inmask=200):
        for index, value in enumerate(history_point):
            
            if value > threshold_point_not_inmask:
                idx_list_need_remove.append(int(index))

        return idx_list_need_remove

    
    def check_and_find_remove_list(self,sta,history_point,threshold_point_not_inmask):
        idx_list_need_remove=[]
        idx_list_need_remove=self.check_status(sta,idx_list_need_remove)
        idx_list_need_remove=self.check_history(history_point,idx_list_need_remove,threshold_point_not_inmask)
        return idx_list_need_remove
    
# class remove_intrack():
#     def remove_duplicates(self,lst):
#         """
#         This function removes duplicates from the list while preserving the order of the elements.
        
#         :param lst: The list from which duplicates need to be removed.
#         :return: A list with duplicates removed, keeping only the first occurrence.
#         """
#         seen = set()  # Set to track seen values
#         result_out = []
        
#         for item in lst:
#             if item not in seen:
#                 result_out.append(item)  # Add item to result if not already seen
#                 seen.add(item)       # Mark item as seen
#         return result_out

#     def remove_featurecpu(self,featurecpu,row_index_list):
#         featurecpu = np.delete(featurecpu, row_index_list, axis=0)
#         featurecpu = vpi.asarray(featurecpu)

#         return featurecpu
#     def remove_status(self,sat,row_index_list):
#         sat = np.delete(sat, row_index_list, axis=0)
#         sat = vpi.asarray(sat)
#         return sat
#     def remove_id_list_intrack(self,id_list_intrack,row_index_list):
#         #removed_element=id_list_intrack.pop(int(row_index))
#         id_list_intrack = [row for i, row in enumerate(id_list_intrack) if i not in row_index_list]

#         return id_list_intrack
#     def remove_history_old_point(self,history,row_index_list):
#         history= [row for i, row in enumerate(history) if i not in row_index_list]
#         return history
    
#     def apply_remove(self,idx_list_need_remove,featurecpu,sat,id_list_intrack,history):
        
#         idx_list_need_remove=self.remove_duplicates(idx_list_need_remove)
#         with featurecpu.lock_cpu() as curFeatures_cpu:
#             with sat.lock_cpu() as status_cpu:
                
                    
#                 removed_status=self.remove_status(status_cpu,idx_list_need_remove)
#                 removed_featurecpu=self.remove_featurecpu(curFeatures_cpu,idx_list_need_remove)
#                 removed_history=self.remove_history_old_point(history,idx_list_need_remove)
#                 removed_id_intrack=self.remove_id_list_intrack(id_list_intrack,idx_list_need_remove)


#         return removed_featurecpu,removed_status,removed_id_intrack,removed_history
    


# def calculate_average_distance(points, center_box):
#     """
#     Calculate the average Euclidean distance of all points to the center_box.

#     Parameters:
#     - points: List of tuples [(x1, y1), (x2, y2), ...].
#     - center_box: Tuple (x_center, y_center) representing the center of the box.

#     Returns:
#     - average_distance: The average Euclidean distance of the points to the center_box.
#     """
#     total_distance = 0
#     for point in points:
#         x, y = point
#         x_center, y_center = center_box
#         distance = np.sqrt((x_center - x)**2 + (y_center - y)**2)
#         total_distance += distance
    
#     # Calculate the average distance
#     average_distance = total_distance / len(points) if points else 0
    
#     return average_distance


# def unique_id_not_in_matched_box(matched_box, b):
#     # Convert list a to a set for faster lookup
#     #set_a = set(matched_box)
    
#     # Use set to remove duplicates from b and filter out values present in set_a
#     result = [value for value in set(b) if value not in matched_box]
    
#     return result


# def need_add_id_and_point(matched_box): #todo: add to check class
#     if matched_box!=None:
#         if  0 in matched_box:
#             return True
#     else:
#         return False


# def need_increase_point_for_id (tracking_list, matched_box,point_per_animal=5):
#     id_matched=set(matched_box)
#     id_need_increase_point_dict={}
#     for i in id_matched:
#         if i !=0:
#             count=tracking_list.count(i)
#             if count <point_per_animal :
#                 id_need_increase_point_dict[i]=point_per_animal-count

    
#     return id_need_increase_point_dict




def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    box = [x_center, y_center, w, h]
    """
    # Calculate the coordinates of the intersection rectangle
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Compute the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute area of intersection
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    # Compute area of both bounding boxes
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    # Compute area of union
    union_area = area1 + area2 - intersection_area

    # Compute IoU
    return intersection_area / union_area if union_area > 0 else 0


def filter_dict_main_overlap_box(bbox_dict, iou_threshold=0.9,iou_threshold_modify=0.9):
    """
    Filter out items with overlapping bounding boxes with IoU greater than a threshold.
    """
    # Create a list to store keys that will be removed
    keys_to_remove = set()
    
    # List of items in the dict
    items = list(bbox_dict.items())
    
    for i in range(len(items)):
        key_i, item_i = items[i]
        bbox_i = item_i['bbox']
        
        # Compare the current bounding box with all other boxes
        for j in range(i + 1, len(items)):
            key_j, item_j = items[j]
            bbox_j = item_j['bbox']
            
            # Calculate IoU between the two bounding boxes
            if iou(bbox_i, bbox_j) > iou_threshold:
                # Remove the item with the larger key
                if key_i > key_j:
                    #keys_to_remove.add(key_i)
                    item_i['visible']=False

                else:


                    #keys_to_remove.add(key_j)
                    item_j['visible']=False

            if calculate_modified_iou(bbox_i, bbox_j) > iou_threshold_modify:
                # Remove the item with the larger key
                if key_i > key_j:
                    #keys_to_remove.add(key_i)
                    item_i['visible']=False

                else:


                    #keys_to_remove.add(key_j)
                    item_j['visible']=False            



    # Remove the items with the larger key that overlap
    # for key in keys_to_remove:
    #     if key in bbox_dict:
    #         del bbox_dict[key]
    
    return bbox_dict

def calculate_modified_iou(box1, box2):
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

# def check_box_overlap(input_box, dict_data, threshold=0.6):
#     """
#     Check if the input_box overlaps with any bounding boxes in the dictionary.
    
#     Args:
#     - input_box: The box to compare, given as (x_center, y_center, w, h).
#     - dict_data: The dictionary of annotations where each entry contains a 'bbox' key.
#     - threshold: The IoU threshold to check for overlap.
    
#     Returns:
#     - True if no overlap exceeds the threshold, False if any overlap exceeds the threshold.
#     """
#     for item in dict_data.values():
#         bbox = item['bbox']  # Extract the bounding box from the dict
#         iou = calculate_modified_iou(input_box, bbox)  # Calculate IoU
        
#         # If IoU exceeds the threshold, return False
#         if iou > threshold:

#             return False
    
#     # If no overlap exceeded the threshold, return True
#     return True

# class add_points():
#     def add_to_feature(self,featurecpuin,new_points):
#         # if isinstance(featurecpuin, vpi.Array):
#         #     with featurecpuin.lock_cpu() as feacpu:
#         #         copy_fea = feacpu

#         print("featurecpuin",type(featurecpuin))
#         points_array = np.array(new_points)
#         print("points_array",points_array.shape)
#         print("featurecpu",featurecpuin.shape)
#         combined_array = np.vstack((featurecpuin, points_array))
#         print("combined_array",combined_array.shape)
#         combined_array = combined_array.astype(np.float32)
#         #cur_Features = vpi.asarray(combined_array)
#         return combined_array
    
#     def add_to_history(self,history,number_points_add):
#         history += [0] * number_points_add
#         return history
    
#     def add_to_id_intrack(self,trackinglist,newid,number_points_add):

#         trackinglist+=[newid]*number_points_add
#         print("insdie tracking list",trackinglist)

#         return trackinglist
    
#     def getnumpyimage_from_yolo(self,yolo_detector,box_target):
#         window_np=yolo_detector[0].orig_img
#         box=yolo_detector[0].boxes.xywh[box_target]
#         x_center, y_center, width, height = box.cpu().numpy()
#         x_min = int(x_center - width / 2)
#         y_min = int(y_center - height / 2)
#         x_max = int(x_center + width / 2)
#         y_max = int(y_center + height / 2)

#         # Ensure the coordinates are within the image boundaries
#         x_min = max(x_min, 0)
#         y_min = max(y_min, 0)
#         x_max = min(x_max, window_np.shape[0])  # width of the image
#         y_max = min(y_max, window_np.shape[1])  

#         cropped_image = window_np[y_min:y_max, x_min:x_max]
#         #yolo_detector[0].masks
#         mask = Polygon(yolo_detector[0].masks.xy[box_target])

#         return cropped_image,mask,x_min,y_min,x_center,y_center,width,height


#     def apply_add_process_new_id(self,rgb_image,dict_inside,matched_box,yolo_detector,centerwindow,featurecpu,trackinglist,history,thesshold_area_of_animal,threshold_conf=0.3):
        
#         for idx,value in enumerate(matched_box):
#             if value==0 and yolo_detector[0].boxes.conf.cpu().numpy()[idx]>threshold_conf :
#                 conf=yolo_detector[0].boxes.conf.cpu().numpy()[idx]
#                 image_np,polygon,minx,miny,xcen,ycen,wid,hei=self.getnumpyimage_from_yolo(yolo_detector,idx)
#                 converted_box=convert_process().convert_bounding_boxes_to_big_frame(np.array([[minx, miny, wid, hei]]),centerwindow,(640,640))
#                 dummy= check_box_overlap(converted_box[0],dict_inside)
#                 five_points=harris_selection_method().filter_some_points(image_np,polygon,minx,miny)

#                 if len(five_points)>0 and wid * hei>thesshold_area_of_animal and dummy:
#                     five_points_in_window=convert_process().convert_points_box_to_full_frame(five_points,minx,miny)
#                     five_points_in_full_frame=convert_process().convert_point_window_to_full_frame(five_points_in_window,centerwindow)
#                     convert_minx_miny=convert_process().convert_point_window_to_full_frame([(minx,miny)],centerwindow)
#                     number_id_exist=len(dict_inside)
#                     newid=number_id_exist+1

#                     print("1 five_points_in_full_frame",five_points_in_full_frame)
#                     centroid=compute_centroid(five_points_in_full_frame)

#                     featurecpu=self.add_to_feature(featurecpu,five_points_in_full_frame)
#                     history=self.add_to_history(history,5)
#                     trackinglist=self.add_to_id_intrack(trackinglist,newid,5)
#                     dict_inside=update().update_list_dict_info(dict_inside,newid,[convert_minx_miny[0][0],convert_minx_miny[0][1],wid,hei],centroid,five_points_in_full_frame,conf)



#         return dict_inside,featurecpu,trackinglist,history

#     def apply_add_process_need_more_points(self,dictid_need_increase_point,rgb_image,dict_inside,matched_box,yolo_detector,centerwindow,featurecpu,trackinglist,history):

#         for idx,value in enumerate(matched_box):
#             if value!=0:
#                 if value in dictid_need_increase_point.keys():

#                     image_np,polygon,minx,miny,xcen,ycen,wid,hei=self.getnumpyimage_from_yolo(yolo_detector,idx)
                    
#                     print("dictid_need_increase_point[value]^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",dictid_need_increase_point[value])
#                     five_points=harris_selection_method().filter_some_points(image_np,polygon,minx,miny,number_point_per_animal=dictid_need_increase_point[value])
#                     if len(five_points)>0:
#                         five_points_in_window=convert_process().convert_points_box_to_full_frame(five_points,minx,miny)
#                         five_points_in_full_frame=convert_process().convert_point_window_to_full_frame(five_points_in_window,centerwindow)
#                         convert_minx_miny=convert_process().convert_point_window_to_full_frame([(minx,miny)],centerwindow)
#                         #number_id_exist=len(dict_inside)
#                         id_need_add=value

#                         print("1 five_points_in_full_frame",five_points_in_full_frame)
#                         #centroid=compute_centroid(five_points_in_full_frame)

#                         featurecpu=self.add_to_feature(featurecpu,five_points_in_full_frame)
#                         history=self.add_to_history(history,dictid_need_increase_point[value])
#                         trackinglist=self.add_to_id_intrack(trackinglist,id_need_add,dictid_need_increase_point[value])
#                         #dict_inside=update().update_list_dict_info(dict_inside,id_need_add,(convert_minx_miny[0][0],convert_minx_miny[0][1],wid,hei),centroid,five_points_in_full_frame)



#         return dict_inside,featurecpu,trackinglist,history






# def compute_centroid(points):
#     """
#     Compute the centroid (average x and y) of a set of points.

#     Parameters:
#     - points: List of tuples [(x1, y1), (x2, y2), ...]

#     Returns:
#     - centroid: Tuple (x_center, y_center)
#     """
#     points_array = np.array(points)
#     centroid_x = np.mean(points_array[:, 0])
#     centroid_y = np.mean(points_array[:, 1])
    
#     return (centroid_x, centroid_y)

# def update_bounding_box(center_points_t0, points_t1, box_t):
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
    
# def reconstruct_process(np_image,dict_coco_annotations,cur_feature,tracking_list):
#     #dict_inside=dict_coco_annotations
#     unique_values = set(tracking_list)
#     print("unique_values in reconstruct ",unique_values)
#     for unique in unique_values:
#         indices_of_unique = np.where(np.array(tracking_list) == unique)[0].tolist()
        
#         #point_of_one_id_step_t=[tuple(prev_feature[i]) for i in indices_of_unique]
        
        
#         ####
#         # print("cur_feature", cur_feature.shape)
#         # print("tracking_list",len(tracking_list))
#         # print("set track ing list ",set(tracking_list))
        
        
        
#         point_of_one_id_step_t1=[tuple(cur_feature[i]) for i in indices_of_unique]

#         #dict_coco_annotations[unique-1]['points_step_t']=point_of_one_id_step_t1
#         '''
#         #Todo truong hop id bi xoa(mat track) co id co step t nhung khong co t+1
#         for item in dict_coco_annotations:
#             # Check if the image_id matches
#             if item['image_id'] == unique:
#                 box_ori=item['ori_bbox']
#                 #box_previous=find_box_by_id(list_boxes_complete_step_t,unique)
#                 center_t0=item['ori_center_points']
#                 reconstructed_box_step_t1=update_bounding_box(center_t0,point_of_one_id_step_t1,box_ori)
#                 #x,y,w,h=reconstructed_box_step_t1
                
#                 simple_box_object=reconstructed_box_step_t1
#                 #print("reconstructed_box_step_t1",type(reconstructed_box_step_t1[0]))
#                 #print("reconstructed_box_step_t1",reconstructed_box_step_t1)
#                 item['bbox']=simple_box_object
#                 #dict_coco_annotations=update_bbox_by_image_id(dict_coco_annotations,unique,simple_box_object)
#                 # listtest=point_of_one_id_step_t1 #ok
#                 # listtest.append(center_t0)
#                 # print("listtest",listtest)
#                 # show_image=visual_image().visualize_points_on_image(image_np=np_image,points=listtest,color=(123,0,10))
#                 # plt.imshow(show_image)
#                 # plt.show()
#             '''

#         if unique in dict_coco_annotations:
#             item = dict_coco_annotations[unique]
#             box_ori=item['ori_bbox']
#             #box_previous=find_box_by_id(list_boxes_complete_step_t,unique)
#             center_t0=item['ori_center_points']
#             reconstructed_box_step_t1, drift =update_bounding_box(center_t0,point_of_one_id_step_t1,box_ori)
#             #x,y,w,h=reconstructed_box_step_t1
#             item['drift']=drift
#             simple_box_object=reconstructed_box_step_t1
#             #print("reconstructed_box_step_t1",type(reconstructed_box_step_t1[0]))
#             #print("reconstructed_box_step_t1",reconstructed_box_step_t1)
#             item['bbox']=simple_box_object
#     return dict_coco_annotations



def find_box_by_id(simple_boxes_list, target_id):
    """
    Find the box with the specified id in the list of boxes.
    
    Parameters:
    - boxes: List of simple_box objects.
    - target_id: The id of the box to find.

    Returns:
    - A tuple (x, y, w, h) if found, otherwise None.
    """
    for box in simple_boxes_list:
        if box.id == target_id:
            return (box.x, box.y, box.w, box.h)
    return None 


# class simple_box:
#     def __init__(self, x, y, width, height, cls=None, box_id=None):
#         """
#         Initialize a simple bounding box with the following attributes:
#         - center: (x, y) coordinates
#         - w: width of the box
#         - h: height of the box
#         - cls: class label
#         - id: unique identifier for the box
#         """
#         self.x = x 
#         self.y = y # x-coordinate of the center
#         self.w = width  # width of the box
#         self.h = height  # height of the box
#         self.cls = cls  # class label
#         self.id = box_id  # unique ID for the box


# def dict_id_center(tracking_list, points):
#     """
#     Calculate the centers of points grouped by their tracking IDs.

#     Parameters:
#     - tracking_list: A list of IDs corresponding to each point.
#     - points: A list of tuples, where each tuple contains (x, y) coordinates of a point.

#     Returns:
#     - A dictionary where keys are IDs and values are the centers of the points corresponding to each ID.
#     """
#     # Initialize an empty dictionary to store centers
#     centers_dict = {}
#     point_of_id_dict= {}

#     # Iterate over unique IDs in the tracking list
#     unique_ids = set(tracking_list)
#     for id_ in unique_ids:
#         # Find all points corresponding to the current ID
#         id_points = [points[i] for i in range(len(tracking_list)) if tracking_list[i] == id_]
        
#         # Calculate the mean (center) of the points
#         center = np.mean(id_points, axis=0)
        
#         # Store the center in the dictionary
#         centers_dict[id_] = tuple(center)
#         point_of_id_dict[id_]= id_points
#         print("id_points", id_points)

#     return centers_dict,point_of_id_dict

# def process_boxes_complete_step_init(list_dict_info,tracking_list,points):

#     '''
#     result: out put of sahi
    
#     #output is list of dict 
#        {1: {'image_id': None,
#     'bbox': [447.6741943359375,
#     309.5724792480469,
#     48.0478515625,
#     32.496734619140625],
#     'score': 0.885761022567749,
#     'category_id': 2,
#     'category_name': 'car',
#     'segmentation': [],
#     'iscrowd': 0,
#     'area': 1561}  }

#             new_id_dict= {'image_id': newid,
#                         'bbox': bbox, # [x,y,w,h]
#                         'score': 0, #float
#                         'category_id': None, #int
#                         'category_name': None, #str
#                         'segmentation': [],
#                         'iscrowd': 0,
#                         'area': 0,
#                         'ori_bbox': bbox, #list [x,y,w,h],
#                         'ori_center_points': groupid_center, #(x,y)
#                         'color': (randint(0, 255),randint(0, 255),randint(0, 255)),
#                         'lastest_point':None,
#                         'disappear_step':None ,   
#                         'points_step_t':list_tuple_five_points
#                         drift:(0,0)
#     '''



#     id_center_dict,point_of_id_dict=dict_id_center(tracking_list,points)
#     print("list_dict_info",list_dict_info)
    
#     for idx,value in enumerate(list_dict_info):
#         value['image_id']=idx+1
#         #value[bbox]=
#         value['ori_center_points']=id_center_dict[idx+1]
#         value['ori_bbox']=value['bbox']
#         value['color']=(randint(0, 255),randint(0, 255),randint(0, 255))
#         value['lastest_point']=None
#         value['disappear_step']=None
#         value['points_step_t']=point_of_id_dict[idx+1]
#         value['drift']=(0,0)
#         if value['score']<0.8:
#             value['visible']=False
#         if value['score']>0.8:
#             value['visible']=True
    
#     list_dict_info=convert_list_dict_to_dict(list_dict_info)

#     print("list_dict_info aaaaaaaaaaa",list_dict_info[2].keys())
    
    
#     return list_dict_info



# def convert_list_dict_to_dict(list_dict_info):
#     end_dict={}
#     for idx,value in enumerate(list_dict_info):
#         end_dict[idx+1]=value

#     print("keyssssssssss",end_dict.keys())
#     return end_dict

def update_bbox_by_image_id(data_list, image_id, new_bbox):
    # can delete the func
    """
    Find the dictionary with the specified image_id in the list and update its bbox.

    Parameters:
    - data_list: List of dictionaries, each representing an object with 'image_id' and 'bbox'.
    - image_id: The image_id to search for.
    - new_bbox: The new bounding box [x, y, w, h] to assign to the matching element.

    Returns:
    - updated_data_list: The updated list of dictionaries.
    """
    # Iterate through the list of dictionaries
    for item in data_list:
        # Check if the image_id matches
        if item['image_id'] == image_id:
            # Update the bbox for the matching element
            item['iscrowd'] = new_bbox
            print(f"Updated bbox for image_id {image_id}: {item['bbox']}")
            break  # Stop once you find the first match (if there are duplicates, remove 'break')

    return data_list


# class generate_video():
#     def create_video_from_images(self,image_folder, video_name):
#         # Get the list of images in the folder
#         images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

#         # Sort the images to ensure they are in the correct order
#         images.sort()

#         # Initialize an empty list to store image paths
#         image_paths = []

#         # Construct the full paths to the images
#         #for img in images:
#         for i in range(len(images)-1):
#             image_paths.append(image_folder+'frame_'+ str(i+1)+'.jpg')

#         # Read images and write to video
#         with imageio.get_writer(video_name, format='mp4',fps=10) as writer:
#             for img_path in image_paths:
#                 image = imageio.imread(img_path)
#                 writer.append_data(image)

#         print(f"Video created: {video_name}")


def save_2d_list_to_txt(data, filename, delimiter="\t"):
    """
    Save a 2D list as a table in a text file.

    Parameters:
    - data (list of lists): The 2D list to save.
    - filename (str): The name of the file to save the data.
    - delimiter (str): The delimiter to use between values (default is tab).
    """
    # Convert to NumPy array if data is purely numeric for convenience
    if all(isinstance(item, (int, float)) for row in data for item in row):
        np.savetxt(filename, np.array(data), fmt="%s", delimiter=delimiter)
    else:
        # For mixed data or non-numeric lists, use plain write method
        with open(filename, "w") as file:
            for row in data:
                line = delimiter.join(map(str, row))  # Join elements in each row
                file.write(line + "\n")
    print(f"Data saved to {filename}")
