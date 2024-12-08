import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point,MultiPolygon

from wildtracker.utils.convert import convert_process

class visual_image():
    def visualize_shapely_polygon_on_image(self,image_np, shapely_polygon, color=(0, 255, 0), thickness=2):
        """
        Visualizes a Shapely polygon on an image by drawing its outline.

        Parameters:
        - image_np (np.array): Input image in numpy array format (BGR).
        - shapely_polygon (shapely.geometry.Polygon): Shapely polygon object.
        - color (tuple): Color of the polygon outline in BGR format (default is green).
        - thickness (int): Thickness of the polygon outline (default is 2).

        Returns:
        - result_image (np.array): Image with the polygon drawn on it.
        """
        # Extract the polygon points and convert to a numpy array of type int32

        polygon_points = np.array(shapely_polygon.exterior.coords, dtype=np.int32)
        
        # Reshape the array to the correct format for polylines
        polygon_points = polygon_points.reshape((-1, 1, 2))

        # Draw the polygon outline on the image
        result_image = image_np.copy()
        cv2.polylines(result_image, [polygon_points], isClosed=True, color=color, thickness=thickness)

        return result_image
    def show_image(self,numpy_image):
        #numpy_image : numpy 3d array 
        # rgb color
        plt.imshow(numpy_image)
        plt.show()

    def visualize_points_on_image(self,image_np, points, color=(0, 0, 255), radius=5, thickness=-1):
        """
        Visualizes points on an image by drawing circles at each point.

        Parameters:
        - image_np (np.array): Input image in numpy array format (BGR).
        - points (list of tuples): List of (x, y) coordinates to visualize.
        - color (tuple): Color of the points (default is red in BGR format).
        - radius (int): Radius of the circles representing the points (default is 5).
        - thickness (int): Thickness of the circle outline. Use -1 for filled circles (default is -1).

        Returns:
        - result_image (np.array): Image with the points drawn on it.
        """
        # Create a copy of the original image to draw on
        copy_image = image_np.copy()

        # Iterate over each point and draw a circle
        for point in points:
            #print("*****point",point)
            #print("copy_image",copy_image.shape)
            cv2.circle(copy_image, (int(point[0]),int(point[1])), radius, color, thickness)

        return copy_image
    

    def draw_bounding_boxes_on_image(self,image, bboxes, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on the image.

        Parameters:
        - image: The original image (as a NumPy array).
        - bboxes: List of bounding boxes in the format [(x, y, w, h), ...]. x,y center of box 
        - color: Color of the bounding box in BGR format.
        - thickness: Thickness of the bounding box lines.
        
        Returns:
        - Image with bounding boxes drawn.
        """
        for (x, y, w, h) in bboxes:
            #print("(x, y, w, h) ",(x, y, w, h) )
            # Calculate the top-left corner from center (x, y)
            top_left_x = int(x-w/2)
            top_left_y = int(y-h/2)
            bottom_right_x = int(x + w/2)
            bottom_right_y = int(y + h/2)

            # Draw the rectangle on the image
            cv2.rectangle(image, (top_left_x, top_left_y), 
                        (bottom_right_x, bottom_right_y), 
                        color, thickness)
        
        return image    
    
    def draw_info_from_main_dict(self,np_image,data_list):
        """
        Draw bounding boxes on the image based on the data_list.
        
        Parameters:
        - image: NumPy array representing the image (BGR format).
        - data_list: main dict dict of dictionaries with 'bbox' information. Each dictionary contains:
        'bbox': [x_top_left, y_toplet, width, height].
        """
        if type(data_list) is not dict:
            for item in data_list:
                
                bbox = item['bbox']
                x_top_left, y_top_left, w, h = bbox
                
                # Convert bbox from center (x, y) to top-left (x1, y1) for OpenCV's rectangle function
                x1 = int(x_top_left)
                y1 = int(y_top_left)
                x2 = int(x_top_left + w )
                y2 = int(y_top_left + h )
                
                # Draw rectangle on the image (blue color with thickness 2)
                cv2.rectangle(np_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                
                # Optionally, display the category name at the top of the bounding box
                category_name = item.get('category_name', 'Unknown')
                category_name='animal'
                cv2.putText(np_image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, color=(255, 0, 0), thickness=1)      


        if type(data_list) is dict:
            for key, item in data_list.items():
                print(item)
                bbox = item['bbox']
                x_top_left, y_top_left, w, h = bbox
                
                # Convert bbox from center (x, y) to top-left (x1, y1) for OpenCV's rectangle function
                x1 = int(x_top_left)
                y1 = int(y_top_left)
                x2 = int(x_top_left + w )
                y2 = int(y_top_left + h )
                
                # Draw rectangle on the image (blue color with thickness 2)
                cv2.rectangle(np_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                
                # Optionally, display the category name at the top of the bounding box
                category_name = item.get('category_name', 'Unknown')
                category_name='animal'
                cv2.putText(np_image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, color=(255, 0, 0), thickness=1)   

        return np_image

    def draw_one_id_on_image(self,image, bbox, image_id, color,conf='0',thickness=10):
        """
        Draw a bounding box with a label on the image.
        Args:
        - image (np.array): Image to draw on
        - bbox (list): Bounding box [x, y, w, h] x,y, topleft
        - image_id (int): The label or ID to display on the bounding box
        - color (tuple): Color for the bounding box (BGR format)
        """
        # Unpack bbox coordinates
        x, y, w, h = bbox
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)


        # Draw rectangle (bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 10
        text = f'ID: {image_id} ; conf: {conf:.2f}'
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 0 else y + text_size[1] + 10
        
        # Draw text (image_id)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)
        return image
    def visual_bounding_box_of_dict(self,dict_inside,np_image,tracking_list):

        unique_values = set(tracking_list)
        #print("unique_values in reconstruct ",unique_values)
        out_img=np_image
        for unique in unique_values:
            '''
            for item in dict_inside:
                if item['image_id'] == unique:
                    #for obj in dict_inside:
                    out_img=self.draw_one_id_on_image(np_image, item['bbox'], item['image_id'], item['color'])
            '''
            if unique in dict_inside: 
                item = dict_inside[unique]
                if item['visible']==True:
                    out_img=self.draw_one_id_on_image(np_image, item['bbox'], item['image_id'], item['color'],item['score'])
                elif item['visible']==False:
                    out_img=self.draw_one_id_on_image(np_image, item['bbox'], ' ' , (211, 211, 211),item['score'])



        return out_img
    def draw_pixels_with_colors(self,image, featurecpu, id_list_intrack, new_id_dict_list):
        """
        Draws pixels on the image with colors corresponding to their IDs.
        
        Args:
            image (np.ndarray): The input image on which to draw pixels.
            featurecpunumpy array (nx2): representing pixel positions.
            id_list_intrack (list of int): A list of IDs corresponding to each pixel in pixel_positions.
            new_id_dict_list (list of dict): A list of dictionaries containing 'image_id' and 'color' fields.
        
        Returns:
            np.ndarray: The image with colored pixels.
        """
        pixel_positions=convert_process().convert_featurecpu_to_list_tuple(featurecpu)
        # Create a dictionary to map image_id to color from the list of dictionaries
        id_to_color = {obj['image_id']: obj['color'] for obj in new_id_dict_list.values()}

        # Iterate through the pixel positions and corresponding IDs
        for i, (x, y) in enumerate(pixel_positions):
            image_id = id_list_intrack[i]  # Get the ID for this pixel position
            color = id_to_color.get(image_id, (0, 0, 0))  # Get the color for this ID, default to black if not found
            
            # Draw the pixel by setting the color at the pixel location
            cv2.circle(image, (x,y), 10, color, -1)
        return image

    def draw_all_on_window(self,yolo_output,box_matched,dict_inside,points,window_center,tracking_list):
        np_img = yolo_output[0].orig_img
        points_in_window=convert_process().convert_points_from_big_to_window(points,window_center)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        #check if have time 
        for dummy,value in enumerate(box_matched):
            #color_id=find_color_by_image_id(dict_inside,value)  #backup
            if value!=0:

                color_id=dict_inside[value]['color']
                box=yolo_output[0].boxes.xywh.cpu().numpy()[dummy]

                np_img=self.visualize_shapely_polygon_on_image(np_img, Polygon(yolo_output[0].masks.xy[dummy]),color_id,5)
                np_img=self.draw_bounding_boxes_on_image(np_img,[box],color_id,1)
                box=convert_process().convert_center_to_topleft(box[0],box[1],box[2],box[3])
                np_img=self.draw_one_id_on_image(np_img,box,value,color_id,5)
            if value==0:

                color_id=(0,0,0)
                box=yolo_output[0].boxes.xywh.cpu().numpy()[dummy]

                np_img=self.visualize_shapely_polygon_on_image(np_img, Polygon(yolo_output[0].masks.xy[dummy]),color_id,5)
                np_img=self.draw_bounding_boxes_on_image(np_img,[box],color_id,1)
                box=convert_process().convert_center_to_topleft(box[0],box[1],box[2],box[3])
                np_img=self.draw_one_id_on_image(np_img,box,value,color_id,5)                

        for idx,point in enumerate(points_in_window):
            
            value=tracking_list[idx]
            #color_id=find_color_by_image_id(dict_inside,value) #backup
            color_id=dict_inside[value]['color']
            np_img=self.visualize_points_on_image(np_img,[point],color_id,radius=8)

        #print("np_img************",np_img.shape)

        
        
        return np_img



    # def draw_window(self,image, center, size=(640,640)):
    #     """
    #     Draws a rectangle on the image given the center and size.
        
    #     Args:
    #         image (np.ndarray): The input image on which to draw the rectangle.
    #         center (tuple): The (x, y) coordinates of the center of the rectangle.
    #         size (tuple): The (width, height) of the rectangle.
        
    #     Returns:
    #         np.ndarray: The image with the drawn rectangle.
    #     """
    #     # Unpack center and size
    #     center_x, center_y = center
    #     width, height = size

    #     # Calculate top-left and bottom-right corners
    #     top_left = (center_x - width // 2, center_y - height // 2)
    #     bottom_right = (center_x + width // 2, center_y + height // 2)

    #     # Draw the rectangle on the image
    #     cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=15)  # Red color

    #     return image  


    def draw_window(self,image, center, size=(640,640), dash_length=25, spacing=30,color=(255, 0, 0)):
        """
        Draws a dashed rectangle on the image given the center and size.
        
        Args:
            image (np.ndarray): The input image on which to draw the dashed rectangle.
            center (tuple): The (x, y) coordinates of the center of the rectangle.
            size (tuple): The (width, height) of the rectangle.
            dash_length (int): The length of each dash.
            spacing (int): The spacing between dashes.
        
        Returns:
            np.ndarray: The image with the drawn dashed rectangle.
        """
        # Unpack center and size
        center_x, center_y = center
        width, height = size

        # Calculate top-left and bottom-right corners
        top_left = (center_x - width // 2, center_y - height // 2)
        bottom_right = (center_x + width // 2, center_y + height // 2)

        # Get the coordinates of the rectangle's corners
        points = [
            (top_left[0], top_left[1]),                        # Top-left
            (bottom_right[0], top_left[1]),                    # Top-right
            (bottom_right[0], bottom_right[1]),                # Bottom-right
            (top_left[0], bottom_right[1]),                    # Bottom-left
            (top_left[0], top_left[1])                         # Close the rectangle
        ]

        # Draw dashes along each edge
        for i in range(4):
            start_point = points[i]
            end_point = points[i + 1]
            
            # Calculate the total length of the current edge
            edge_length = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))

            # Calculate the number of dashes and spacing
            num_dashes = edge_length // (dash_length + spacing)

            for j in range(num_dashes):
                # Calculate the starting point of the dash
                dash_start_x = int(start_point[0] + j * (dash_length + spacing) * (end_point[0] - start_point[0]) / edge_length)
                dash_start_y = int(start_point[1] + j * (dash_length + spacing) * (end_point[1] - start_point[1]) / edge_length)
                
                # Calculate the ending point of the dash
                dash_end_x = int(dash_start_x + (dash_length * (end_point[0] - start_point[0]) / edge_length))
                dash_end_y = int(dash_start_y + (dash_length * (end_point[1] - start_point[1]) / edge_length))
                
                # Draw the dash
                cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color=color, thickness=15)

        return image


    def add_text_with_background(self,image: np.ndarray, text: str, position=(10, 10), 
                                font_scale=4, font_thickness=5, 
                                text_color=(255, 255, 255), background_color=(0, 0, 0)) -> np.ndarray:
        """
        Adds text with a tightly fitted background rectangle to an image.
        
        Parameters:
            image (np.ndarray): Input image as a NumPy array (BGR format for OpenCV).
            text (str): Text to display.
            position (tuple): Top-left position for the text and rectangle.
            font_scale (int): Scale of the font.
            font_thickness (int): Thickness of the font.
            text_color (tuple): Color of the text in (B, G, R).
            background_color (tuple): Color of the rectangle background in (B, G, R).
            
        Returns:
            np.ndarray: Image with text and background rectangle added.
        """
        # Copy image to avoid modifying the original
        output_image = image.copy()

        # Font for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Define rectangle coordinates with minimal padding
        top_left = position
        bottom_right = (top_left[0] + text_width + 4, top_left[1] + text_height + 8)  # Minimal padding

        # Draw background rectangle
        cv2.rectangle(output_image, top_left, bottom_right, background_color, -1)

        # Define text position inside the rectangle
        text_position = (top_left[0] + 2, top_left[1] + text_height + 4)

        # Put text on top of the rectangle
        cv2.putText(output_image, text, text_position, font, font_scale, text_color, font_thickness)
        
        return output_image
