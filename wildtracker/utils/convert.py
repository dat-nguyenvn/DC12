from shapely.geometry import Polygon, Point,MultiPolygon
import numpy as np

class convert_process():
    def transform_polygon_window_to_box(self,polygon, xmin, ymin):
        """
        Transforms a polygon from the original frame to the cropped frame by adjusting coordinates.
        
        Parameters:
        - polygon (shapely.geometry.Polygon): The polygon in the original frame.
        - xmin (int or float): The x-coordinate of the top-left corner of the cropped region.
        - ymin (int or float): The y-coordinate of the top-left corner of the cropped region.

        Returns:
        - transformed_polygon (shapely.geometry.Polygon): The polygon transformed to the cropped frame.
        """
        # Get the coordinates of the polygon
        #print("polygon in trasform",polygon)
        if isinstance(polygon, MultiPolygon):
            polygon=polygon.geoms[0]
            original_coords = list(polygon.exterior.coords)
            cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]
            transformed_polygon = Polygon(cropped_coords)
            
        else:
            original_coords = list(polygon.exterior.coords)

            # Transform each point in the polygon to the cropped frame
            cropped_coords = [(x - xmin, y - ymin) for x, y in original_coords]

            # Create a new polygon with the transformed coordinates
            transformed_polygon = Polygon(cropped_coords)

        return transformed_polygon
    def convert_points_box_to_full_frame(self,cropped_points, xmin, ymin):
        """
        Converts points from the cropped frame back to the original image frame.

        Parameters:
        - cropped_points (list of tuples): List of (x, y) coordinates in the cropped frame.
        - xmin (int or float): x-coordinate of the top-left corner of the bounding box in the original frame.
        - ymin (int or float): y-coordinate of the top-left corner of the bounding box in the original frame.

        Returns:
        - original_points (list of tuples): List of (x, y) coordinates in the original frame.
        """
        original_points = [(x + xmin, y + ymin) for (x, y) in cropped_points]
        return original_points    
    def convert_featurecpu_to_list_tuple(self,featurecpu):
        list_of_tuples = [tuple(map(int, row)) for row in featurecpu.tolist()]
        return list_of_tuples
    def convert_polygon_window_to_full_frame(self,cropped_polygon,center_point,frame_size=640):
        center_x, center_y = center_point
        top_left_x = center_x - frame_size//2
        top_left_y = center_y - frame_size//2
        # Extract the coordinates of the polygon
        cropped_coords = list(cropped_polygon.exterior.coords)
        
        # Add the top-left offset to each vertex
        big_frame_coords = [(x + top_left_x, y + top_left_y) for x, y in cropped_coords]
        
        # Create a new polygon in the big frame using the transformed coordinates
        big_frame_polygon = Polygon(big_frame_coords)
        
        return big_frame_polygon
    
    
    def convert_point_window_to_full_frame(self,points_in_window, window_center, window_size=640):
        # Extract center coordinates
        center_x, center_y = window_center
        
        # Calculate the top-left corner of the window
        
        top_left_x = center_x - window_size // 2
        top_left_y = center_y - window_size // 2
        
        # Create a list to hold the converted positions
        points_in_full_frame = []
        
        # Convert each position in the window to the larger frame
        for (x, y) in points_in_window:
            # Adjust the position based on the top-left corner
            big_x = top_left_x + x
            big_y = top_left_y + y
            points_in_full_frame.append((int(big_x), int(big_y)))
        
        return points_in_full_frame
        

    def convert_bounding_boxes_to_big_frame(self,bboxes, window_center, window_size):
        """
        Convert bounding boxes from a smaller window to a larger frame.
        
        Parameters:
        - bboxes: np.ndarray of shape (n, 4) with each row as [x, y, w, h]
        - window_center: Tuple[int, int], the center (x, y) of the window in the big frame
        - window_size: Tuple[int, int], size of the window (width, height)
        - big_frame_size: Tuple[int, int], size of the big frame (width, height)

        Returns:
        - List of converted bounding boxes in the big frame as [(x, y, w, h), ...]
        """
        window_center_x, window_center_y = window_center
        window_width, window_height = window_size
        
        
        # Calculate scaling factors

        top_left_x=window_center_x - window_width/2
        top_left_y=window_center_y - window_height/2

        big_frame_bboxes = []
        
        for box in bboxes:
            x, y, w, h = box

            # Convert to big frame coordinates
            big_frame_x = top_left_x + x
            big_frame_y = top_left_y + y
            
            # Scale the width and height
            big_w = w 
            big_h = h
            
            # Append the new bounding box to the list
            big_frame_bboxes.append((big_frame_x, big_frame_y, big_w, big_h))

        return big_frame_bboxes
    def convert_points_from_big_to_window(self,points, center, window_size=(640, 640)):
        #keep point in window only
        # Calculate half-width and half-height of the window
        half_width, half_height = window_size[0] // 2, window_size[1] // 2
        
        # Calculate the bounds of the small window
        x_min = center[0] - half_width
        x_max = center[0] + half_width
        y_min = center[1] - half_height
        y_max = center[1] + half_height
        
        # # Filter points that are inside the small window
        # mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
        #     (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        # cropped_points = points[mask]
        
        # Translate the points to the small window's coordinate system
        translated_points = points - np.array([x_min, y_min])
        
        # Convert to list of tuples
        return [tuple(point) for point in translated_points]
    def convert_center_to_topleft(self,x_center, y_center, w, h):
        x_topleft = x_center - w / 2
        y_topleft = y_center - h / 2
        return [x_topleft, y_topleft, w, h]


    def convert_xywh_to_top_left(self,xywh):
        """
        Convert bounding box format from (x_center, y_center, width, height)
        to (x_top_left, y_top_left, width, height) for a single NumPy array.

        Parameters:
            xywh (np.ndarray): A NumPy array of shape [1, 4] representing (x_center, y_center, width, height).
        
        Returns:
            np.ndarray: A NumPy array of shape [1, 4] representing (x_top_left, y_top_left, width, height).
        """
        x_center, y_center, width, height = xywh
        x_top_left = x_center - (width / 2)
        y_top_left = y_center - (height / 2)
        return np.array([x_top_left, y_top_left, width, height])

