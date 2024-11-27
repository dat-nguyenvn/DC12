import cv2
from shapely.geometry import Polygon, Point,MultiPolygon
import numpy as np
from abc import ABC, abstractmethod
import random
class selected_point_method(ABC):
    @abstractmethod
    def process(self, cropped_image_numpy):
        pass
    def visual_result(self,cropped_image_numpy):
        result_image = cropped_image_numpy.copy()
        corners_sorted=self.process(cropped_image_numpy)
        for x, y in corners_sorted:
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), 2)  # Red circles for corners
         
        return result_image
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
    def filter_points_inside_polygon(self,points, polygon):
        """
        Filters the list of points to return only those that are inside the given polygon.
        
        Parameters:
        - points (list of tuples): List of (x, y) coordinates representing the points.
        - polygon (shapely.geometry.Polygon): A Shapely Polygon object.

        Returns:
        - inside_points (list of tuples): List of (x, y) coordinates that are inside the polygon.
        """
        inside_points = []

        # Iterate over each point and check if it is inside the polygon
        for point in points:
            shapely_point = Point(point)
            if polygon.contains(shapely_point):
                inside_points.append(point)
        #print("inside_points",len(inside_points))
        return inside_points

    def filter_some_points(self,cropped_image_numpy,polygon,xmin,ymin,margin=-5,number_point_per_animal=5):
        '''
        xmin,ymin : topleft position of window
        polygon in window
        xmin,ymin topleft box
        '''
        corners_sorted=self.process(cropped_image_numpy)
        #filter in polygon
        polygon=polygon.buffer(distance=margin)
        transformed_polygon=self.transform_polygon_window_to_box(polygon,xmin,ymin)
        points_inside_polygon=self.filter_points_inside_polygon(corners_sorted,transformed_polygon)
        try:
            five_points= random.sample(points_inside_polygon,number_point_per_animal)
            return five_points
        except ValueError as e:
            print(f"Error: {e} Can not pick point in the mask")
            return []



