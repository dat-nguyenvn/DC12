import cv2
from shapely.geometry import Polygon, Point,MultiPolygon
import numpy as np
from abc import ABC, abstractmethod
import random

from wildlive.points_selection.harris import harris_selection_method
        
def apply_processing(method, cropped_image,polygon,minx,miny):
    methods = {
        'harris': harris_selection_method(),
        'edge_detection': harris_selection_method(),
        'blur': harris_selection_method()
    }
    return methods[method].filter_some_points(cropped_image_numpy=cropped_image, polygon= polygon,xmin=minx,ymin=miny )


