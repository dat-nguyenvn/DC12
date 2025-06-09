import numpy as np
import yaml

def need_add_id_and_point(matched_box): #todo: add to check class
    if matched_box!=None:
        if  0 in matched_box:
            return True
    else:
        return False
def iou_calulation(box1, box2, format="topleft"):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    :param box1: List [x, y, w, h], either center-based or top-left format.
    :param box2: List [x, y, w, h], either center-based or top-left format.
    :param format: Specify the format of the boxes, "center" or "topleft".
    :return: IoU value
    """
    if format == "center":
        # Convert from center to top-left format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

    elif format == "topleft":
        # Use as is for top-left format
        x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    else:
        raise ValueError("Invalid format. Use 'center' or 'topleft'.")

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
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute area of union
    union_area = area1 + area2 - intersection_area

    # Compute IoU
    return intersection_area / union_area if union_area > 0 else 0

def need_increase_point_for_id (tracking_list, matched_box,point_per_animal=5):
    id_matched=set(matched_box)
    id_need_increase_point_dict={}
    for i in id_matched:
        if i !=0:
            count=tracking_list.count(i)
            if count <point_per_animal :
                id_need_increase_point_dict[i]=point_per_animal-count

    
    return id_need_increase_point_dict

def compute_centroid(points):
    """
    Compute the centroid (average x and y) of a set of points.

    Parameters:
    - points: List of tuples [(x1, y1), (x2, y2), ...]

    Returns:
    - centroid: Tuple (x_center, y_center)
    """
    points_array = np.array(points)
    centroid_x = np.mean(points_array[:, 0])
    centroid_y = np.mean(points_array[:, 1])
    
    return (centroid_x, centroid_y)


def check_box_overlap(input_box, dict_data,uniqueid_intrack,threshold=0.5):
    """
    Check if the input_box does not overlap with any bounding boxes 
    in dict_data whose keys are in uniqueid_intrack.

    Parameters:
        input_box (list): [x, y, w, h] of the input box.
        dict_data (dict): Dictionary containing bounding boxes with 'bbox' keys.
        uniqueid_intrack (list): List of keys to check in dict_data.
        threshold (float): IoU threshold to consider as overlapping.

    Returns:
        bool: True if no overlap is found; False if overlap exists.
    """
    for key in uniqueid_intrack:
        if key not in dict_data:
            continue
        
        bbox = dict_data[key]['bbox']
        iou = calculate_modified_iou(input_box, bbox)
        #iou = iou_calulation(input_box, bbox)

        # If any IoU exceeds the threshold, return False immediately
        if iou > threshold:
            return False

    # If no overlap exceeded the threshold, return Trues
    return True


def is_not_box_at_edge(bbox, window_size=(640,640)):
    # Extract bounding box parameters from the input array
    x_center, y_center, w, h = bbox
    window_width,window_height=window_size
    # Calculate the top-left and bottom-right coordinates
    x_top_left = x_center - w / 2
    y_top_left = y_center - h / 2
    x_bottom_right = x_center + w / 2
    y_bottom_right = y_center + h / 2
    
    # Check if any corner is at the edge of the detection window
    if x_top_left <= 5 or y_top_left <= 5 or x_bottom_right >= window_width-5 or y_bottom_right >= window_height-5:
        return False
    return True


class simple_box:
    def __init__(self, x, y, width, height, cls=None, box_id=None):
        """
        Initialize a simple bounding box with the following attributes:
        - center: (x, y) coordinates
        - w: width of the box
        - h: height of the box
        - cls: class label
        - id: unique identifier for the box
        """
        self.x = x 
        self.y = y # x-coordinate of the center
        self.w = width  # width of the box
        self.h = height  # height of the box
        self.cls = cls  # class label
        self.id = box_id  # unique ID for the box


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


def generate_high_contrast_colors():
    """
    Generate `n_colors` BGR colors with high contrast against gray.
    """
    gray_value = np.array([128, 128, 128])  # Mid-gray in BGR

    while True:
        # Generate a random BGR color
        color_gen = np.random.randint(0, 256, size=3)
        
        # Check contrast against gray using Euclidean distance
        contrast = np.linalg.norm(color_gen - gray_value)
        
        # Set a contrast threshold (e.g., >100)
        if contrast > 100:
            return (int(color_gen[0]), int(color_gen[1]),int(color_gen[2]))
            
    

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

def find_positions_multiple_values(main_list, values_to_find):
    """
    Finds all positions (indices) of any value from a given set of values in a list.

    Args:
        main_list (list): The list to search within.
        values_to_find (list or single value): A single value or a list of values
                                               whose positions are to be found.

    Returns:
        list: A list of indices where any of the values_to_find are found,
              sorted in ascending order of index.
    """
    positions = []
    # Ensure values_to_find is an iterable (e.g., convert single value to a list)
    if not isinstance(values_to_find, (list, tuple, set)):
        values_to_find = [values_to_find]

    values_set = set(values_to_find) # Use a set for faster lookup

    for index, item in enumerate(main_list):
        if item in values_set:
            positions.append(index)
    return positions