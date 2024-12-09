import numpy as np
def need_add_id_and_point(matched_box): #todo: add to check class
    if matched_box!=None:
        if  0 in matched_box:
            return True
    else:
        return False


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


def check_box_overlap(input_box, dict_data, threshold=0.6):
    """
    Check if the input_box overlaps with any bounding boxes in the dictionary.
    
    Args:
    - input_box: The box to compare, given as (x_center, y_center, w, h).
    - dict_data: The dictionary of annotations where each entry contains a 'bbox' key.
    - threshold: The IoU threshold to check for overlap.
    
    Returns:
    - True if no overlap exceeds the threshold, False if any overlap exceeds the threshold.
    """
    for item in dict_data.values():
        bbox = item['bbox']  # Extract the bounding box from the dict
        iou = calculate_modified_iou(input_box, bbox)  # Calculate IoU
        
        # If IoU exceeds the threshold, return False
        if iou > threshold:

            return False
    
    # If no overlap exceeded the threshold, return True
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
            
    
    