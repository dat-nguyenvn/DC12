def crop_bbox(image_np, bbox):
    """
    Crops the image using the bounding box coordinates.

    Parameters:
    - image_np: NumPy array of the image (H, W, C)
    - bbox: list or tuple of bounding box [minx, miny, maxx, maxy]

    Returns:
    - Cropped image as a NumPy array
    """
    minx, miny, maxx, maxy = bbox
    
    # Crop the image using array slicing
    cropped_image = image_np[miny:maxy, minx:maxx]

    return cropped_image

def crop_window(full_frame, center_list,window_size=640):
    cropped_window_list=[]
    #print("center_list",type(center_list))
    for center in center_list:


        #center : tuple (x,y)
        center_x=center[0]
        center_y=center[1]
        #print("center",center)
        # Get the dimensions of the full frame
        frame_height, frame_width = full_frame.shape[:2]
        #print("frame_height, frame_width",frame_height, frame_width)
        # Half of the window size to help with calculations
        half_window_size = window_size // 2

        # Calculate the top-left corner (x, y) of the window
        top_left_x = max(center_x - half_window_size, 0)
        top_left_y = max(center_y - half_window_size, 0)
        #print("top_left_x",top_left_x)
        #print("top_left_y",top_left_y)

        # Calculate the bottom-right corner, ensuring we don't go out of frame bounds
        bottom_right_x = min(center_x + half_window_size, frame_width)
        bottom_right_y = min(center_y + half_window_size, frame_height)
        #print("bottom_right_x",bottom_right_x)
        #print("bottom_right_y",bottom_right_y)
        # Crop the window from the full frame
        cropped_window = full_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cropped_window_list.append(cropped_window)
        #print("cropped_windowcropped_window",cropped_window)


    return cropped_window_list