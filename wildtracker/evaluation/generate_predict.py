def generate_to_mot_format(frame_number, new_id_dict):
    """
    Generate MOT format annotations for predictions from tracker results.
    
    Args:
    - frame_number: The current frame number.
    - new_id_dict: A dictionary where keys are object IDs and values are object information.
    
    Returns:
    - A list of strings in MOT format for the given frame.
    """
    mot_lines = []

    # Iterate over each object in the dictionary
    for object_id, object_data in new_id_dict.items():
        # Extract bounding box (xmin, ymin, width, height)
        bbox = object_data['bbox']
        xmin, ymin, w, h = bbox
        
        # Extract confidence score (used for prediction)
        confidence = object_data.get('score', -1)  # assuming score is the confidence
        
        # Format the MOT line for this object
        mot_line = [frame_number, object_id, xmin, ymin, w, h, confidence, -1, -1, -1]
        
        # Append this line to the mot_lines list
        mot_lines.append(mot_line)
        print("mot_lines",mot_lines)
    
    return mot_lines
