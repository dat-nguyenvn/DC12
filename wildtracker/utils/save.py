import os
import numpy as np
import pickle
import json
def convert_to_serializable(obj):
    """
    Recursively converts NumPy data types to Python native types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    return obj

def save_in_step(output_dir, index, dict_obj, array_obj, list_obj):
    """
    Save a dictionary, a NumPy array, and a list to separate files in specified subdirectories.

    Args:
        output_dir (str): Base output directory.
        index (int): Index to include in the file names.
        dict_obj (dict): Dictionary to be saved.
        array_obj (numpy.ndarray): NumPy array of shape (n, 2) to be saved.
        list_obj (list): List to be saved.
    """
    # Create subdirectories if they don't exist
    dict_dir = os.path.join(output_dir, "dict")
    point_dir = os.path.join(output_dir, "point")
    trackinglist_dir = os.path.join(output_dir, "trackinglist")

    os.makedirs(dict_dir, exist_ok=True)
    os.makedirs(point_dir, exist_ok=True)
    os.makedirs(trackinglist_dir, exist_ok=True)

    # Construct file names
    dict_filename = os.path.join(dict_dir, f"frame_{index}.json")
    array_filename = os.path.join(point_dir, f"frame_{index}.npy")
    list_filename = os.path.join(trackinglist_dir, f"frame_{index}.pkl")

    # Convert dictionary to serializable format
    serializable_dict = convert_to_serializable(dict_obj)

    # Save dictionary as JSON
    with open(dict_filename, 'w') as dict_file:
        json.dump(serializable_dict, dict_file)

    # Save NumPy array as .npy file
    np.save(array_filename, array_obj)

    # Save list as a pickle file
    with open(list_filename, 'wb') as list_file:
        pickle.dump(list_obj, list_file)