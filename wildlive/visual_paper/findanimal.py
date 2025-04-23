import os
import cv2
import pandas as pd

def crop_objects_from_video(csv_path, video_folder):
    # Check if the frames folder exists
    frames_folder = os.path.join(video_folder, "frames")
    if not os.path.exists(frames_folder):
        print(f"Frames folder not found in video folder: {video_folder}")
        return
    
    # Create a "cropped" folder to save cropped images
    cropped_folder = os.path.join(video_folder, "cropped")
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Read the CSV file
    data = pd.read_csv(csv_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis", "abc"])
    
    # Process each row in the CSV file
    for _, row in data.iterrows():
        frame_index = int(row["frame"])  # Get the frame index
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])  # Get the bounding box
        
        # Apply the condition for cropping (40 < w < 45 or 69 < h < 73)
        if not (40 < w < 50 or 65 < h < 75):
            continue  # Skip this bounding box if the condition is not met
        
        # Construct the frame image path
        frame_path = os.path.join(frames_folder, f"frame_{frame_index}.jpg")
        
        # Check if the frame exists
        if not os.path.exists(frame_path):
            print(f"Frame not found: {frame_path}")
            continue
        
        # Read the frame image
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to read image: {frame_path}")
            continue
        
        # Crop the object from the image
        cropped_img = img[y:y+h, x:x+w]
        
        # Save the cropped image in the "cropped" folder
        cropped_img_path = os.path.join(cropped_folder, f"cropped_{frame_index}.jpg")
        cv2.imwrite(cropped_img_path, cropped_img)
        print(f"Cropped image saved: {cropped_img_path}")

# Example usage
csv_path = "../../demo_data/eval/GT_MOT/DJI_20230719145816_0003_V_video2.csv"  # Replace with the actual path to the CSV file
video_folder = "../../demo_data/eval/videos/DJI_20230719145816_0003_V_video2"  # Replace with the actual path to the video folder
crop_objects_from_video(csv_path, video_folder)

