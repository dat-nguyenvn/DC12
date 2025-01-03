import os
import json
import csv

# Define the folder path where your .json files are located
#json_folder = '/media/ah23975/Crucial X9/captest/eval/frames/'  # Change this to your actual folder path
json_folder = '/data/captest/cap_eval_data/vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V/frames/'  # Change this to your actual folder path

output_csv_file = 'ground_truth_vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V.csv'  # Output file for MOT format

# Function to convert the bounding box points to MOT format (xmin, ymin, width, height)
def convert_to_bbox(points):
    # Points are given in a clockwise or counter-clockwise manner
    # Calculate xmin, ymin, xmax, ymax and then width, height
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    
    width = xmax - xmin
    height = ymax - ymin
    
    return xmin, ymin, width, height

# Function to process each .json file and convert to MOT format
def convert_json_to_mot(json_folder, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['frame_id', 'object_id', 'xmin', 'ymin', 'width', 'height', 'confidence', '-1', '-1', '-1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop through all json files in the folder
        for json_file in os.listdir(json_folder):
            if json_file.endswith('.json'):
                frame_id = int(json_file.split('_')[1].split('.')[0])  # Extract the frame number from the filename
                json_path = os.path.join(json_folder, json_file)
                
                # Read the .json file
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract object annotations
                for shape in data['shapes']:
                    # Convert the points into bounding box coordinates
                    xmin, ymin, width, height = convert_to_bbox(shape['points'])
                    object_id = shape['group_id']
                    confidence = 1  # Assuming all bounding boxes have a confidence of 1

                    # Write the row to the CSV file
                    writer.writerow({
                        'frame_id': frame_id,
                        'object_id': object_id,
                        'xmin': xmin,
                        'ymin': ymin,
                        'width': width,
                        'height': height,
                        'confidence': confidence,
                        '-1': -1,
                        '-1': -1,
                        '-1': -1
                    })

# Run the conversion function
convert_json_to_mot(json_folder, output_csv_file)

print(f"Ground truth has been converted to MOT format and saved to {output_csv_file}")
