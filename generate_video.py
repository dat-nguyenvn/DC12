'''
import cv2
import os

# Directory containing the images
image_folder = './saved_videos/display_tracks/'

# Define the video filename
video_name = 'output_generate_video.mp4'

# Get the list of images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Get the dimensions of the first image
image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(image_path)
height, width, _ = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

# Iterate through the images and write them to the video
for i in range(len(images)):
    image_path = image_folder+'frame_'+str(i)+'.jpg'
    print(image_path)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the video writer object
video.release()

print(f"Video created: {video_name}")
'''
import os
import imageio

# Function to create video from images in a folder
def create_video_from_images(image_folder, video_name):
    # Get the list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort the images to ensure they are in the correct order
    images.sort()

    # Initialize an empty list to store image paths
    image_paths = []

    # Construct the full paths to the images
    #for img in images:
    for i in range(len(images)):
        image_paths.append(image_folder+'frame_'+ str(i)+'.jpg')

    # Read images and write to video
    with imageio.get_writer(video_name, format='mp4',fps=30) as writer:
        for img_path in image_paths:
            image = imageio.imread(img_path)
            writer.append_data(image)

    print(f"Video created: {video_name}")

# Example usage
image_folder = './saved_videos/display_tracks/'
video_name = 'output_generated_video.mp4'
create_video_from_images(image_folder, video_name)
