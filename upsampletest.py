import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

from ultralytics.utils.ops import non_max_suppression

path_input="/mount/wilddrone/data/frame_1.jpg"
crop_dir="/mount/wilddrone/data/cropped_images1"
normal_pre_dir="/mount/wilddrone/data/cropped_images1/normal_predict"

from PIL import Image
import os
from IPython.display import display

def crop_and_save_with_overlap(image_path="frame_1.jpg", output_dir="cropped_images1", target_size=(640, 640), overlap=10):
    """
    Crops a large image into multiple images with a specified overlap,
    saves them, and then displays them in Google Colab.

    Args:
        image_path (str, optional): Path to the input image file.
                                     Defaults to "frame_0.jpg".
        output_dir (str, optional): Directory to save the cropped images.
                                     Defaults to "cropped_images".
        target_size (tuple, optional): The desired size of the cropped images (width, height).
                                       Defaults to (640, 640).
        overlap (int, optional): The number of pixels to overlap between adjacent crops.
                                  Defaults to 10.
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        target_width, target_height = target_size

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        cropped_image_paths = []
        count = 0
        for y in range(0, height - target_height + 1, target_height - overlap):
            for x in range(0, width - target_width + 1, target_width - overlap):
                # Define the bounding box for the crop
                left = x
                top = y
                right = x + target_width
                bottom = y + target_height

                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))

                # Save the cropped image
                output_path = os.path.join(output_dir, f"cropped_{count}.png")
                cropped_img.save(output_path)
                cropped_image_paths.append(output_path)
                count += 1

        print(f"Successfully cropped '{image_path}' into {count} images with a {overlap}-pixel overlap and saved them in '{output_dir}'.")

        # Display the cropped images in Colab
        print("\nDisplaying the cropped images:")
        for path in cropped_image_paths:
            display(Image.open(path))

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Make sure 'frame_0.jpg' is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

crop_and_save_with_overlap(image_path=path_input,output_dir=crop_dir,overlap=10)