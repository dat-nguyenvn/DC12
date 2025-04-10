from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolov8x-seg.pt") #
import time
# Generate two random numpy arrays as example images
image1 = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)  # Random image
image2 = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
images = [image1, image2,image1,image2,image1,image2,image1,image1,image2,image1,image1,image2]



start_time = time.time()  # Start timing
results = model.predict(image1, save=False, verbose=False)  # Set `save=False` to disable auto-saving
end_time = time.time()  # End timing
print(f"\n Single Inference Time: {end_time - start_time:.4f} seconds")

start_time = time.time()  # Start timing
results = model.predict(images, save=False, verbose=False,batch=12)  # Set `save=False` to disable auto-saving
end_time = time.time()  # End timing
print(f"\n Batch Inference Time: {end_time - start_time:.4f} seconds")


start_time = time.time()  # Start timing
results = model.predict(image1, save=False, verbose=False)  # Set `save=False` to disable auto-saving
end_time = time.time()  # End timing
print(f"\n Single Inference Time: {end_time - start_time:.4f} seconds")