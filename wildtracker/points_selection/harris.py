import cv2
import numpy as np


from wildtracker.points_selection.base_selection import selected_point_method

class harris_selection_method(selected_point_method):
    def process(self,cropped_image_numpy,blockSize=5, ksize=3, k=0.04, threshold=0.2):
        print("cropped_image_numpy",cropped_image_numpy.shape)
        gray_image = cv2.cvtColor(cropped_image_numpy, cv2.COLOR_BGR2GRAY)

        # Convert to float32 for cornerHarris
        gray_image = np.float32(gray_image)

        # Apply Harris corner detection
        dst = cv2.cornerHarris(gray_image, blockSize, ksize, k)

        # Dilate the result to mark the corners (optional for visualization)
        dst = cv2.dilate(dst, None)

        # Copy the original image for visualization
        result_image = cropped_image_numpy.copy()

        # Find coordinates of corners by applying the threshold
        corners = np.argwhere(dst > threshold * dst.max())

        # Create a list of (x, y, confidence) tuples
        corners_with_confidence = [(int(x), int(y), dst[y, x]) for y, x in corners]

        # Sort the corners by confidence (Harris response) from highest to lowest
        corners_with_confidence.sort(key=lambda x: x[2], reverse=True)

        # Extract only the (x, y) coordinates, sorted by confidence
        corners_sorted = [(x, y) for x, y, _ in corners_with_confidence]

        for x, y in corners_sorted:
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), 2)  # Red circles for corners
 
        return corners_sorted