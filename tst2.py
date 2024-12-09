import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_high_contrast_colors():
    """
    Generate a BGR color with high contrast against gray.
    The function will keep generating random colors until one with high contrast is found.
    Returns the color in BGR tuple format (int, int, int).
    """
    gray_value = np.array([128, 128, 128])  # Mid-gray in BGR

    while True:
        # Generate a random BGR color
        color_gen = np.random.randint(0, 256, size=3)  # This gives an array, which needs to be converted to tuple
        
        # Check contrast against gray using Euclidean distance
        contrast = np.linalg.norm(color_gen - gray_value)
        
        # Set a contrast threshold (e.g., >100)
        if contrast > 100:
            return tuple(color_gen)  # Return color as a tuple (B, G, R)

def visualize_shapely_polygon_on_image(result_image, polygon_points, thickness=5):
    """
    Draw a polygon on the image using a high-contrast color generated against gray.
    """
    # Generate a high-contrast color for the polygon
    color = generate_high_contrast_colors()
    
    # Ensure the color is in the proper BGR tuple format
    if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color):
        print(f"Generated Color (BGR): {color}")
    else:
        raise ValueError(f"Color is not in the correct format: {color}")
    
    # Convert polygon points to integer values (just in case)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    
    # Draw the polygon with the generated color
    cv2.polylines(result_image, [polygon_points], isClosed=True, color=color, thickness=thickness)
    
    # Return the image with the polygon drawn
    return result_image

# Example usage:
# Create a blank image (500x500 pixels, 3 color channels)
result_image = np.zeros((500, 500, 3), dtype=np.uint8)

# Define the points for the polygon (4 points for a square)
polygon_points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.int32)

# Draw the polygon on the image
thickness = 5
result_image_with_polygon = visualize_shapely_polygon_on_image(result_image, polygon_points, thickness)

# Display the result using matplotlib
# Convert BGR image to RGB for matplotlib
result_image_rgb = cv2.cvtColor(result_image_with_polygon, cv2.COLOR_BGR2RGB)

# Show the image using matplotlib
plt.imshow(result_image_rgb)
plt.axis('off')  # Hide the axes for better visual
plt.title('Polygon with High-Contrast Color')
plt.show()
