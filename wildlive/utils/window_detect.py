import random
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, abstractmethod

class generate_centers():
    def generate_tile_centers(self,frame_width, frame_height, tile_width=640, tile_height=640, overlap_width=10, overlap_height=0):
        centers = []
        
        # Calculate the number of tiles needed along width and height
        num_tiles_x = (frame_width + tile_width - overlap_width - 1) // (tile_width - overlap_width)
        num_tiles_y = (frame_height + tile_height - overlap_height - 1) // (tile_height - overlap_height)
        
        # Calculate actual step sizes (this will help distribute overlap in the center)
        step_x = (frame_width - tile_width) // (num_tiles_x - 1) if num_tiles_x > 1 else frame_width
        step_y = (frame_height - tile_height) // (num_tiles_y - 1) if num_tiles_y > 1 else frame_height
        
        # Calculate tile centers based on the number of tiles
        for j in range(num_tiles_y):
            for i in range(num_tiles_x):
                # Calculate the center of the tile
                x_center = tile_width // 2 + i * step_x
                y_center = tile_height // 2 + j * step_y
                centers.append((x_center, y_center))

        return centers

    def generate_tile_centers_border_and_salient(self, frame_width, frame_height, tile_width=640, tile_height=640):
        centers = []
        border_centers = []
        salient_centers = []

        # Compute the optimal overlap to balance coverage
        num_tiles_x = (frame_width + tile_width - 1) // tile_width
        num_tiles_y = (frame_height + tile_height - 1) // tile_height

        step_x = (frame_width - tile_width) // max(num_tiles_x - 1, 1)
        step_y = (frame_height - tile_height) // max(num_tiles_y - 1, 1)

        # Generate tile centers
        for j in range(num_tiles_y):
            for i in range(num_tiles_x):
                x_center = min(tile_width // 2 + i * step_x, frame_width - tile_width // 2)
                y_center = min(tile_height // 2 + j * step_y, frame_height - tile_height // 2)
                centers.append((x_center, y_center))

                # Classify as border or salient
                if i == 0 or i == num_tiles_x - 1 or j == 0 or j == num_tiles_y - 1:
                    border_centers.append((x_center, y_center))
                else:
                    salient_centers.append((x_center, y_center))

        return centers, border_centers, salient_centers


    def visual(self, frame_width, frame_height, centers, border_centers, salient_centers, tile_width=640, tile_height=640):
        """
        Visualizes the tile centers and draws 640x640 windows with different colors.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title("Tile Centers and Windows Visualization")
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)  # Invert y-axis to match image coordinates
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

        # Plot all windows and centers
        for x, y in centers:
            rect_color = 'green' if (x, y) in salient_centers else 'red'  # Red for border, Green for salient
            rect = patches.Rectangle((x - tile_width // 2, y - tile_height // 2), tile_width, tile_height, 
                                     linewidth=2, edgecolor=rect_color, facecolor='none')
            ax.add_patch(rect)
            ax.scatter(x, y, color=rect_color, s=50)
            ax.text(x + 10, y - 10, f"({x}, {y})", fontsize=8, color=rect_color)

        # Show the visualization
        plt.grid(True)
        plt.show()


def strategy_pick_window(step, list_all_center, border_centers, salient_centers, current_track_points):
    big_list = [border_centers, salient_centers, current_track_points]
    current_list = big_list[step % 3]

    try:
        if step % 3 == 0:  # Border
            if current_list:  # Check if current_list is not empty
                center = current_list[step // 3 % len(current_list)]
                win_color = (255, 255, 0)
                print("000 center", center)
            else:
                raise ValueError("Border centers list is empty.")

        elif step % 3 == 1:  # Salient
            if current_list:  # Check if current_list is not empty
                center = current_list[step // 3 % len(current_list)]
                win_color = (0, 255, 0)
                print("111 center", center)
            else:
                raise ValueError("Salient centers list is empty.")

        else:  # step % 3 == 2
            if current_track_points:  # Check if current_track_points is not empty
                center = random.choice(current_track_points)
                win_color = (255, 0, 0)
                print("strategy center", center)
            else:
                raise ValueError("Current track points list is empty.")

    except ValueError as e:
        print(f"Error: {e}. Using default strategy.")
        if current_track_points:  # Fallback to track points if available
            center = random.choice(current_track_points)
        elif list_all_center:  # Fallback to any center if available
            center = random.choice(list_all_center)
        else:  # Final fallback to a default value
            center = (640, 640)
        win_color = (255, 0, 0)

    return center, win_color



# Step 1: Define the base strategy interface
class base_strategy(ABC):
    @abstractmethod
    def execute(self):
        pass


class Strategy1(base_strategy):
    def execute(self,step, list_all_center, border_centers, salient_centers, current_track_points):
        cen=[]
        #def strategy_pick_window(step, list_all_center, border_centers, salient_centers, current_track_points):
        big_list = [border_centers, salient_centers, current_track_points]
        current_list = big_list[step % 3]

        try:
            if step % 3 == 0:  # Border
                if current_list:  # Check if current_list is not empty
                    center = current_list[step // 3 % len(current_list)]
                    win_color = (255, 255, 0)
                    print("000 center", center)
                else:
                    raise ValueError("Border centers list is empty.")

            elif step % 3 == 1:  # Salient
                if current_list:  # Check if current_list is not empty
                    center = current_list[step // 3 % len(current_list)]
                    win_color = (0, 255, 0)
                    print("111 center", center)
                else:
                    raise ValueError("Salient centers list is empty.")

            else:  # step % 3 == 2
                if current_track_points:  # Check if current_track_points is not empty
                    center = random.choice(current_track_points)
                    win_color = (255, 0, 0)
                    print("strategy center", center)
                else:
                    raise ValueError("Current track points list is empty.")

        except ValueError as e:
            print(f"Error: {e}. Using default strategy.")
            if current_track_points:  # Fallback to track points if available
                center = random.choice(current_track_points)
            elif list_all_center:  # Fallback to any center if available
                center = random.choice(list_all_center)
            else:  # Final fallback to a default value
                center = (640, 640)
            win_color = (255, 0, 0)
        
        
        cen.append(center)
        return cen, win_color
        #return "Executing Strategy 1"
    
class Strategy2(base_strategy):
    def execute(self, step, list_all_center, border_centers, salient_centers, current_track_points):
        cen = []  # List to hold centers (tuples)
        
        # First center: Randomly pick from current_track_points
        if current_track_points:
            center_1 = random.choice(current_track_points)  # Randomly pick from current_track_points
            win_color = (255, 0, 0)  # Red color for the first center
            #print("First center (from current_track_points):", center_1)
        else:
            # Fallback if current_track_points is empty
            center_1 = (640, 640)
            win_color = (255, 0, 0)  # Red
            #print("Error: current_track_points is empty. Using default center:", center_1)
        
        cen.append(center_1)  # Add first center to the list
        
        # Second center: Randomly pick from the combined list of border_centers and salient_centers
        combined_centers = border_centers + salient_centers
        if combined_centers:
            center_2 = random.choice(combined_centers)  # Randomly pick from combined centers
            #print("Second center (from combined border or salient centers):", center_2)
        else:
            # Fallback if combined_centers is empty
            center_2 = (640, 640)
            #print("Error: border_centers and salient_centers are empty. Using default center:", center_2)
        
        cen.append(center_2)  # Add second center to the list

        # Return the list of centers and the color (same for both centers)
        return cen, win_color
    

class Strategy4(base_strategy):
    def execute(self, step, list_all_center, border_centers, salient_centers, current_track_points):
        cen = []  # List to hold centers (tuples)
        
        # Combine border_centers and salient_centers into one list
        combined_centers = border_centers + salient_centers

        # Ensure that there are enough centers in both lists
        if len(combined_centers) < 2 or len(current_track_points) < 2:
            raise ValueError("Not enough centers in combined_centers or current_track_points for strategy.")

        # Select 2 unique centers from combined_centers (border_centers + salient_centers)
        selected_combined_centers = random.sample(combined_centers, 2)
        #print("Selected centers from combined_centers:", selected_combined_centers)

         
        selected_track_points = random.sample(current_track_points, 2)
        #print("Selected centers from current_track_points:", selected_track_points)


        selected_centers = selected_combined_centers + selected_track_points
        # Append all selected centers to the cen list
        cen.extend(selected_centers)
        # Define the window color (same color for all centers)
        win_color = (255, 0, 0)  # Red for all centers

        # Return the list of 4 centers and the window color
        return cen, win_color  
    
class Strategy8(base_strategy):
    def execute(self, step, list_all_center, border_centers, salient_centers, current_track_points):
        cen = []  # List to hold centers (tuples)
        
        # Combine border_centers and salient_centers into one list
        combined_centers = border_centers + salient_centers
        
        # Calculate how many centers to pick from current_track_points
        num_current_track_points = len(current_track_points) // 5
        
        # Ensure the number of centers to pick from current_track_points is between 1 and 4
        num_current_track_points = max(1, min(num_current_track_points, 4))
        
        # Select centers from current_track_points
        selected_current_track_points = random.sample(current_track_points, num_current_track_points)
        #print(f"Selected {num_current_track_points} centers from current_track_points:", selected_current_track_points)
        
        # Calculate the number of remaining centers to pick from combined_centers
        remaining_centers_needed = 8 - num_current_track_points
        
        # Select remaining centers from combined_centers
        selected_combined_centers = random.sample(combined_centers, remaining_centers_needed)
        #print(f"Selected {remaining_centers_needed} centers from combined_centers:", selected_combined_centers)
        
        # Combine all selected centers
        selected_centers = selected_current_track_points + selected_combined_centers
        
        # Append all selected centers to the cen list
        cen.extend(selected_centers)
        
        # Define the window color (same color for all centers)
        win_color = (255, 0, 0)  # Red for all centers
        
        # Return the list of 8 centers and the window color
        return cen, win_color  
    
class Strategy16(base_strategy):
    def execute(self, step, list_all_center, border_centers, salient_centers, current_track_points):
        cen = []  # List to hold centers (tuples)
        
        # Combine border_centers and salient_centers into one list
        combined_centers = border_centers + salient_centers
        
        # Calculate how many centers to pick from current_track_points
        num_current_track_points = len(current_track_points) // 5
        
        # Ensure the number of centers to pick from current_track_points is between 1 and 4
        num_current_track_points = max(1, min(num_current_track_points, 8))
        
        # Select centers from current_track_points
        selected_current_track_points = random.sample(current_track_points, num_current_track_points)
        #print(f"Selected {num_current_track_points} centers from current_track_points:", selected_current_track_points)
        
        # Calculate the number of remaining centers to pick from combined_centers
        remaining_centers_needed = 16 - num_current_track_points
        
        # Select remaining centers from combined_centers
        selected_combined_centers = random.sample(combined_centers, remaining_centers_needed)
        #print(f"Selected {remaining_centers_needed} centers from combined_centers:", selected_combined_centers)
        
        # Combine all selected centers
        selected_centers = selected_current_track_points + selected_combined_centers
        
        # Append all selected centers to the cen list
        cen.extend(selected_centers)
        
        # Define the window color (same color for all centers)
        win_color = (255, 0, 0)  # Red for all centers
        
        # Return the list of 8 centers and the window color
        return cen, win_color  
class Strategyfull(base_strategy):
    def execute(self, step, list_all_center, border_centers, salient_centers, current_track_points):
        cen = []  # List to hold centers (tuples)
        
        # Combine border_centers and salient_centers into one list
        combined_centers = border_centers + salient_centers
        
        # Calculate how many centers to pick from current_track_points
        num_current_track_points = len(current_track_points) // 5
        
        # Ensure the number of centers to pick from current_track_points is between 1 and 4
        num_current_track_points = max(1, min(num_current_track_points, 12))
        
        # Select centers from current_track_points
        selected_current_track_points = random.sample(current_track_points, num_current_track_points)
        #print(f"Selected {num_current_track_points} centers from current_track_points:", selected_current_track_points)
        
        # Calculate the number of remaining centers to pick from combined_centers
        remaining_centers_needed = 24 - num_current_track_points
        
        # Select remaining centers from combined_centers
        selected_combined_centers = random.sample(combined_centers, remaining_centers_needed)
        #print(f"Selected {remaining_centers_needed} centers from combined_centers:", selected_combined_centers)
        
        # Combine all selected centers
        selected_centers = selected_current_track_points + selected_combined_centers
        
        # Append all selected centers to the cen list
        cen.extend(selected_centers)
        
        # Define the window color (same color for all centers)
        win_color = (255, 0, 0)  # Red for all centers
        
        # Return the list of 8 centers and the window color
        return cen, win_color     
    

class StrategySelector:
    def __init__(self):
        self.strategies = {
            1: Strategy1(),
            2: Strategy2(),
            4: Strategy4(),
            8: Strategy8(),
            16: Strategy16(),
            24: Strategyfull()

        }

    def get_strategy(self, strategy_number):
        return self.strategies.get(strategy_number, 1)