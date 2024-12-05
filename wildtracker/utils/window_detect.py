import random


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

    def generate_tile_centers_border_and_salient(self,frame_width, frame_height, tile_width=640, tile_height=640, overlap_width=10, overlap_height=0):
        centers = []
        border_centers = []
        salient_centers = []
        
        # Calculate the number of tiles needed along width and height
        num_tiles_x = (frame_width + tile_width - overlap_width - 1) // (tile_width - overlap_width)
        num_tiles_y = (frame_height + tile_height - overlap_height - 1) // (tile_height - overlap_height)
        
        # Calculate actual step sizes (this will help distribute overlap in the center)
        step_x = (frame_width - tile_width) // (num_tiles_x - 1) if num_tiles_x > 1 else frame_width
        step_y = (frame_height - tile_height) // (num_tiles_y - 1) if num_tiles_y > 1 else frame_height
        
        # Calculate tile centers and categorize them
        for j in range(num_tiles_y):
            for i in range(num_tiles_x):
                # Calculate the center of the tile
                x_center = tile_width // 2 + i * step_x
                y_center = tile_height // 2 + j * step_y
                centers.append((x_center, y_center))
                
                # Determine if the center is at an edge
                if i == 0 or i == num_tiles_x - 1 or j == 0 or j == num_tiles_y - 1:
                    border_centers.append((x_center, y_center))
                else:
                    salient_centers.append((x_center, y_center))

        return centers, border_centers, salient_centers


def strategy_pick_window(step,list_all_center,border_centers,salient_centers,current_track_points ):
    big_list=[border_centers,salient_centers,current_track_points]
    current_list = big_list[step % 3]

    # if step%3==0: #border
    #     center=current_list[step // 3 % len(current_list)]
    #     win_color=(255, 255, 0)
    # elif step%3==1:  #salient
    #     center=current_list[step // 3 % len(current_list)]
    #     win_color=(0, 255, 0)
    # else : #step%3==2:
    #     center =random.choice(current_track_points)
    #     win_color=(255, 0, 0)

    center=random.choice(list_all_center)
    # center=random.choice(current_track_points)
    win_color=(255, 0, 0)
    return center,win_color
