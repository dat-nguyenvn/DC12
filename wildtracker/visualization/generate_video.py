import imageio
import os

class generate_video():
    def create_video_from_images(self,image_folder, video_name):
        # Get the list of images in the folder
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

        # Sort the images to ensure they are in the correct order
        images.sort()

        # Initialize an empty list to store image paths
        image_paths = []

        # Construct the full paths to the images
        #for img in images:
        for i in range(len(images)-1):
            image_paths.append(image_folder+'frame_'+ str(i+1)+'.jpg')

        # Read images and write to video
        with imageio.get_writer(video_name, format='mp4',fps=10) as writer:
            for img_path in image_paths:
                image = imageio.imread(img_path)
                writer.append_data(image)

        print(f"Video created: {video_name}")
