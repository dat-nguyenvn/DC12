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
            image_paths.append(image_folder+'frame_'+ str(i+3)+'.jpg')

        # Read images and write to video
        with imageio.get_writer(video_name, format='mp4',fps=20) as writer:
            for img_path in image_paths:
                image = imageio.imread(img_path)
                writer.append_data(image)

        print(f"Video created: {video_name}")

if __name__ == "__main__":
    gv = generate_video()
    gv.create_video_from_images("/home/ah23975/mypc/2025/papergraphic/DJI_20230719145816_0003_V_video2/outputs/", "/home/ah23975/mypc/2025/papergraphic/DJI_20230719145816_0003_V_video2/output_video.mp4")

