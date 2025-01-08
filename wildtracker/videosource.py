import cv2  
#from jetson_utils import videoSource, videoOutput, Log

import os
import sys
import numpy as np
from natsort import natsorted

from jetson_utils import videoSource, videoOutput
import jetson.utils



class videosourceprovider:
    """Base class for frame providers."""

    def __init__(self, source_type):
        self.instance = self.create_video_source(source_type)

    def create_video_source(self, source_type):
        """
        Create an appropriate video source based on the input string.

        Parameters:
        source_type (str): A string describing the source (e.g., "rtsp://...", "camera", or "/path/to/folder").
        kwargs (dict): Additional configuration options for each source type.

        Returns:
        VideoSourceProvider: An instance of the appropriate video source class.
        """
        if "rtsp" in source_type.lower():
            # RTSP stream
            print("hahah rtsp hree")
            return rtsp_stream(rtsp_link=source_type)
        elif "camera" in source_type.lower():
            # USB camera
            #camera_index = kwargs.get('camera_index', 0)
            return usb_camera()
        elif source_type.startswith("/"):
            # Folder path
            print("hahainhere")
            return input_folder(folder=source_type)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def get_frame(self):
        """Fetch the next frame."""
        return self.instance.get_frame()
    def frame_size(self):
        """Fetch the next frame."""
        return self.instance.frame_size()    
    def get_RGBframe_numpy(self):
        # frame=self.get_frame()
        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.instance.get_RGBframe_numpy()
    def index(self):
        return self.instance.index


class rtsp_stream(videosourceprovider):
    def __init__(self, rtsp_link="rtsp://192.168.144.25:8554/main.264"):
        # from jetson_utils import videoSource, videoOutput
        # import jetson.utils
        #self.jetson_utils = jetson.utils
        self.rtsp = videoSource(rtsp_link, argv=sys.argv) 
        #self.output = videoOutput('', argv=sys.argv) 
        self.index = 0



    # def get_frame(self):
    #     #format avaiable : rgb8 , rgb32f; 
    #     # detail check here:https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtsp
    #     frame = self.rtsp.Capture(format='rgb8', timeout=1000)
    #     np_frame=jetson.utils.cudaToNumpy(frame)
    #     np_frame=np.ascontiguousarray(np_frame[..., ::-1])
    #     self.index += 1
    #       #type cuda image  #RGB
    #     return np_frame

    def get_frame(self):
        try:
            # Capture frame
            frame = self.rtsp.Capture(format='rgb8', timeout=1000)
            
            # Debugging: Check frame type
            if frame is None:
                raise ValueError("No frame captured. Check RTSP stream.")
            
            print("Captured frame type:", type(frame))
            
            # Convert to NumPy array
            np_frame = jetson.utils.cudaToNumpy(frame)
            print("Converted frame shape:", np_frame.shape)

            # Make contiguous and convert RGB to BGR
            np_frame = np.ascontiguousarray(np_frame[..., ::-1])
            self.index += 1
            
            return np_frame
        except Exception as e:
            print("Error in get_frame:", e)
            raise

    def get_cuda_frame(self):
        #format avaiable : rgb8 , rgb32f; 
        # detail check here:https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtsp
        frame = self.rtsp.Capture(format='rgb8', timeout=1000)
        return frame

    def get_RGBframe_numpy(self):
        frame=self.get_frame()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #np_frame=jetson.utils.cudaToNumpy(frame)
        return image_rgb
    def get_frame_numpy(self): #BGR
        frame=self.get_frame()

        np_frame=jetson.utils.cudaToNumpy(frame)
        np_frame=np.ascontiguousarray(np_frame[..., ::-1])
        return np_frame


    def frame_size(self):
        frame=self.get_frame()
        height, width, channels = frame.shape
        return height, width, channels

class usb_camera(videosourceprovider):
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None  # No frame captured

    def release(self):
        self.cap.release()             

class input_folder(videosourceprovider):
    def __init__(self, folder):
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = natsorted(self.image_files)

        self.index = 0
    def get_frame(self):
        if self.index < len(self.image_files):
            print("self.image_files[self.index]",self.image_files[self.index])
            frame = cv2.imread(self.image_files[self.index])
            self.index += 1
        return frame  

    def frame_size(self):
        self.index =0
        frame=cv2.imread(self.image_files[self.index])

        #self.index =0
        height, width, channels = frame.shape
        #channels=3
        return width, height,channels 

    def get_RGBframe_numpy(self):
        frame=self.get_frame()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image_rgb



#clastoutput
def main():
    input= rtsp_stream(rtsp_link="rtsp://192.168.144.25:8554/main.264")

    #input = videoSource("rtsp://192.168.144.25:8554/main.264", argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
    output = videoOutput('', argv=sys.argv)
    while True:
        img = input.get_frame()
        #img = input.Capture()
        output.Render(img)
    
        # update the title bar
        #output.SetStatus("Video Viewer | {:d}x{:d} |".format(img.width, img.height))


if __name__ == '__main__':
    main()
