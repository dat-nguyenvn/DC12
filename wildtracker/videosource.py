import cv2  
#from jetson_utils import videoSource, videoOutput, Log

import os
import sys
import numpy as np
from natsort import natsorted


# from jetson_utils import videoSource, videoOutput
# import jetson.utils


class videosourceprovider:
    """Base class for frame providers."""
    def get_frame(self):
        """Fetch the next frame."""
        raise NotImplementedError
    def frame_size(self):
        """Fetch the next frame."""
        raise NotImplementedError    
    def get_RGBframe_numpy(self):
        frame=self.get_frame()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image_rgb


class rtsp_stream(videosourceprovider):
    def __init__(self, rtsp_link="rtsp://192.168.144.25:8554/main.264"):
        self.rtsp = videoSource(rtsp_link, argv=sys.argv) 
        self.output = videoOutput('', argv=sys.argv) 
        self.index = 0



    def get_frame(self):
        #format avaiable : rgb8 , rgb32f; 
        # detail check here:https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#rtsp
        frame = self.rtsp.Capture(format='rgb8', timeout=1000)
        np_frame=jetson.utils.cudaToNumpy(frame)
        np_frame=np.ascontiguousarray(np_frame[..., ::-1])
        self.index += 1
          #type cuda image  #RGB
        return np_frame


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
    def __init__(self, input_folder):
        self.image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = natsorted(self.image_files)

        self.index = 0
    def get_frame(self):
        if self.index < len(self.image_files):
            print("self.image_files[self.index]",self.image_files[self.index])
            frame = cv2.imread(self.image_files[self.index])
            self.index += 1
            return frame
        else:
            return None  

    def frame_size(self):
        frame=self.get_frame()
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