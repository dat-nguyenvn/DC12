import sys
import argparse

from jetson_utils import videoSource, videoOutput

# parse command line
#parser = argparse.ArgumentParser()
#parser.add_argument("input", type=str, help="URI of the input stream")
#parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
#args = parser.parse_known_args()[0]

# create video sources & outputs
inputcam = videoSource("rtsp://192.168.144.25:8554/main.264",  argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
output = videoOutput("rtsp://192.168.144.23:1234/output", argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}
#output = videoOutput("abc.mp4", argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}
# capture frames until end-of-stream (or the user exits)
while True:
    # format can be:   rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be:  -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    image = inputcam.Capture(format='rgb8', timeout=1000)  
	
    if image is None:  # if a timeout occurred
        continue
		
    output.Render(image)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
