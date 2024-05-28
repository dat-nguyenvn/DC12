import os
import torch
import argparse
import imageio.v3 as iio
import random

import cv2
import numpy as np

def prase_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        #default="./assets/apple.mp4",
        #default='/home/src/yolo/DJI_20230607090521_0006_V.mp4',
        #default='/home/src/yolo/trimvideo.mp4',
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default='./checkpoints/cotracker2.pth',
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=10,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()


def draw_cross(img,point,color):
    cv2.line(img, (point[0], point[1]-15), (point[0], point[1]+15), color, 4) 
    cv2.line(img, (point[0]-15, point[1]), (point[0]+15, point[1]), color, 4)

def display_cross(image_path,txt_path,display_tracks,colour_object):
    # image_path='./saved_videos/save_frames/'
    # txt_path='./saved_videos/save_tracks/'
    # display_tracks='./saved_videos/display_tracks/'
    # colour_object = [tuple(random.randint(1, 255) for _ in range(3)) for _ in range(10)]

    for i in range (len(os.listdir(txt_path))):
        image = cv2.imread(image_path+'frame_'+str(i)+'.jpg')
        data = np.loadtxt(txt_path+'frame_'+str(i)+'.txt')
        print("data shape",data.shape)
        for point in data:
            draw_cross(image,tuple((point[0].astype(int),point[1].astype(int))),colour_object[point[2].astype(int)])
            #cv2.circle(image, tuple((point[0].astype(int),point[1].astype(int))), 5, colour_object[point[2].astype(int)], -1) 
        
        
        
        
        cv2.imwrite(display_tracks+'frame_'+str(i)+'.jpg', image)



def display_text(image_path,txt_path,display_tracks,colour_object):
    # image_path='./saved_videos/save_frames/'
    # txt_path='./saved_videos/save_tracks/'
    # display_tracks='./saved_videos/display_tracks/'
    # colour_object = [tuple(random.randint(1, 255) for _ in range(3)) for _ in range(10)]

    for i in range (len(os.listdir(txt_path))):
        image = cv2.imread(image_path+'frame_'+str(i)+'.jpg')
        data = np.loadtxt(txt_path+'frame_'+str(i)+'.txt')
        text1='Frame '+str(i)
        org = (100, 100) #bottom left text
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color1 =  (36,255,50)
        fontScale = 2
        thickness = 7
        x,y,w,h=30,25,400,100 #top left

        image=cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0) , -1)

        image = cv2.putText(image, text1, org, font, fontScale, color1, thickness, cv2.LINE_AA) 
        print("data shape",data.shape)
        for point in data:
            draw_cross(image,tuple((point[0].astype(int),point[1].astype(int))),colour_object[point[2].astype(int)])
    
    #draft note ID
        
        x1,y1=100,180
        x2,y2=x1,y1+100
        draw_cross(image,tuple((x1,y1)),colour_object[1])
        image = cv2.putText(image, 'ID: 01', (x1+30,y1+15), font, fontScale, color1, thickness, cv2.LINE_AA) 
        draw_cross(image,tuple((x2,y2)),colour_object[2])
        image = cv2.putText(image, 'ID: 02', (x2+30,y2+15), font, fontScale, color1, thickness, cv2.LINE_AA) 

    #display_cross(image_path,txt_path,display_tracks,colour_object)
        cv2.imwrite(display_tracks+'frame_'+str(i)+'.jpg', image)

if __name__ == "__main__":
    image_path='./saved_videos/save_frames/'
    txt_path='./saved_videos/save_tracks/'
    display_tracks='./saved_videos/display_tracks/'
    colour_object=[(0,0,0),(25, 224, 174),(237, 5, 47),(191, 106, 222),(22, 184, 7)]
    #colour_object = [tuple(random.randint(1, 255) for _ in range(3)) for _ in range(10)]
    #display_cross(image_path,txt_path,display_tracks,colour_object)
    display_text(image_path,txt_path,display_tracks,colour_object)

    '''    
    image = cv2.imread(image_path)
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = np.loadtxt(txt_path)
    print(data.shape)
    print("data",data)
    for point in data:
        cv2.circle(image, tuple(point.astype(int)), 5, (100, 255, 0), 2) 
    
    with open(txt_path) as f:
        lines=f.readlines()
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
    
    print(myarray)
    print(myarray.shape)
    
    cv2.imwrite('/home/src/tracker/saved_videos/points_image.jpg', image)
    
    #cv2.imshow('RGB Image', image)
    #cv2.waitKey(0)
    '''
