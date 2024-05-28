# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import random
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import time
import cv2
import queue
import threading
from shapely.geometry import Polygon, Point
from collections import Counter
from ultralytics import YOLO
from utils_DC12.object import Animal,Zoo
# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

def cameraStream():
    #stream=cv2.VideoCapture(0)
    cap = VideoCapture(0)
    dim=(640,640)
    number=100
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        #ret,frame = stream.read()
        frame = cap.read()
        print("frame size", frame.shape)
        #frame =np.asarray(frame)
    
        #results = model.predict(frame, save=True,imgsz=2304,show=True,conf=0.6)

def remove_all_items_in_directory(directory):
    """
    Remove all items (files and directories) within a directory.

    Args:
    - directory: Path to the directory.

    Returns:
    - None
    """
    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if it's a file
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove the file

def compute_center_from_xywh(x, y, width, height):
    """
    Compute the center of a bounding box given its coordinates and dimensions.

    Args:
    x (float): The x-coordinate of the top-left corner of the bounding box.
    y (float): The y-coordinate of the top-left corner of the bounding box.
    width (float): The width of the bounding box.
    height (float): The height of the bounding box.

    Returns:
    tuple: A tuple representing the center of the bounding box in the format (center_x, center_y).
    """
    center_x = x + (width / 2)
    center_y = y + (height / 2)
    
    #center_point_box = torch.tensor([[int(center_x),int(center_y)]])
    center_point_box = torch.tensor([[int(x),int(y)]])
    return center_point_box

def support_points(center_point_box,d):
    support_pixels=torch.tensor([[center_point_box[0,0],center_point_box[0,1]+d], 
                                 [center_point_box[0,0]+d,center_point_box[0,1]],
                                 [center_point_box[0,0],center_point_box[0,1]-d],
                                 [center_point_box[0,0]-d,center_point_box[0,1]]
                                  ])
    
    return support_pixels
def five_points(cen,sp):
    print("sp",sp)
    print("cen",cen)
    a=torch.cat((cen,sp),dim=0)
    return a

def input_queries(fr,points):
    #sub_tensor = torch.tensor([fr])
    frame_number = torch.stack([torch.tensor([fr]) for _ in range(points.shape[0])])
    #frame_number.to(DEFAULT_DEVICE)
    result_tensor_with_column = torch.cat((frame_number, points), dim=1)
    return result_tensor_with_column    
def create_grid_point(out_detector,zoo,point_pre_ani=3):
    for r in out_detector:
        #print("results shape",r.orig_img) #orig_img.copy()
        imageread=r.orig_img
        
        
        #print("box shape",r.boxes[0].cpu().numpy())
        #print("",r.masks.xy[0].shape)
        grid_points_frame=[]
        id_frame=[]
        in_track=[]
        for i in range (len(r.masks.xy)):
            #imageread = cv2.imread(source) 
            '''
            for point in r.masks.xy[i]:
                #print("point",point)
                
                point=point.astype(int)
                # print("imageread",imageread.shape)
                # print("type points ",type(point))
                # print("shape points ",point.shape)
                # print(point)
                #print("tuple(int(point))",tuple(point.astype(np.int)))
                
                imageread= cv2.circle(imageread, (point[0],point[1]), 5, (0, 255, 0), -1)
            '''
            start_time = time.time()
            polygon = Polygon(r.masks.xy[i])
            polygon=polygon.buffer(distance=-30)
            grid_spacing = 60
            min_x, min_y, max_x, max_y = polygon.bounds
            grid_points_object = []
            #identify_pixels_object=[]
            id_per_object=[]
            in_track_object=[]
            for x in np.arange(min_x, max_x, grid_spacing):
                for y in np.arange(min_y, max_y, grid_spacing):
                    point = Point(x, y)
                    if polygon.contains(point):
                        grid_points_object.append((x, y))
                        id_per_object.append(i+1)
                        in_track_object.append(int(0))
            #todo
            #print("type grid_points_object",type(grid_points_object))
            #print(" torch.tensor(grid_points_object)", torch.tensor(grid_points_object).float())
            #input("Press Enter to continue...")
            grid_points_object = random.sample(grid_points_object, point_pre_ani)
            id_per_object=random.sample(id_per_object, point_pre_ani)
            in_track_object=random.sample(in_track_object, point_pre_ani)
            id_frame.extend(id_per_object)
            in_track.extend(in_track_object)

            object=Animal(id=i+1,points= torch.tensor(grid_points_object),color=None)  # list tensor 1d to tensor 2d
            zoo.add_animal(object)

            end_time = time.time()
            print("grid_points_object &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",len(grid_points_object))
            elapsed_time = end_time - start_time
            #print("Elapsed Time:", elapsed_time, "seconds")
            #print("grid point 0", type(grid_points_object[0]))
            grid_points_frame.append(grid_points_object)
            print("grid_points_frame", len(grid_points_frame))
            #id_frame.append(identify_pixels_object)
        grid_points_frame = [item for sublist in grid_points_frame for item in sublist]
        print("grid_points_frame", len(grid_points_frame))
        grid_points_frame = torch.tensor(grid_points_frame).float()
        print("grid_points_frame", grid_points_frame.shape)    
        #print("grid_points_frame ***** end", grid_points_frame[1,2,3,4])
            # for i in range (len(grid_points)):
            #     grid_points[i]=tuple(int(x) for x in grid_points[i])
            #     #print(grid_points[i])
            #     imageread= cv2.circle(imageread, grid_points[i], 4, (0, 0, 255), -1)
            
        # cv2.imwrite('./pick/test.jpg', imageread)

    return grid_points_frame,id_frame,in_track  #torch tensor 2d for grid and list for id
def most_common_number(numbers):
    # Count occurrences of each number
    counts = Counter(numbers)
    # Find the most common number(s)
    most_common = counts.most_common(1)
    # If there are ties for the most common number, return all of them
    most_common_numbers = [num for num, count in counts.items() if count == most_common[0][1]]
    return most_common_numbers

def add_new_point_to_track(grid_points_frame,id_frame,in_track,out_detector,mk_id):
    #new_grid,new_id_frame,new_in_track=create_grid_point(out_detector)
    for r in out_detector:
        #print("results shape",r.orig_img) #orig_img.copy()
        #imageread=r.orig_img
        
        
        #print("box shape",r.boxes[0].cpu().numpy())
        #print("",r.masks.xy[0].shape)

        for i in range (len(r.masks.xy)):
            new_grid=[]
            new_id_frame=[]
            new_in_track=[]
            #imageread = cv2.imread(source) 
            '''
            for point in r.masks.xy[i]:
                #print("point",point)
                
                point=point.astype(int)
                # print("imageread",imageread.shape)
                # print("type points ",type(point))
                # print("shape points ",point.shape)
                # print(point)
                #print("tuple(int(point))",tuple(point.astype(np.int)))
                
                imageread= cv2.circle(imageread, (point[0],point[1]), 5, (0, 255, 0), -1)
            '''
            #start_time = time.time()
            polygon = Polygon(r.masks.xy[i])
            polygon=polygon.buffer(distance=-30)
            grid_spacing = 30
            min_x, min_y, max_x, max_y = polygon.bounds
            #identify_pixels_object=[]
            for x in np.arange(min_x, max_x, grid_spacing):
                for y in np.arange(min_y, max_y, grid_spacing):
                    point = Point(x, y)
                    if polygon.contains(point):
                        new_grid.append((x, y))
                        new_id_frame.append(mk_id[i])
                        new_in_track.append(int(0))
            #new_grid = [item for sublist in new_grid for item in sublist]

            new_grid = torch.tensor(new_grid).float()
            print("new_grid",new_grid)
            print("new_grid shape",new_grid.shape)
            random_position_inlist=random.sample(range(len(new_id_frame)), 2)

            for i in random_position_inlist:
                in_track.append(new_in_track[i])
                id_frame.append(new_id_frame[i])
                print("grid_points_frame",grid_points_frame.shape)
                print("new_grid[i]",new_grid[i].shape)
                grid_points_frame=torch.cat((grid_points_frame, new_grid[i].view(1, -1)), dim=0)
    
            print("in_track in addd point",len(in_track))
            print("Id frame in addd point",len(id_frame))
            print("grip point in add point",grid_points_frame.shape)

    
    return grid_points_frame,id_frame,in_track
def add_new_point_to_track2(grid_points_frame:torch.Tensor,id_frame,in_track,out_detector,mk_id,listid_need_add):
    for i, value in enumerate(mk_id):
        if value in listid_need_add[0]:
            #for r in out_detector:
            new_grid=[]
            new_id_frame=[]
            new_in_track=[]
            polygon = Polygon(out_detector[0].masks.xy[i])
            polygon=polygon.buffer(distance=-30)
            grid_spacing = 30
            min_x, min_y, max_x, max_y = polygon.bounds
            for x in np.arange(min_x, max_x, grid_spacing):
                for y in np.arange(min_y, max_y, grid_spacing):
                    point = Point(x, y)
                    if polygon.contains(point):
                        new_grid.append((x, y))
                        new_id_frame.append(mk_id[i])
                        new_in_track.append(int(0))
            new_grid = torch.tensor(new_grid).float()
            print("new_grid",new_grid)
            print("new_grid shape",new_grid.shape)

            random_position_inlist=random.sample(range(len(new_id_frame)), 2)

            for i in random_position_inlist:
                in_track.append(new_in_track[i])
                id_frame.append(new_id_frame[i])
                print("grid_points_frame",grid_points_frame.shape)
                print("new_grid[i]",new_grid[i].shape)
                grid_points_frame=torch.cat((grid_points_frame, new_grid[i].view(1, -1)), dim=0)
    return grid_points_frame,id_frame,in_track

def match_mask_id_and_id_pixels(out_detector, id_frame, grid_points_frame,in_track):
    # grid_points_frame : tensor (number points, 2) [[x,y],[x2,y2]]]
    mk_id=[]
    print("id_frame",id_frame)
    new_id_frame=id_frame
    #grid_points_frame=tuple(tuple(row) for row in grid_points_frame)
    #grid_points_frame.cpu()
    
    #print("grid_points_frame  in matchhhh  ",grid_points_frame.shape)
    #print("grid_points_frame.size()  in matchhhh ",grid_points_frame.size(0))
    for i in range(len(in_track)):
        in_track[i]+=1    
    
    for r in out_detector:
        for i in range (len(r.masks.xy)):
            #print("len(r.masks.xy)",len(r.masks.xy))
            polygon = Polygon(r.masks.xy[i])

            dummy=[]
            dummy_position=[]
            for j in range (grid_points_frame.size(0)):  #number of point
                #print("shape grid_points_frame",grid_points_frame.shape)
                #print("grid_points_frame[j]",grid_points_frame[j])
                point=Point(grid_points_frame[j,0],grid_points_frame[j,1])
                if polygon.contains(point):
                    dummy.append(id_frame[j])
                    dummy_position.append(j)
            

            maskid=most_common_number(dummy)[0]  
            #ToDO: if max < thresshold -> not assert -> add new ID
            #todo have more objects
            mk_id.append(maskid)
            print("len(dummy_position)+++++++++++++++++++++++",len(dummy_position))

            
            
            for k in range (len(dummy_position)):
                #print("grid_points_frame[dummy_position[k]]",grid_points_frame[dummy_position[k]])
                
                #input("Press Enter to continue...")
                new_id_frame[dummy_position[k]]=maskid
                in_track[dummy_position[k]]=0

    
    # print("len(in_track)",len(in_track))
    # for i in range(len(in_track)):
    #     print("in_track[i] -------------------------------------",in_track[i])
    #     if in_track[i]!=0:
    #         in_track+=1
    #print("mk_iDDDDDDDDDDDd",mk_id)
    return new_id_frame,in_track,mk_id #list based on results yolo
def remove_pixels_not_intrack(grid_points_frame,new_id_frame,in_track,limit=10):
    print("in_track", type(in_track))
    a=len(in_track)
    print("aaaaaaaaa",a)
    elements_to_remove=[]
    for i in range (a):
        if in_track[i]>=limit:
            elements_to_remove.append(i)
            #print("vailonnnnnnnnnnnnnnnnnnnnnnnnnnnn")
    for i in range (len(elements_to_remove)):        
        del in_track[i]
        del new_id_frame[i]
        grid_points_frame = torch.cat((grid_points_frame[:i], grid_points_frame[i + 1:]), dim=0)

    return grid_points_frame,new_id_frame,in_track

def make_parse():
    parser=argparse.ArgumentParser("DC12 Demo !")
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        #default='/home/src/yolo/DJI_20230607090521_0006_V.mp4',
        #default='/home/src/yolo/trimvideo.mp4',
        #default='/home/src/yolo/lowresolution.mp4',
        #default='/home/src/yolo/DJI_20230607092838_0003_V.mp4',
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default='./checkpoints/cotracker2.pth',
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--save",
        default=False,
        help="Save frame and track points",
    )    
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=20,
        help="Compute dense and grid tracks starting from this frame",
    )

    return parser

def main(args):
    


    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame,quetest):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            #grid_query_frame=grid_query_frame,
            queries=quetest[None],
        )
    #model_detector = YOLO('yolov8l.pt')
    model_detector = YOLO('yolov8x-seg.pt')
    
    
    #results=model.predict(source, save=True, classes=[20,22,23,47],conf=0.25,imgsz=640)
    '''
    queries_new = torch.tensor([
    [0, 2520., 604.],
    [0, 2522., 606.],
    [0, 2524., 608.],
    [0, 2526., 604.],
    [0, 2026., 1629.],
    [0, 2028., 1629.], 
    [0, 2024., 1629.], 
    [0, 2026., 1625.],   # point tracked from the first frame # frame number 10 
    [0, 2026., 1633.]])
    '''
    queries_new= torch.tensor([[0,300.,300.],
                            [0,400.,400.],
                            [0,500.,500.],
                            [0,450.,450.],
                            [0,250.,250.]]) #float point number is madatory
    
    # queries = torch.tensor([
    # [0, 250., 300.],  - Khoảng 70% Graduate Visa holders là người Ấn, Pakistan & Nigeri
    '''
    pred_tracks_numpy=pred_tracks.cpu().numpy()
    save_track_path='./saved_videos/save_tracks'
    num_files = len(os.listdir(save_track_path))
    for l in range (pred_tracks_numpy.shape[1]):
        np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt',\
                    pred_tracks_numpy[0,l].astype(int), fmt='%d')
    '''        
    # [0, 2526., 604.],
    # [0, 250., 250.],
    # [0, 2028., 1629.],     # point tracked from the first frame # frame number 10 
    # [0, 300., 300.]])
    #queries_new = torch.zeros((30, 3))
    #queries_new=torch.cat((queries_test,queries_new), dim=0)
    #replace_position=2
    queries_new= queries_new.cuda()
    
    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    #video = cv2.VideoCapture(args.video_path)
    #if not video.isOpened():
    #    print("Error: Could not open video file.")
    #for i in range(20):
    #    ret, frame = video.read()
        #frame = torch.tensor(frame)
    #    print("type", type(frame))

    remove_all_items_in_directory("./saved_videos/save_tracks")
    remove_all_items_in_directory("./saved_videos/save_frames")
    remove_all_items_in_directory("./saved_videos/display_tracks")
    zoo=Zoo()

    for i, frame in enumerate(iio.imiter(args.video_path,plugin="FFMPEG")):
        image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #for yolo.
        #print("queries_newwwwwwwwwwwwwwwwwwwww", queries_new[2,1])
        
        if i==0:
            print("1111111111111111111111111")
            results=model_detector.predict(image_bgr,save=True, classes=[20,22,23,47],conf=0.25,imgsz=640) #47 apple  # results : list (1 image len = 1)
            print("hahaaaa",type(results))
            print("lenn ",len(results))
            #input("Press Enter to continue...")
            que2,id_pixels,intrack_pixels= create_grid_point(results,zoo)
            
            que2=input_queries(0,que2)
            #que2=que2.cuda()
            queries_new=que2.cuda()
            #queries_new=torch.cat((queries_new, que2), dim=0) #su dung neu class k co trong coco data
            print("shape queries_new",queries_new.shape)
            print("len id pixels topppppppppppppppppp",len(id_pixels))
        
        '''    
            for r in results:
                if r.boxes.xywh.numel()!=0:
                    for k in range (r.boxes.xywh.shape[0]):
                        cen_points=compute_center_from_xywh(r.boxes.xywh[k,0],r.boxes.xywh[k,1],r.boxes.xywh[k,2],r.boxes.xywh[k,3])
                        que=cen_points
                        sup_points=support_points(cen_points,5)
                        que=five_points(cen_points,sup_points)
                        que2=input_queries(0,que)
                        que2=que2.cuda()
                        queries_new=torch.cat((queries_new, que2), dim=0)
                        #queries_new=input_queries(i,que) # note
                        #queries_new=queries_new.cuda()
                        
                        print("queries_new",queries_new)
        '''
        print("image_bgr shape",image_bgr.shape)
        if args.save:
            print("o kiaaaaaa")
            cv2.imwrite('./saved_videos/save_frames/frame_'+str(i)+'.jpg', image_bgr)

        if i % model.step == 0 and i != 0:
            #print("3333333333")
            #image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            '''
            if i % 80 ==0:
                results=model_detector.predict(image_bgr,save=True, classes=[20,22,23,47],conf=0.25,imgsz=640) #47 apple
                queries_new=pred_tracks[0,pred_tracks.shape[1]-1].cpu()
                queries_new=input_queries(0,queries_new)
                queries_new=queries_new.cuda()
                for r in results:
                    print("rrrrrr")
                    if r.boxes.xywh.numel()!=0:
                        for k in range (r.boxes.xywh.shape[0]):
                            print("box",r.boxes.xywh)
                            print("x", r.boxes.xywh[k,1])
                            cen_points=compute_center_from_xywh(r.boxes.xywh[k,0],r.boxes.xywh[k,1],r.boxes.xywh[k,2],r.boxes.xywh[k,3])
                            print('cen_points',cen_points)
                            que=cen_points
                            
                            #print("cen_points",cen_points)
                            sup_points=support_points(cen_points,5)
                            que=five_points(cen_points,sup_points)
                            #print("type five_points",type(five_points))
                            
                            que2=input_queries(0,que) #when detect cotrack restart so frame =0
                            que2=que2.cuda()
                            
                            #queries_new=pred_tracks[0,pred_tracks.shape[1]-1].cpu()
                            #queries_new=input_queries(0,queries_new)
                            #queries_new=queries_new.cuda()
                            queries_new=torch.cat((queries_new, que2), dim=0)
                        #queries_new[replace_position]=que2
                        #replace_position+=1
                        print("queries new shape",queries_new.shape)
                        
                        is_first_step = True
                        
                        del window_frames[:50]
                        if queries_new.shape[0]>30:
                            queries_new=queries_new[:10]
                    pred_tracks_numpy=pred_tracks.cpu().numpy()
                    save_track_path='./saved_videos/save_tracks'id_per_object
                    num_files = len(os.listdir(save_track_path))
                    for l in range (pred_tracks_numpy.shape[1]):
                        np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt',\
                                   pred_tracks_numpy[0,l].astype(int), fmt='%d')
            '''    
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,            # if queries_new.shape[0]>200:
            #     num_rows_to_keep = queries_new.size(0) - 20 
            #     indices_to_keep = random.sample(range(queries_new.size(0)), num_rows_to_keep)
            #     queries_new=queries_new[indices_to_keep]
                grid_size=args.grid_size,
                #grid_query_frame= queries[None],
                grid_query_frame= args.grid_query_frame,
                quetest=queries_new,
            )
            is_first_step = False
        ###################################################    
        if i % 40 == 0 and i!=0:
            print("if 4000000000000000")               
            #print("gagag")
            pred_tracks_numpy=pred_tracks.cpu().numpy()
            #print("shape eeeeeeeee pred_tracks_numpy",pred_tracks_numpy.shape)
            #print("pred_tracks_numpy[0,l]", pred_tracks_numpy[0,2].shape)
            test_id=np.transpose(np.array([id_pixels]))
            print("shape test id",test_id.shape)
            #print("shape test id",test_id)
            save_track_path='./saved_videos/save_tracks'
            num_files = len(os.listdir(save_track_path))
            if args.save:
                for l in range (pred_tracks_numpy.shape[1]): #l is step/frame
                    #print("pred_tracks_numpy[0,l] in forrrrr ", pred_tracks_numpy[0,l].shape)
                    pred_tracks_numpy_save = np.hstack((pred_tracks_numpy[0,l], test_id))
                    #print("pred_tracks_numpy saveeeeeeeeeee", pred_tracks_numpy.shape)
                    #np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt',\
                    #            pred_tracks_numpy[0,l].astype(int), fmt='%d')
                    np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt',\
                                pred_tracks_numpy_save.astype(int), fmt='%d')
            
            queries_new=pred_tracks[0,pred_tracks.shape[1]-1].cpu()
            queries_new=input_queries(0,queries_new)
            
            results=model_detector.predict(image_bgr,save=True, classes=[20,22,23,47],conf=0.25,imgsz=640) #47 apple
            print("len intrack_pixels ",type(intrack_pixels))
            #print("results ",results)
            #print("results[0].masks",results[0].masks)
            #print("results[0].masks.xy",results[0].masks.xy)
            
            if results[0].masks !=None:
                id_pixels,intrack_pixels,mask_id=match_mask_id_and_id_pixels(results, id_pixels, queries_new[:, -2:].cpu(),intrack_pixels) #maskid tuong ung voi mask after yolo8
                # intrack =0 is tracking (in mask) =2 mean no in mask 2 times
                print("len id pixels",len(id_pixels))
                print("len intrack_pixels ",len(intrack_pixels))
                print("shape after iDDDDDD queries_new",queries_new.shape)
                #queries_new,id_pixels,intrack_pixels
                que2,id_pixels,intrack_pixels=remove_pixels_not_intrack(queries_new[:, -2:].cpu(),id_pixels, intrack_pixels)
                # todo zoo append, count point / ID
                zoo.arrange_point_to_object(que2,id_pixels)
                zoo.display_all_animals()
                animal_need_add_point_list=zoo.find_animal_need_add_point(point_per_object=4)
                
                # if len(animal_need_add_point_list)!=0:
                #     #todo: 
                #     que2,id_pixels,intrack_pixels=add_new_point_to_track2(que2,id_pixels,intrack_pixels,results,mask_id,animal_need_add_point_list)



                print("afterrrrrrrrrrrr  *******       len id pixels",len(id_pixels))
                print("que2 ",type(que2))
                print("shape after iDDDDDD queries_new",que2.shape)
                #input("Press Enter to continue...")
                
                #if que2.size(0)<70:
                #    que2,id_pixels,intrack_pixels=add_new_point_to_track(que2,id_pixels,intrack_pixels,results,mask_id) 
                    
                # que2,id_pixels,intrack_pixels=add_new_point_to_track(que2,id_pixels,intrack_pixels,results,mask_id) 
                # que2=input_queries(0,que2)
                que2=input_queries(0,que2)
                queries_new=que2.cuda()
            queries_new=queries_new.cuda()
            #ueries_new=queries_new.cuda()


            '''
            for r in results:
                print("haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                if r.boxes.xywh.numel()!=0:
                    for k in range (r.boxes.xywh.shape[0]):
                        print("box",r.boxes.xywh)
                        print("x", r.boxes.xywh[k,1])
                        cen_points=compute_center_from_xywh(r.boxes.xywh[k,0],r.boxes.xywh[k,1],r.boxes.xywh[k,2],r.boxes.xywh[k,3])
                        print('cen_points',cen_points)
                        que=cen_points
                        
                        #print("cen_points",cen_points)
                        sup_points=support_points(cen_points,5)
                        que=five_points(cen_points,sup_points)
                        #print("type five_points",type(five_points))
                        
                        que2=input_queries(0,que) #when detect cotrack restart so frame =0
                        que2=que2.cuda()
                        
                        #queries_new=pred_tracks[0,pred_tracks.shape[1]-1].cpu()
                        #queries_new=input_queries(0,queries_new)
                        #queries_new=queries_new.cuda()
                        queries_new=torch.cat((queries_new, que2), dim=0)
                    #queries_new[replace_position]=que2
                    #replace_position+=1
                    print("queries new shape",queries_new.shape)
            '''    
            is_first_step = True
                
            window_frames=[]

            # if queries_new.shape[0]>200:
            #     num_rows_to_keep = queries_new.size(0) - 20 
            #     indices_to_keep = random.sample(range(queries_new.size(0)), num_rows_to_keep)
            #     queries_new=queries_new[indices_to_keep]
            '''
            pred_tracks_numpy=pred_tracks.cpu().numpy()
            save_track_path='./saved_videos/save_tracks'
            num_files = len(os.listdir(save_track_path))
            for l in range (pred_tracks_numpy.shape[1]):
                np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt',\
                            pred_tracks_numpy[0,l].astype(int), fmt='%d')
            '''        
        ############################################
                    
                    
                            
        window_frames.append(frame)
        print("window frame len", len(window_frames))

    # Processing the final video frames in case video length is not a multiple of model.step
    print("done step , print i", i)
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        #grid_query_frame= queries[None],
        grid_query_frame= args.grid_query_frame, 
        quetest=queries_new,
    )
    
    print("Tracks are computed")
    
    # print("pred_tracks type ",type(pred_tracks))
    # print("pred_tracks shape ",pred_tracks.shape)
    # pred_tracks_numpy=pred_tracks.cpu().numpy()
    # save_track_path='./saved_videos/save_tracks'
    # num_files = len(os.listdir(save_track_path))
    # for l in range (pred_tracks_numpy.shape[1]):
    #     print('pred_tracks_numpy[0,l].astype(int)',pred_tracks_numpy[0,l].astype(int))
    #     np.savetxt('./saved_videos/save_tracks/frame_'+str(num_files+l)+'.txt', \
    #                 pred_tracks_numpy[0,l].astype(int), fmt='%d')
    
    print("DONE")
    #save a video with predicted tracks
    # seq_name = args.video_path.split("/")[-1]
    # video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
    # vis = Visualizer(save_dir="./saved_videos", pad_value=2, linewidth=3,tracks_leave_trace=-1)  #tracks_leave_trace=-1 to see the trace
    # vis.visualize(video, pred_tracks, pred_visibility)

if __name__ == "__main__":
    args=make_parse().parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print("Total time",end_time-start_time)


