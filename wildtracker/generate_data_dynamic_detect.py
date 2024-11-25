from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
import time
import random
import argparse
import time
import vpi
import matplotlib.pyplot as plt
import os
import json
import natsort
import imageio
import random

def make_parse():
    parser=argparse.ArgumentParser("Image processing !")
    parser.add_argument("--input_fordel_path",help="test 1 image",type=str,
                        default="/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/frames/")
    parser.add_argument('--pyramid_levels', type=int,default=5,
                    help='Number of levels in the pyramid used with the algorithm')
    #parser.add_argument("--img_batch",help="Load image batch ie:2,4,8",type=int,default=1)
    #parser.add_argument("--img_folder",help="folder path",type=str,default="/opt/nvidia/deepstream/deepstream-5.0/ds/detect_nho_2/segmentation/test/")
    return parser

#def create_grid_point(out_detector,zoo,point_pre_ani=3):

def detect_first_frame(model,input_fordel_path,start_frame,point_pre_ani=10,grid_spacing=10,bound_distance=-10):
    #source = '/home/src/yolo/ultralytics/frame_0.jpg'  
    #source = '/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/frames/frame_0.jpg'
    source =os.path.join(input_fordel_path,'frames/frame_'+str(start_frame)+'.jpg')
    out_detector=model.predict(source,boxes=True, save_crop=True ,show_labels=True,show_conf=False,save=True, classes=[20,22,23],conf=0.35,imgsz=(2160,3840),save_dir='/home/src/yolo/ultralytics/')
    #print(out_detector)
    
    for r in out_detector:  #r la toan bo box trong 1 image
        imageread=r.orig_img
        boxes = r.boxes 
        #print(boxes)
        grid_points_frame=[]
        id_frame=[]
        if len(r.masks.xy)>0:
            for i in range (len(r.masks.xy)):
                polygon = Polygon(r.masks.xy[i])
                polygon=polygon.buffer(distance=bound_distance)
                grid_spacing = 10
                min_x, min_y, max_x, max_y = polygon.bounds
                grid_points_object = []
                id_per_object=[]
                for x in np.arange(min_x, max_x, grid_spacing):
                    for y in np.arange(min_y, max_y, grid_spacing):
                        point = Point(x, y)
                        if polygon.contains(point):
                            grid_points_object.append((x, y))
                            id_per_object.append(i+1) 
        
                grid_points_object = random.sample(grid_points_object, point_pre_ani)
                id_per_object=random.sample(id_per_object, point_pre_ani)
                id_frame.extend(id_per_object)
                #print("id_frame ====",id_frame)
                grid_points_frame.append(grid_points_object)
                #print("grid_points_frame", len(grid_points_frame))
                #print("grid_points_frame ", grid_points_frame)


    grid_points_frame = [item for sublist in grid_points_frame for item in sublist]

    grid_points_frame = np.array(grid_points_frame,dtype=np.float32)
    
    id_frame = np.array(id_frame,dtype=np.float32)
    print("grid_points_frame",grid_points_frame.shape)
    #print("id_frame",id_frame)                        


    return grid_points_frame, id_frame #list , #list
    #selected_points(polygon,)

def save_txt_point(array_info,save_path, name_file):
    file_path = os.path.join(save_path, name_file)
    array_info = array_info.astype(np.int32)

    print(file_path)

    np.savetxt(file_path, array_info, fmt='%d', delimiter=' ')


#def visual_point_on_frame:
def append_column(append_info,input_fordel_path):
    #save_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkout/'
    save_path=os.path.join(input_fordel_path,'lkout/')
    for filename in os.listdir(save_path):
        #print('filename',filename)
        if filename.endswith('.txt'):
            file_path = os.path.join(save_path, filename)
            output_file_path = os.path.join(save_path, filename)

            # Load the existing data
            data = np.loadtxt(file_path)    
            extended_data = np.column_stack((data, append_info))
            np.savetxt(output_file_path, extended_data, fmt='%d')

def draw_black_image():
    width, height = 3840, 2160
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image
def crop_image_from_top_left(image: np.ndarray, top_left_x: int, top_left_y: int, crop_width: int = 480, crop_height: int = 270) -> np.ndarray:
    # Calculate the bottom-right coordinates based on the top-left point and desired crop size
    x2 = top_left_x + crop_width
    y2 = top_left_y + crop_height
    
    # Ensure the crop doesn't exceed the image boundaries
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    
    # Crop the image
    cropped_image = image[top_left_y:y2, top_left_x:x2]
    
    return cropped_image
def create_video_from_images(input_fordel_path,name,start=0):
    image_folder=os.path.join(input_fordel_path,name+'/')
    video_name=os.path.join(input_fordel_path,name+'.mp4')
    # Get the list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort the images to ensure they are in the correct order
    images.sort()

    # Initialize an empty list to store image paths
    image_paths = []

    # Construct the full paths to the images
    #for img in images:
    #for i in range(len(images)):
    for i in range(1,500):
        image_paths.append(image_folder+'frame_'+ str(i+1)+'.jpg')

    # Read images and write to video
    with imageio.get_writer(video_name, format='mp4',fps=30) as writer:
        for img_path in image_paths:
            image = imageio.imread(img_path)
            #image = crop_image_from_top_left(image,1400,200,2250,1200)
            #image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_LINEAR)
            print(type(image))
            writer.append_data(image)

    print(f"Video created: {video_name}")

def generate_random_colors(k):
    """
    Generate k random colors.

    Args:
        k (int): Number of random colors to generate.

    Returns:
        List[Tuple[int, int, int]]: A list of k tuples representing colors in BGR format.
    """
    colors = []
    for _ in range(k):
        # Generate random B, G, R values (each between 0 and 255)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors
def reconstruct_box(color,input_fordel_path):
    # maxid=5
    # info_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkout/'
    # save_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkbox/'
    #save_path_box='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/visual_simple_box/'
    save_path=os.path.join(input_fordel_path,'lkbox/')
    info_path=os.path.join(input_fordel_path,'lkout/')
    save_path_box=os.path.join(input_fordel_path,'visual_simple_box/')
    os.makedirs(save_path_box, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    all_files = os.listdir(info_path)
    txt_files = [file for file in all_files if file.endswith('.txt')]
    #color = generate_random_colors(10)
    print(color[1])
    for file in txt_files:
        file_path=os.path.join(info_path,file)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = np.loadtxt(file_path, dtype=int)
        bounding_boxes = {}
        #print("np.unique(data[:, 2])np.unique(data[:, 2])",max(np.unique(data[:, 2])) )
        for unique_id in np.unique(data[:, 2]):
            print(unique_id)
            points = data[data[:, 2] == unique_id]
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            bounding_boxes[int(unique_id)] = (int(min_x),int( min_y),int(max_x),int(max_y))

        output_path= os.path.join(save_path, f"{base_name}.json")
        #print("bounding_boxes",bounding_boxes)
        with open(output_path, 'w') as json_file:
            json.dump(bounding_boxes, json_file, indent=4)


        image=draw_black_image()
        #fig, ax = plt.subplots(figsize=(12, 6))

        for box_id, coords in bounding_boxes.items():
            print("box_id",box_id)
            if box_id==1:
                print("colorcolor",color[box_id])
            min_x, min_y = coords[0], coords[1]
            max_x, max_y = coords[2], coords[3]
            
            # Generate a random color for each bounding box
            
            
            # Draw the rectangle on the image (top-left corner to bottom-right corner)
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color[box_id],-1)

        # Save or display the image
        #plt.show()
        output_path= os.path.join(save_path_box, f"{base_name}.jpg")
        cv2.imwrite(output_path, image)
    

    return None

def visual_point(color,input_fordel_path):
    
    #info_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkout/'
    #save_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkbox/'
    #save_path_box='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/visual_simple_point/'
    info_path=os.path.join(input_fordel_path,'lkout/')
    save_path_point=os.path.join(input_fordel_path,'visual_simple_point/')
    frames_path=os.path.join(input_fordel_path,'frames/')
    os.makedirs(save_path_point, exist_ok=True)

    all_files = os.listdir(info_path)
    txt_files = [file for file in all_files if file.endswith('.txt')]
    
    print(color[1])
    for file in txt_files:    
        file_path=os.path.join(info_path,file)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = np.loadtxt(file_path, dtype=int)
        image=draw_black_image()
        for k in data:
            #print("k",k)
            #print("data[k][3]",k[3])
            colorid=color[k[3]]
            if k[2]==0:
                cv2.circle(image, (k[0],k[1]), 15, colorid, -1)
            elif k[2]==1:
                cv2.circle(image, (k[0],k[1]), 15, colorid, -1)

        output_path= os.path.join(save_path_point, f"{base_name}.jpg")
        cv2.imwrite(output_path, image)

def visual_point_onframe(color,input_fordel_path):
    
    #info_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkout/'
    #save_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkbox/'
    #save_path_box='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/visual_simple_point/'
    info_path=os.path.join(input_fordel_path,'lkout/')
    save_path_point=os.path.join(input_fordel_path,'visual_simple_point_onframeLK/')
    frames_path=os.path.join(input_fordel_path,'frames/')
    os.makedirs(save_path_point, exist_ok=True)

    all_files = os.listdir(info_path)
    txt_files = [file for file in all_files if file.endswith('.txt')]
    
    print(color[1])
    for file in txt_files:    
        file_path=os.path.join(info_path,file)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = np.loadtxt(file_path, dtype=int)
        image_path=frames_path+base_name+'.jpg'
        image=cv2.imread(image_path)
        #image=draw_black_image()
        for k in data:
            print("k",k)
            print("data[k][2]",k[3])
            colorid=color[k[3]]
            if k[2]==0:
                cv2.circle(image, (k[0],k[1]), 15, colorid, -1)
            elif k[2]==1:
                cv2.circle(image, (k[0],k[1]), 15, colorid, 3)

        output_path= os.path.join(save_path_point, f"{base_name}.jpg")
        cv2.imwrite(output_path, image)
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

def LK_cuda(selected_point,start_frame,input_fordel_path,end_frame=1500):
    
    #source = '/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/frames/frame_0.jpg'
    #save_path='/home/src/data/captest/capture/DJI_20230719145427_0002_V_video5/lkout/'
    source =os.path.join(input_fordel_path,'frames/frame_'+str(start_frame)+'.jpg')
    save_path=os.path.join(input_fordel_path,'lkout/')
    remove_all_items_in_directory(save_path)
    os.makedirs(save_path, exist_ok=True)
    cvFrame=cv2.imread(source)
    #print("selected_point shape",selected_point.shape)
    images_list = [img for img in os.listdir(os.path.join(input_fordel_path,'frames/')) if img.endswith(".jpg")]

    with vpi.Backend.CPU:
        frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures = vpi.asarray(selected_point)
    # with curFeatures.lock_cpu():
    #     curFeatures.size = min(curFeatures.size, 25)
    with vpi.Backend.CUDA:
        optflow = vpi.OpticalFlowPyrLK(frame, curFeatures,5)
    idFrame = start_frame
    while True:
        print(idFrame)
        prevFeatures = curFeatures
        #if idFrame >= len(images_list)-1:
        if idFrame >= end_frame- start_frame:
            print("Video ended.")
            break
        idFrame += 1
        # if idFrame == 20:
        #     idFrame += 1
        # else:
        #     idFrame += 1
        path=input_fordel_path+"frames/frame_"+str(idFrame)+".jpg"
        cvFrame=cv2.imread(path)
        with vpi.Backend.CUDA:
            frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
        curFeatures, status = optflow(frame)

        #print(curFeatures.cpu())
        #print(status.cpu())
        file_name="frame_"+str(idFrame)+".txt"
        print("curFeatures.cpu()",curFeatures.cpu().shape)
        print("status.cpu()",status.cpu())
        status_reshaped = status.cpu().reshape(-1, 1)

        # Concatenate along the second axis (columns)
        result = np.hstack((curFeatures.cpu(), status_reshaped))
        save_txt_point(result,save_path,file_name)

    return curFeatures.cpu(),status.cpu()#,prevFeatures.cpu() # wrong output don't use

def pad_list_2d(lst, n):
    # Create a new list by copying the original list
    padded_list = lst.copy()
    
    # Check if the current length is less than the desired length
    while len(padded_list) < n:
        # Append [0, 0] to the list
        padded_list.append([0, 0])
    
    return padded_list

def ocl_point_data_gen(point_per_ani, number_point_ocl, input_data):
    output_list=input_data.copy()
    random_values = random.sample(range(0, point_per_ani-1), number_point_ocl)
    for i in random_values:
        output_list[i]=[0,0]
    #print("output_list",output_list)
    return output_list
def convert_numpy_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    elif isinstance(data, (np.int64, np.float64)):
        return data.item()  # Convert NumPy scalar to native Python type
    else:
        return data

def build_data(input_fordel_path,point_per_ani,number_point_ocl,data):

    # Todo: open 2 consecutive files
    info_path=os.path.join(input_fordel_path,'lkout/')
    save_path=os.path.join(input_fordel_path,'cnn1_data/')
    frames_path=os.path.join(input_fordel_path,'frames/')
    os.makedirs(save_path, exist_ok=True)

    all_files = os.listdir(info_path)
    txt_files = [file for file in all_files if file.endswith('.txt')]
    txt_files=natsort.natsorted(txt_files)
    #print(txt_files)

    for i in range(1,len(txt_files) - 1):
        file1 = txt_files[i]
        file2 = txt_files[i + 1]
        print("file1:",file1,"___ file2:",file2)

        file_path1=os.path.join(info_path,file1)
        file_path2=os.path.join(info_path,file2)
        data1 = np.loadtxt(file_path1, dtype=int)
        data2 = np.loadtxt(file_path2, dtype=int)
        #print("$$$$$$$$",type(data1))
        for unique_id in np.unique(data1[:, 3]):
            id_rows1 = data1[data1[:, 3] == unique_id]
            id_rows2 = data2[data2[:, 3] == unique_id]
            label_data=[]
            inputa=[]
            for k in range(id_rows2.shape[0]):
                if id_rows2[k][2]==0 and id_rows1[k][2]==0:
                    output= id_rows2[k,:2]
                    x=id_rows2[k][0]
                    y=id_rows2[k][1]
                    po=[x,y] #po=[x/3840,y/2160]
                    label_data.append(po)
                    
                    inputa.append([id_rows1[k][0],id_rows1[k][1]]) #inputa.append([id_rows1[k][0]/3840,id_rows1[k][1]/2160])
                    #print("inputa", inputa)
            
            label_data=pad_list_2d(label_data,point_per_ani)
            #print("labellabel&&&& Trueeee",label_data)
            inputa=pad_list_2d(inputa,point_per_ani)
            #print("inputainputa &&&&&&",inputa)
            dmm=ocl_point_data_gen(point_per_ani=point_per_ani, number_point_ocl=number_point_ocl, input_data = label_data)
            inputa=inputa+dmm

            print("inputainputa",inputa)
            print("labelllll",label_data)


            # data_label.append(label_data)
            # input_data.append(inputa)

            data['input'].append(inputa)
            data['label'].append(label_data)
    # data['label']=data_label
    # data['input']=input_data

    print("data",len(data['label'])) 
    
    #output_path= os.path.join(save_path, "data.json")
    output_path='/home/src/data/captest/data.json'
    #print("bounding_boxes",bounding_boxes)
    converted_data = convert_numpy_to_native(data)
    with open(output_path, 'w') as json_file:
        json.dump(converted_data, json_file, indent=4)
    # print("data_label len",len(data_label))
    #ocl_point_data_gen(point_per_ani=point_per_ani, number_point_ocl=number_point_ocl, input_list)
            


                # if id_rows2[k][2]==0:

                #     print("id_rows2[:,:2]")
                #     output= np.delete(id_rows2[:,:2], [0, 2], 0)


            #print("id_rows",id_rows)
       

    #print(color[1])

    # for file in txt_files:    
    #     file_path=os.path.join(info_path,file)
    #     base_name = os.path.splitext(os.path.basename(file_path))[0]
    #     data = np.loadtxt(file_path, dtype=int)
    #     for k in data:
    #         print("k",k)
    #         print("data[k][3]",k[3])
    #         colorid=color[k[3]]
    #         if k[2]==0:
    #             cv2.circle(image, (k[0],k[1]), 15, colorid, -1)
    #         elif k[2]==1:
    #             cv2.circle(image, (k[0],k[1]), 15, colorid, -1)

    #     output_path= os.path.join(save_path_point, f"{base_name}.jpg")
    #     cv2.imwrite(output_path, image)
    # # return  label (n points t+1)
    #input t+1 append  t 
    # create random  0,1 array (0: fail , 1 good) # same cotrack but different lk
    # modify input
    # save input

    return None



def main_predict_miss_point(args):
    #[DJI_20230720075532_0007_V_video2,DJI_0133_video1,DJI_20230719145816_0003_V_video2,DJI_0119]
    start_frame=0
    point_pre_ani=5
    number_point_ocl=1
    data = {'input':[],'label':[]}
    data_path='/home/src/data/captest/data.json'
    # with open(data_path, 'w') as json_file:
    #     json.dump(data, json_file, indent=4)
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    model = YOLO('yolov8x-seg.pt')

    input_fordel_path='/home/src/data/captest/capture/DJI_0119/'
    selected_point,points_id= detect_first_frame(model,input_fordel_path=input_fordel_path, start_frame=start_frame,point_pre_ani=point_pre_ani) 
    feature, status=LK_cuda(selected_point,start_frame=start_frame,input_fordel_path=input_fordel_path)
    append_column(points_id,input_fordel_path=input_fordel_path)
    build_data(input_fordel_path=input_fordel_path,point_per_ani=point_pre_ani,number_point_ocl=number_point_ocl,data=data)

def main(args):
    # list_video=[DJI_0119 ,'DJI_20230719145816_0003_V_video2','DJI_20230719145427_0002_V_video5','briszoo','DJI_0133_video1']
    # #start_time = time.time()

    model = YOLO('yolov8x-seg.pt')
    input_fordel_path='/home/src/data/captest/capture/DJI_20230719145816_0003_V_video2/'
    # selected_point,points_id= detect_first_frame(model,input_fordel_path=input_fordel_path, start_frame=0,point_pre_ani=5)
    # # print(selected_point.dtype)


    # feature, status=LK_cuda(selected_point,start_frame=0,input_fordel_path=input_fordel_path,end_frame=500)
    # append_column(points_id,input_fordel_path=input_fordel_path)
    
    # # #append_column(status,input_fordel_path=input_fordel_path)
    # color = generate_random_colors(20)
    # # # # # # # #reconstruct_box(color,input_fordel_path=input_fordel_path)

    # # # visual_point(color,input_fordel_path=input_fordel_path)
    # visual_point_onframe(color,input_fordel_path=input_fordel_path)
    create_video_from_images(input_fordel_path,name='visual_tracklet')
    # # create_video_from_images(input_fordel_path,name='frames')
    # # visual_points_on_black_foreground(saved_folder)
    # # label_step(saved_folder, labeled folder)
    print("Done")

if __name__=="__main__":
   args=make_parse().parse_args()

   print(args.pyramid_levels)
   main(args)
   #main_predict_miss_point(args)