#!/usr/bin/env bash
xhost +local:docker

sudo docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/ah23975/mypc/2023/tracker/co-tracker:/home/src/tracker \
 -v /home/ah23975/mypc/2023/yolo:/home/src/yolo \
 -v /home/ah23975/mypc/world_data:/home/data\
 --name kenya  1cff6923bda0
 
 
 sudo docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/ah23975/mypc/2023/tracker/co-tracker:/home/src/tracker \
 -v /home/ah23975/mypc/2023/dino:/home/src/dino \
 -v /home/ah23975/mypc/2023/yolo:/home/src/yolo \
 -v /media/ah23975/Crucial\ X9:/home/src/data \
 --name kenyanew3  1cff6923bda0
 
apt install gnupg
apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
apt install software-properties-common
add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main'
apt update
apt install libnvvpi3 vpi3-dev vpi3-samples
apt install python3.10-vpi3

git clone https://github.com/obss/sahi.git


 sudo docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/ah23975/mypc/2025/github/DC12:/DC12\
 -v /home/ah23975/mypc/2025/github/mount:/mount\
 --name github 1cff6923bda0
 
  sudo docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/ah23975/mypc/2025/YOLOv7-DeepSORT:/YOLOv7-DeepSORT\
 -v /home/ah23975/mypc/mount:/mount\
 --name yolo7 1cff6923bda0
 
 
