![Alt text](images/title.png)
## About the Project
We have developed an ultra-fast, multi-animal recognition and tracking method specifically designed for drones, with applications in wildlife monitoring and conservation. At the core of our hardware system is the Jetson Orin AGX that integrated into the drone to enable the deployment of advanced deep learning techniques. The software provides a dynamic, autonomous tracking system, offering conservation scientists an innovative alternative to traditional camera trap technology. The research is part of [WildDrone project](https://wilddrone.eu/)

## Table of Contents
1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Getting Started](#getting-started)
   - [Installation](#installation)




### Download WildTracker
```bash
git clone --recurse-submodules git@github.com:dat-nguyenvn/DC12.git
```

### Setup environment
#### Docker
```bash
docker pull nvcr.io/nvidia/pytorch:23.12-py3

docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/boss/mypc/phd/DC12:/DC12\
 -v /home/boss/mypc/mount:/mount\
 --name envgit 1cff6923bda0
```
* Replace `/home/boss/mypc/phd/DC12` to your **DC12 directory**


#### Depend libraries
```bash
cd /DC12
./setup.sh
```


```bash
cd ultralytics
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
```

```bash
cd sahi
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
```

### Run WildTracker
```bash 
python3 updatekenya.py
```

