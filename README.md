# DC12
*A project that does something amazing.*
# commit from laptop
# commit fron PC
# commit from jetson

## Table of Contents
1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage](#usage)

### Download WildTracker
```bash
git clone --recurse-submodules https://github.com/dat-nguyenvn/DC12.git
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

