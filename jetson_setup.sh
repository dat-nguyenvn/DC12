#!/usr/bin/env bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y libx11-dev
apt-get install -y python3-tk
pip install imageio[ffmpeg]


pip install shapely natsort
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
pip install hydra-core==1.1.0 mediapy

cd ultralytics
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

cd sahi
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

pip install -e .
