#!/usr/bin/env bash
apt-get update -y
apt-get install -y libx11-dev
apt-get install -y python3-tk
pip install imageio[ffmpeg]
pip install -e .
pip install shapely
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
cd /home/src/yolo/ultralytics
pip install -e .