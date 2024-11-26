#!/usr/bin/env bash
cd ./ultralytics
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH

cd ..
pip install -e .

apt-get update -y
apt-get install -y libx11-dev
apt-get install -y python3-tk
pip install imageio[ffmpeg]
pip install natsort
pip install shapely
pip install matplotlib flow_vis tqdm tensorboard

cd sahi
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

#install vpi
apt install gnupg
apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
apt install software-properties-common
add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main'
apt update
apt install libnvvpi3 vpi3-dev vpi3-samples
apt install python3.10-vpi3
