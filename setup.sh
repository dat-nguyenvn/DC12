#!/usr/bin/env bash
export DEBIAN_FRONTEND=noninteractive

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

cd ./sahi
pip install -e .
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

#install vpi
apt-get install -y gnupg
apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
apt install -y software-properties-common
add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main'
apt update -y
apt-get install -y libnvvpi3 vpi3-dev vpi3-samples
apt-get install -y python3.10-vpi3
