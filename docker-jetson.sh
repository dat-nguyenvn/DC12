#!/usr/bin/env bash
#sudo systemctl restart jtop.service
sudo systemctl restart docker
#sudo docker rm DC12_7Feb
xhost +
#sudo systemctl restart jtop.service
cd jetson-inference
# -v /run/jtop.sock:/run/jtop.sock \
docker/run.sh \
 -v /home/leader/myjetson/2025/github/DC12:/DC12/ \
 -v /home/leader/myjetson/2025/github/mount:/mount/ \
 --container 4d7a58b2dfb0

#docker/run.sh -v /home/leader/myjetson/2023/jetson-infer/mount-folder:/mount-folder 4d7a58b2dfb0
