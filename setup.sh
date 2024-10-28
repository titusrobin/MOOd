#!/bin/bash

# Update and install required packages
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev wget

# Set up a virtual environment
python3.8 -m venv env
source env/bin/activate

pip install tools
pip install torch
git add 
# Install MMCV without CUDA
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.1
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install MMSegmentation
pip install mmsegmentation==0.20.2

# Install MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install PETR
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts
mkdir data
ln -s ../mmdetection3d ./mmdetection3d
cd ..

# Download and extract the nuScenes mini dataset
mkdir -p data/nuscenes
cd data/nuscenes
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xvf v1.0-mini.tgz
cd ../..

# Run data preparation script for nuScenes mini dataset
python3 mmdetection3d/tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version v1.0-mini

echo "Setup complete. Activate your virtual environment with 'source env/bin/activate'"
