#!/bin/bash

# Install required system libraries
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Define versions for installations
MMCV_VERSION="1.4.0"
MMDET_VERSION="2.24.1"
MMD3D_VERSION="0.17.1"
NUSCENES_DEVKIT_VERSION="1.1.0"  # Version for the nuScenes devkit

# Define paths
NUSCENES_PATH="./nuscenes"  # Path to download the nuScenes dataset
PETR_PATH="./PETR"          # Path for the PETR model
# Download and extract nuScenes mini dataset

# Create directory for nuScenes dataset
mkdir -p $NUSCENES_PATH

# Download and extract nuScenes mini dataset
wget https://www.nuscenes.org/data/v1.0-mini.tgz -O nuscenes_mini.tgz
tar -xf nuscenes_mini.tgz -C $NUSCENES_PATH

# Install nuScenes devkit
pip install nuscenes-devkit==$NUSCENES_DEVKIT_VERSION

# Install MMCV (CPU version)
pip install mmcv-full==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v$MMDET_VERSION
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v$MMD3D_VERSION
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install PETR
# git clone https://github.com/megvii-research/PETR.git
# cd PETR
!git clone https://github.com/WangYueFt/detr3d.git
%cd detr3d
!pip install -v -e 
cd ..

echo "Setup complete! The nuScenes mini dataset has been downloaded and extracted."
