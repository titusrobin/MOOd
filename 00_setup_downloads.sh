#!/bin/bash
set -e

# Install required system libraries
# sudo apt-get update
# sudo apt-get install -y libgl1-mesa-glx

# Define versions for installations
MMCV_VERSION="1.3.14"
MMDET_VERSION="2.16.0"
MMD3D_VERSION="0.17.1"
NUSCENES_DEVKIT_VERSION="1.1.0"  # Version for the nuScenes devkit

# Define paths
NUSCENES_PATH="./nuscenes"  # Path to download the nuScenes dataset
PETR_PATH="./PETR"          # Path for the PETR model
# Download and extract nuScenes mini dataset

# Create directory for nuScenes dataset
mkdir -p $NUSCENES_PATH

# # Download and extract nuScenes mini dataset
wget https://www.nuscenes.org/data/v1.0-mini.tgz -O nuscenes_mini.tgz
tar -xf nuscenes_mini.tgz -C $NUSCENES_PATH

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git

# Install MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git

# Clone detr3d
git clone https://github.com/WangYueFt/detr3d.git

echo "Setup complete! The nuScenes mini dataset has been downloaded and extracted."
