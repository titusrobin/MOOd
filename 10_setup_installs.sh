#!/bin/bash
set -e

# Install required system libraries
# sudo apt-get update
# sudo apt-get install -y libgl1-mesa-glx

conda activate base
conda env remove detr3d

conda create -n detr3d
conda activate detr3d
conda install pytorch python=3.10 jupyter cuda cython

# Define versions for installations
MMCV_VERSION="1.3.14"
MMDET_VERSION="2.16.0"
MMD3D_VERSION="0.17.1"
NUSCENES_DEVKIT_VERSION="1.1.0"  # Version for the nuScenes devkit

# Define paths
NUSCENES_PATH="/hpc/group/ssri/nuscenes"  # Path to download the nuScenes dataset
# Download and extract nuScenes mini dataset

# Install nuScenes devkit
pip install nuscenes-devkit==$NUSCENES_DEVKIT_VERSION

# Install MMCV (CPU version)
pip install mmcv-full==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
pip install mmsegmentation==0.17.0

# Install MMDetection
cd mmdetection
git checkout v$MMDET_VERSION
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install MMDetection3D
cd mmdetection3d
git checkout v$MMD3D_VERSION
pip install -r requirements/build.txt
python setup.py develop
cd ..

# Install detr3d
cd detr3d
pip install -v -e 
cd ..

echo "Setup complete! The nuScenes mini dataset has been downloaded and extracted."
