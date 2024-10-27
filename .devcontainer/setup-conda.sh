#!/bin/bash

# Install Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
/bin/bash /tmp/miniconda.sh -b -p $HOME/miniconda
rm /tmp/miniconda.sh

# Initialize Conda for bash
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate
conda init bash

# Create and activate the Conda environment
conda create -y -n myenv python=3.8

# Install required packages
conda install -y -n myenv pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=11.1 -c pytorch
conda install -y -n myenv opencv=4.5.3 -c conda-forge
conda install -y -n myenv mmcv=1.3.14 -c open-mmlab
conda install -y -n myenv mmdet=2.16.0 mmsegmentation=0.17.0 mmdetection3d=0.17.0 -c open-mmlab

# Activate the environment by default in bash sessions
echo "conda activate myenv" >> ~/.bashrc
