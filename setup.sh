#!/bin/bash

echo "This script sets up the environment for the Azure virtual machines."

echo "1. Installing requirements..."

sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

echo "2. Creating virtualenv..."
python -m venv .venv
echo "export MUJOCO_PY_MUJOCO_PATH=/home/azureuser/localfiles/mujoco210" >> .venv/bin/activate
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/azureuser/localfiles/mujoco210/bin" >> .venv/bin/activate
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> .venv/bin/activate

source .venv/bin/activate

echo "2. Installing Python requirements..."
pip install mujoco-py
pip install -r aml22-rl/requirements.txt


echo "3. Downloading MuJoCo Python bindings..."
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar xf mujoco210-linux-x86_64.tar.gz

echo "Done!"
