#!/bin/bash

echo "This script sets up the environment for the Azure virtual machines."

echo "1. Installing requirements..."

apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

echo "2. Cloning the repo..."

git clone https://github.com/rogierz/aml22-rl


echo "3. Installing Python requirements..."
pip install mujoco-py
pip install -r aml22-rl/requirements.txt


echo "4. Downloading MuJoCo Python bindings..."
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar xf mujoco210-linux-x86_64.tar.gz


echo "Done!"
if [[ ! $(echo $LD_LIBRARY_PATH) ]]; then
echo "Don't forget to set up the environment variable LD_LIBRARY_PATH !"
echo "Example: > env MUJOCO_PY_MUJOCO_PATH=/content/mujoco210"
fi

if [[ ! $(echo $MUJOCO_PY_MUJOCO_PATH) ]]; then
echo "Don't forget to set up the environment variable MUJOCO_PY_MUJOCO_PATH"
echo "Example: > env LD_LIBRARY_PATH=/usr/lib64-nvidia:/content/mujoco210/bin:/usr/lib/nvidia"
fi
