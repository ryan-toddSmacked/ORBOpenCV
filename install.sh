#!/bin/bash

# usage: ./install.sh <install path>

# This script downloads opencv and opencv_contrib from github and builds it locally
# It attempts to build with CUDA support, but if that fails, it will build without it

# Check if the user has provided an install path
if [ -z "$1" ]
then
    echo "No install path provided"
    echo "Installing to $HOME/.local/opencv-4.8.0"
    PREFIX="$HOME/.local/opencv-4.8.0"
else
    PREFIX="$1"
fi

# Use wget to download opencv-4.8.0
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.8.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip

# Check if the opencv directory already exists, and remove it if it does
if [ -d "opencv-4.8.0" ]
then
    rm -rf opencv-4.8.0
fi
if [ -d "opencv_contrib-4.8.0" ]
then
    rm -rf opencv_contrib-4.8.0
fi

OPENCV_CONTRIB_DIR=$(pwd)/opencv_contrib-4.8.0

# unzip the files
unzip opencv.zip
unzip opencv_contrib.zip

# Move into the opencv directory
cd opencv-4.8.0

# Create a build directory
mkdir build

# Move into the build directory
cd build

# Run cmake to configure the build
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$PREFIX \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=5.0 \
-D OPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB_DIR/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_LIST=core,calib3d,cudafeatures2d,highgui,imgcodecs,imgproc,cudev \
..


# Run make to build opencv
N=$(nproc)
make -j$N

# Run make install to install opencv
make install

# cd back to the original directory
cd ../..

# Remove the zip files
rm opencv.zip
rm opencv_contrib.zip

# Remove the opencv and opencv_contrib directories
rm -rf opencv-4.8.0
rm -rf opencv_contrib-4.8.0