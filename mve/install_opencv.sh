#!/bin/bash

INSTALL_BASE_DIR="$PWD/.."
INSTALL_DIR="$PWD"

echo "Installing OpenCV into: $INSTALL_DIR"

# Install opencv dependencies
echo "Installing OpenCV dependencies"
apt-get update -qq --fix-missing && \
apt-get upgrade -y && \
apt-get install -y \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libx265-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    openexr \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*


# Download OpenCV and build from source
OPENCV_VERSION="4.5.5"
echo "Downloading OpenCV source"
cd "$INSTALL_BASE_DIR"
wget -O "$INSTALL_BASE_DIR"/opencv.zip https://github.com/opencv/opencv/archive/"$OPENCV_VERSION".zip
unzip "$INSTALL_BASE_DIR"/opencv.zip
mv "$INSTALL_BASE_DIR"/opencv-"$OPENCV_VERSION"/ "$INSTALL_BASE_DIR"/opencv/
rm -rf "$INSTALL_BASE_DIR"/opencv.zip

echo "Configuring OpenCV"
cd "$INSTALL_BASE_DIR"/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_GENERATE_PKGCONFIG=YES \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_ENABLE_NONFREE=OFF \
      -D BUILD_LIST=core,imgproc ..
echo "Compiling OpenCV"
make -j $(nproc)
make install
ldconfig