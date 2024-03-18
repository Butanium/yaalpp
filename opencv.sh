#!/bin/sh

mkdir -p opencv
cd opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mkdir -p build
cd build
cmake ../opencv-4.x
cmake --build .
