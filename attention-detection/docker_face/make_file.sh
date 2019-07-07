#!/bin/bash
cd caffe-fast-rcnn
rm -rf build
mkdir build
cd build
cmake -DUSE_CUDNN=1 ..
make -j20 && make pycaffe
cd ../..
