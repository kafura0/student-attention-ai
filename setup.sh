#!/bin/bash
pip install --upgrade pip
pip uninstall -y opencv-python opencv-contrib-python || true
pip install opencv-python-headless==4.10.0.84
pip install mediapipe==0.10.9
