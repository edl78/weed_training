#!/bin/bash
docker build -t training:v2 -f Dockerfile .
git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.13.1
git clone https://github.com/rafaelpadilla/Object-Detection-Metrics.git
cp Object-Detection-Metrics/lib/*.py code/