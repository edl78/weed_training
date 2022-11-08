#!/bin/bash
docker build -t training:v1 -f Dockerfile .
git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.10.0
