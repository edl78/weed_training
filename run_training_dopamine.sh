#!/bin/bash
current_dir="$(pwd)"
docker run --rm --gpus device=0 -v $current_dir/train:/train -v /fs/sefs1/obdb:/weed_data -v $current_dir/code/:/code -v $current_dir/vision:/code/vision --shm-size=8g --net=host training:v1 python3 /code/torch_model_runner.py
