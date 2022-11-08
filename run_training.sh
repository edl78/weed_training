#!/bin/bash
current_dir="$(pwd)"
docker run -it --rm --runtime=nvidia -v $current_dir/train:/train \
-v /home/daniel/.ssh/known_hosts:/home/dopamine/.ssh/known_hosts \
-v /home/daniel/code/cvat_stuff/weed_data_obdb:/weed_data -v $current_dir/code/:/code \
-v $current_dir/vision:/code/vision --shm-size=5g --net=host training:v1 /bin/bash