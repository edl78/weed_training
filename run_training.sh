#!/bin/bash
current_dir="$(pwd)"
docker run -it --rm --runtime=nvidia -v $current_dir/train:/train \
-v /media/datalager/work/weed_data:/weed_data -v $current_dir/code/:/code \
-e CVAT_USERNAME= -e CVAT_PASSWORD= \
-e CVAT_BASE_URL=http://192.168.68.137:8080/api/v1/ -e MONGODB_PORT=27017 \
-e MONGODB_USERNAME= -e MONGODB_PASSWORD= \
-v $current_dir/vision:/code/vision --shm-size=5g --net=host training:v1 /bin/bash