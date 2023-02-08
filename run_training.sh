#!/bin/bash

### Change this
echo "Will not run until exported variable is set in this file!"
export WEED_DATA_PATH=<YOUR_WEED_DATA_PATH>



current_dir="$(pwd)"
docker run -it --rm --runtime=nvidia -v $current_dir/train:/train \
-v $WEED_DATA_PATH:/weed_data -v $current_dir/code/:/code \
-e CVAT_USERNAME= -e CVAT_PASSWORD= \
-e CVAT_BASE_URL=http://192.168.68.137:8080/api/v1/ -e MONGODB_PORT=27017 \
-e MONGODB_USERNAME= -e MONGODB_PASSWORD= \
-v $current_dir/vision:/code/vision -v $current_dir/file_lists:/file_lists --shm-size=5g --net=host training:v2 /bin/bash