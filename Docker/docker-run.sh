#!/bin/bash
nvidia-docker run -it --rm \
    --name object-detection-filter-$USER \
    --hostname object-detection-filter-$USER \
    -v /mnt/poplin:/mnt/poplin -v /mnt/Data:/mnt/Data -v ~/:/home/$USER \
    -v /mnt/workspace2016/:/mnt/workspace2016 \
    object-detection-filter:latest \
    bash