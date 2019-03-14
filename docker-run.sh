#!/bin/bash
nvidia-docker run -it --rm \
    --name object-detection-filter-$USER \
    --hostname object-detection-filter-$USER \
    miyaro386/object-detection-filter:latest \
    bash