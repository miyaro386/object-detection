# Object detection filter for LSUN

This project is for [Shape-conditioned Image Generation by Learning Latent Appearance Representation from Unpaired Data](https://arxiv.org/abs/1811.11991).  
Training data are collected from LSUN by filtering of this project.  

## Usage

1. Download LSUN dataset from [here](http://tigress-web.princeton.edu/~fy/lsun/public/release/)  
1. Export images following  https://github.com/fyu/lsun  

    You will get files like a below 
    ```
    root_dir
    |-car
    | |- 1.webp
    | |- 2.webp
    |   ・
    |   ・
    |-chair
    | |- 1.webp
    | |- 2.webp
    |   ・
    |   ・
    ```

1. Install dependencies following [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection")
    or run `docker-run.sh`
    
    I uploaded my docker image to DockerHub and recommend to use Docker.
    
1. Edit and run `run.sh`


    
    

