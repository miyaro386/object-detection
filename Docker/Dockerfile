FROM  nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Miyauchi

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    vim \
    libjpeg-dev libpng-dev \
    python-tk \
    python-imaging \
    python-scipy \
    python-nose \
    python-matplotlib \
    python-shapely \
    python-opencv \
    protobuf-compiler \
    python-pil \
    python-lxml \
    wget \
    gcc g++ \
    git-core 
    
RUN apt-get update
RUN apt-get install -y \
    file \
    dos2unix \
    unzip \
    graphviz

# install anaconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

#ENV HOME=/home/$USER \
#    SHELL=/bin/bash

#RUN conda update numpy
RUN conda update pip setuptools -y
RUN pip install -U pip
RUN pip install tensorflow-gpu
RUN apt-get install graphviz -y
RUN pip install pydot pydot-ng graphviz

RUN pip install h5py imageio pillow tqdm \
	pydot pydot-ng graphviz opencv-python imageio seaborn

RUN conda install -c menpo opencv3=3.1.0 -y

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /home/$USER


CMD bash
