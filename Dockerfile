From nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-pip 
RUN pip3 install --upgrade pip
RUN apt-get install -y python3
RUN pip3 install tensorboard
RUN pip3 install pandas numpy scipy scikit-learn flask gpyopt
RUN pip3 install matplotlib setuptools
RUN pip3 install parameter-sherpa
RUN apt-get update && apt-get install -y xauth
RUn pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install pymongo
RUN pip3 install flask
RUN pip3 install numpyencoder
RUN pip3 install --upgrade pip && pip3 install mapcalc
RUN pip3 install scp
RUN pip3 install shapely
RUN pip3 install xmltodict
RUN pip3 install wget
RUN pip3 install torchmetrics
RUN pip3 install opencv-python
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0
