#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

COPY ./sources.list.se /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-pip 
RUN pip3 install --upgrade pip
RUN apt-get install -y python3
RUN pip3 install tensorflow==2.3.0
RUN pip3 install pandas numpy scipy scikit-learn flask gpyopt
RUN pip3 install matplotlib setuptools
RUN pip3 install parameter-sherpa
RUN apt-get update && apt-get install -y xauth
#RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pymongo
RUN pip3 install flask
RUN pip3 install numpyencoder
RUN pip3 install --upgrade pip && pip3 install mapcalc
RUN pip3 install scp
RUN pip3 install shapely
RUN pip3 install xmltodict
RUN pip3 install wget

#WORKDIR '/code'
#ENTRYPOINT ["python3", "torch_model_runner.py"]
