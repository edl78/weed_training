FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

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
RUN pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pymongo
RUN pip3 install flask
RUN pip3 install numpyencoder
RUN pip3 install --upgrade pip && pip3 install mapcalc
RUN pip3 install scp
RUN pip3 install shapely
RUN pip3 install xmltodict

#WORKDIR '/code'
#ENTRYPOINT ["python3", "torch_model_runner.py"]
