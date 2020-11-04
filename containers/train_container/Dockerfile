FROM ubuntu:focal

RUN apt-get update
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx python3.7 python3-pip
RUN apt-get update
RUN apt-get -y install python3-opencv

COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt
ADD . .
