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

RUN apt-get install -y locales

RUN sed -i -e \
  's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen \
   && locale-gen

ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU:ru
ENV LC_LANG ru_RU.UTF-8
ENV LC_ALL ru_RU.UTF-8

ADD . .
