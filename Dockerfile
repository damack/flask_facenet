FROM ubuntu:latest

RUN \
    apt update && \
    apt upgrade -y
    
RUN apt install -y python3 python3-pip python3-setuptools libboost-all-dev libgtk2.0-dev
RUN pip3 install --upgrade pip
RUN pip3 install numpy Pillow flask typing opencv-python scipy sklearn tensorflow

WORKDIR /usr/src/app
RUN mkdir models && apt install unzip

WORKDIR /usr/src/app/models
RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-" > /dev/null && curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-" -o 20180402-114759.zip && rm -rf ./cookie
RUN unzip 20180402-114759.zip && rm 20180402-114759.zip

WORKDIR /usr/src/app/models/20180402-114759
RUN mv * ../

WORKDIR /usr/src/app/models
RUN rm -rf 20180402-114759


WORKDIR /usr/src/app
COPY . .

CMD [ "python3", "./app.py" ]