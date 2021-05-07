FROM python:3.7
MAINTAINER tianyi

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN echo 'deb http://mirrors.ustc.edu.cn/debian stable main contrib non-free' >>/etc/apt/sources.list
RUN echo 'deb http://mirrors.ustc.edu.cn/debian stable-updates main contrib non-free' >>/etc/apt/sources.list
RUN apt-get update && apt-get install nginx libgl1-mesa-glx -y
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ wheel psycopg2-binary

RUN mkdir /navi
WORKDIR /navi

COPY . /navi
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r Requirements.txt

RUN chmod 777 /navi/rundocker-pre.sh

ENTRYPOINT ["/navi/rundocker-pre.sh"]
