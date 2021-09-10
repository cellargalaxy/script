FROM ubuntu:18.04
ENV TZ=Asia/Shanghai
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
          && apt-get update \
          && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
          && apt-get install mysql-client -y \
          && apt-get clean \
          && apt-get autoclean \
          && rm -rf /var/lib/apt/lists/
WORKDIR /
COPY . .
VOLUME /local
CMD ["bash" , "/run.sh"]