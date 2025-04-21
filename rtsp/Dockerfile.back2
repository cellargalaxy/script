FROM python:3.9-alpine

WORKDIR /
COPY clean.sh .
COPY snapshot_and_recording.sh .
COPY snapshot_and_recording.py .

RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk update
RUN apk --no-cache add ca-certificates
RUN apk add tzdata && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && apk del tzdata
RUN apk add --no-cache ffmpeg

VOLUME /output
CMD ["python", "/snapshot_and_recording.py"]
