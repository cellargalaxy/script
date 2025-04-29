FROM python:3.9-alpine

RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk update
RUN apk --no-cache add ca-certificates
RUN apk add tzdata && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && apk del tzdata
RUN apk add ffmpeg socat procps

WORKDIR /
COPY main.py .
COPY socat.sh .
COPY survive_monitor.sh .
COPY clean.sh .
COPY snapshot_and_recording.sh .
RUN chmod +x /socat.sh /survive_monitor.sh /clean.sh /snapshot_and_recording.sh

HEALTHCHECK CMD pgrep ffmpeg || exit 1
CMD ["python", "main.py"]
