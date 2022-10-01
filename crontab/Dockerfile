FROM alpine
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk update
RUN apk --no-cache add ca-certificates
RUN apk add tzdata && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && apk del tzdata

WORKDIR /crontab
COPY . .
RUN chmod 755 crontab.sh job.sh
RUN /usr/bin/crontab /crontab/crontab.txt
CMD ["/crontab/crontab.sh"]