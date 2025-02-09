FROM elestio/docker-desktop-vnc

# download from https://www.interactivebrokers.com.hk/cn/index.php?f=15948
COPY ./tws-latest-standalone-linux-x64.sh /tmp/tws-latest-standalone-linux-x64.sh
RUN /bin/sh /tmp/tws-latest-standalone-linux-x64.sh -q

ENTRYPOINT ["/usr/local/tws/tws"]