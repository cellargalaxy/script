#!/usr/bin/env bash
docker container stop peerbanhelper
docker container rm peerbanhelper
docker image rm ghostchu/peerbanhelper
docker run -d \
--name peerbanhelper \
--restart unless-stopped \
-p 9898:9898 \
-v ~/software/peerbanhelper:/app/data/ \
ghostchu/peerbanhelper:latest

xdg-open http://127.0.0.1:9898/
xdg-open http://127.0.0.1:8080/

~/software/qbittorrent-nox