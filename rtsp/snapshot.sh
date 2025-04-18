#!/usr/bin/env bash

mkdir -p snapshots
ffmpeg -rtsp_transport tcp -i "rtsp://rtspstream:jG2zm1ADvRayJ7XLoiBT5@zephyr.rtsp.stream/people" -r 1 -q:v 2 -f image2 -strftime 1 "snapshots/%Y%m%d_%H%M%S.jpg"
