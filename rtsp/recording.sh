#!/usr/bin/env bash

mkdir -p recordings
ffmpeg -rtsp_transport tcp -i "rtsp://rtspstream:jG2zm1ADvRayJ7XLoiBT5@zephyr.rtsp.stream/people" -c copy -f segment -segment_time 300 -reset_timestamps 1 -strftime 1 "recordings/%Y%m%d_%H%M%S.mp4"
