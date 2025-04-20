#!/usr/bin/env bash

snapshot_dir="output/snapshot"
record_dir="output/record"
mkdir -p $snapshot_dir
mkdir -p $record_dir

echo 'rtsp_url:'$rtsp_url
if [ -z $rtsp_url ]; then
  sleep 1
  exit 1
fi

ffmpeg -rtsp_transport tcp -i "$rtsp_url" \
  -vf "fps=1" -vsync vfr -c:v libwebp -q:v 80 -an -f image2 -strftime 1 "output/snapshot/%Y%m%d_%H%M%S.webp" \
  -c:v copy -an -f segment -segment_time 300 -strftime 1 "output/record/%Y%m%d_%H%M%S.mp4"

