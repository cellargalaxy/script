#!/usr/bin/env bash

snapshot_dir="output/snapshot"
record_dir="output/record"

mkdir -p $snapshot_dir
mkdir -p $record_dir

echo $rtsp_url


./ffmpeg -rtsp_transport tcp -i "$rtsp_url" \
  -vf "fps=1" -vsync vfr -c:v libwebp -q:v 80 -an -f image2 -strftime 1 "output/snapshot/%Y%m%d_%H%M%S.webp" \
  -c:v copy -f segment -segment_time 60 -strftime 1 "output/record/%Y%m%d_%H%M%S.mp4"


#./ffmpeg -rtsp_transport tcp -i "$rtsp_url" \
#  -vf "scale=1280:720,fps=1" -vsync vfr -c:v libwebp -q:v 80 -an -f image2 -strftime 1 "output/snapshot/%Y%m%d_%H%M%S.webp" \
#  -c:v libx265 -preset ultrafast -crf 40 -c:a aac -b:a 32k -vf "scale=1280:720,fps=10" -f segment -segment_time 300 -strftime 1 "output/record/%Y%m%d_%H%M%S.mp4"







