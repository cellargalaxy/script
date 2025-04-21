#!/usr/bin/env sh

log_info() {
  echo "INFO $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
  echo "ERRO $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

if [ -z $snapshot_dir ]; then
  snapshot_dir="output/snapshot"
fi
if [ -z $record_dir ]; then
  record_dir="output/record"
fi

mkdir -p $snapshot_dir
mkdir -p $record_dir

echo 'rtsp_url:'$rtsp_url
if [ -z $rtsp_url ]; then
  log_error 'rtsp_url is blank'
  sleep 1
  exit 1
fi

ffmpeg -rtsp_transport tcp -i "$rtsp_url" \
  -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 10 \
  -vf "fps=1,hqdn3d" -vsync vfr -c:v libwebp -compression_level 6 -preset picture -q:v 50 -an -f image2 -strftime 1 "$snapshot_dir/%Y%m%d_%H%M%S.webp" \
  -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 10 \
  -c:v copy -an -f segment -segment_time 300 -strftime 1 "$record_dir/%Y%m%d_%H%M%S.mp4"

