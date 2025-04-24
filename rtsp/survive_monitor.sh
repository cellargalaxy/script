#!/usr/bin/env sh

log_info() {
  echo "INFO $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
  echo "ERRO $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

if pgrep ffmpeg > /dev/null; then
  echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ffmpeg_running\": true}"
else
  echo -e "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\r\n{\"ffmpeg_running\": false}"
fi
