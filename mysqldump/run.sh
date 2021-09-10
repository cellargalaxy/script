#!/bin/sh

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo $text
}

if [ -z "$exec_time" ]; then
  exec_time="00:00"
fi

while :; do
  now_time=$(date "+%H:%M")
  if [ $now_time != $exec_time ]; then
    log "now time $now_time not is exec time $exec_time"
    sleep 30
    continue
  fi
  log "exec command start"
  sh ./mysqldump.sh
  log "exec command done"
  sleep 30
done