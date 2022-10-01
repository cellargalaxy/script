#!/usr/bin/env bash

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo "$text"
}

if [ -z "$server_name" ]; then
  read -p "please enter server_name(default:crontab):" server_name
fi
if [ -z "$server_name" ]; then
  server_name="crontab"
fi

if [ -z "$cron" ]; then
  read -p "please enter cron(default:*/1 * * * *):" cron
fi
if [ -z "$cron" ]; then
  cron="*/1 * * * *"
fi

log "server_name: $server_name"
log "cron: $cron"
log "input any key go on, or control+c over"
read

echo "$cron cd /crontab && /crontab/job.sh" >"crontab.txt"

log 'create volume'
docker volume create $server_name'_backup'
log 'stop container'
docker stop $server_name
log 'remove container'
docker rm $server_name
log 'remove image'
docker rmi $server_name
log 'docker build'
docker build -t $server_name .
log 'docker run'
docker run -d \
  --restart=always \
  --name $server_name \
  -e server_name=$server_name \
  $server_name

log 'all finish'
