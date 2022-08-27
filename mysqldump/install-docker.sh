#!/usr/bin/env bash

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo "$text"
}

if [ -z "$server_name" ]; then
  read -p "please enter server_name(default:mysqldump):" server_name
fi
if [ -z "$server_name" ]; then
  server_name="mysqldump"
fi

while :; do
  if [ ! -z "$host" ]; then
    break
  fi
  read -p "please enter host(required):" host
done

while :; do
  if [ ! -z "$port" ]; then
    break
  fi
  read -p "please enter port(required):" port
done

while :; do
  if [ ! -z "$user" ]; then
    break
  fi
  read -p "please enter user(required):" user
done

while :; do
  if [ ! -z "$password" ]; then
    break
  fi
  read -p "please enter password(required):" password
done

if [ -z "$save_batch" ]; then
  read -p "please enter save_batch(default:2):" save_batch
fi
if [ -z "$save_batch" ]; then
  save_batch="2"
fi

if [ -z "$cron" ]; then
  read -p "please enter cron(default:0 0 * * *):" cron
fi
if [ -z "$cron" ]; then
  cron="0 0 * * *"
fi

log "server_name: $server_name"
log "host: $host"
log "port: $port"
log "user: $user"
log "password: ***"
log "cron: $cron"
log "input any key go on, or control+c over"
read

echo "$cron cd / && /mysqldump.sh" >"crontab.txt"

echo 'create volume'
docker volume create $server_name'_backup'
echo 'stop container'
docker stop $server_name
echo 'remove container'
docker rm $server_name
echo 'remove image'
docker rmi $server_name
echo 'docker build'
docker build -t $server_name .
echo 'docker run'
docker run -d \
  --restart=always \
  --name $server_name \
  -v $server_name'_backup':/backup \
  -e server_name=$server_name \
  -e host=$host \
  -e port=$port \
  -e user=$user \
  -e password=$password \
  -e save_batch=$save_batch \
  $server_name

log 'all finish'
