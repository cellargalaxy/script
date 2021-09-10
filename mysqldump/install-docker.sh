#!/usr/bin/env bash

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo $text
}

if [ -z $server_name ]; then
  read -p "please enter server_name(default:server_center):" server_name
fi
if [ -z $server_name ]; then
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

if [ -z $exec_time ]; then
  read -p "please enter exec_time(default:00:00):" exec_time
fi
if [ -z $exec_time ]; then
  exec_time="00:00"
fi

log "server_name: $server_name"
log "host: $host"
log "port: $port"
log "user: $user"
log "password: ***"
log "exec_time: $exec_time"
log "input any key go on, or control+c over"
read

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
  -e exec_time=$exec_time \
  $server_name

log 'all finish'
