#!/usr/bin/env bash

log() {
  time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "$time $*"
}

while :; do
  if [ ! -z "$name" ]; then
    break
  fi
  read -p "please enter name(required):" name
done

while :; do
  if [ ! -z "$execHour" ]; then
    break
  fi
  read -p "please enter execHour(required):" execHour
done

while :; do
  if [ ! -z "$remoteip" ]; then
    break
  fi
  read -p "please enter remoteip(required):" remoteip
done

while :; do
  if [ ! -z "$remoteport" ]; then
    break
  fi
  read -p "please enter remoteport(required):" remoteport
done

while :; do
  if [ ! -z "$remoteuser" ]; then
    break
  fi
  read -p "please enter remoteuser(required):" remoteuser
done

while :; do
  if [ ! -z "$remotepwd" ]; then
    break
  fi
  read -p "please enter remotepwd(required):" remotepwd
done

while :; do
  if [ ! -z "$remotedir" ]; then
    break
  fi
  read -p "please enter remotedir(required):" remotedir
done

while :; do
  if [ ! -z "$timeout" ]; then
    break
  fi
  read -p "please enter timeout(required):" timeout
done

log "name: $name"
log "execHour: $execHour"
log "remoteip: $remoteip"
log "remoteport: $remoteport"
log "remoteuser: $remoteuser"
log "remotepwd: ***"
log "remotedir: $remotedir"
log "localfile: $localfile"
log "timeout: $timeout"
log "input any key go on, or control+c over"
read

echo 'create volume'
docker volume create "$name"'_data'

echo 'stop container'
docker stop $name

echo 'remove container'
docker rm $name

echo 'remove image'
docker rmi $name

echo 'docker build'
docker build -t $name .

echo 'docker run'
docker run -d \
  --restart=always \
  --name $name \
  -e execHour="$execHour" \
  -e remoteip="$remoteip" \
  -e remoteport="$remoteport" \
  -e remoteuser="$remoteuser" \
  -e remotepwd="$remotepwd" \
  -e remotedir="$remotedir" \
  -e timeout="$timeout" \
  -v "$name"'_data':/local \
  $name

echo 'all finish'
