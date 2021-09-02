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
  if [ ! -z "$execTime" ]; then
    break
  fi
  read -p "please enter execTime(required):" execTime
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

while :; do
  if [ ! -z "$localPath" ]; then
    break
  fi
  read -p "please enter localPath(required):" localPath
done

log "name: $name"
log "execTime: $execTime"
log "remoteip: $remoteip"
log "remoteport: $remoteport"
log "remoteuser: $remoteuser"
log "remotepwd: ***"
log "remotedir: $remotedir"
log "timeout: $timeout"
log "localPath: $localPath"
log "input any key go on, or control+c over"
read

echo 'stop container'
docker stop "$name"

echo 'remove container'
docker rm "$name"

echo 'remove image'
docker rmi "$name"

echo 'docker build'
docker build -t "$name" .

echo 'docker run'
docker run -d \
  --restart=always \
  --name "$name" \
  -e execTime="$execTime" \
  -e remoteip="$remoteip" \
  -e remoteport="$remoteport" \
  -e remoteuser="$remoteuser" \
  -e remotepwd="$remotepwd" \
  -e remotedir="$remotedir" \
  -e timeout="$timeout" \
  -v "$localPath":/local \
  "$name"

log 'all finish'
