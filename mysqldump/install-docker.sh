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

while :; do
  if [ ! -z "$dbName" ]; then
    break
  fi
  read -p "please enter dbName(required):" dbName
done

while :; do
  if [ ! -z "$localPath" ]; then
    break
  fi
  read -p "please enter localPath(required):" localPath
done

log "name: $name"
log "execTime: $execTime"
log "host: $host"
log "port: $port"
log "user: $user"
log "password: ***"
log "dbName: $dbName"
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
  -e host="$host" \
  -e port="$port" \
  -e user="$user" \
  -e password="$password" \
  -e dbName="$dbName" \
  -v "$localPath":/local \
  "$name"

log 'all finish'
