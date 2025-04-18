#!/usr/bin/env bash

if [ -z $server_name ]; then
  read -p "please enter server_name(default:rtsp):" server_name
fi
if [ -z $server_name ]; then
  server_name="rtsp"
fi

echo
echo "server_name: $server_name"
echo "input any key go on, or control+c over"
read

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
  -e server_name=$server_name \
  $server_name

echo 'all finish'