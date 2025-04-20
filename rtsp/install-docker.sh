#!/usr/bin/env bash

if [ -z $server_name ]; then
  read -p "please enter server_name(default:rtsp):" server_name
fi
if [ -z $server_name ]; then
  server_name="rtsp"
fi

while :; do
  if [ ! -z $rtsp_url ]; then
    break
  fi
  read -p "please enter rtsp_url(required):" rtsp_url
done

if [ -z $snapshot_save ]; then
  read -p "please enter snapshot_save(default:500):" snapshot_save
fi
if [ -z $snapshot_save ]; then
  snapshot_save=500
fi

if [ -z $record_save ]; then
  read -p "please enter record_save(default:300):" record_save
fi
if [ -z $record_save ]; then
  record_save=300
fi

echo
echo "server_name: $server_name"
echo "rtsp_url: $rtsp_url"
echo "snapshot_save: $snapshot_save"
echo "record_save: $record_save"
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