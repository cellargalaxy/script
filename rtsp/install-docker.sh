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

if [ -z $output_dir ]; then
  read -p "please enter output_dir(default:volume):" output_dir
fi
if [ -z $output_dir ]; then
  output_dir=$server_name'_resource'
fi

if [ -z $cpu_use ]; then
  read -p "please enter cpu_use(default:0.8):" cpu_use
fi
if [ -z $cpu_use ]; then
  cpu_use="0.8"
fi

if [ -z $snapshot_save ]; then
  read -p "please enter snapshot_save(default:500):" snapshot_save
fi
if [ -z $snapshot_save ]; then
  snapshot_save=500
fi

if [ -z $record_save ]; then
  read -p "please enter record_save(default:100):" record_save
fi
if [ -z $record_save ]; then
  record_save=100
fi

echo
echo "server_name: $server_name"
echo "rtsp_url: $rtsp_url"
echo "output_dir: $output_dir"
echo "cpu_use: $cpu_use"
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
  -v $output_dir:/output \
  --net=host \
  --cpus=$cpu_use \
  -e server_name=$server_name \
  -e rtsp_url=$rtsp_url \
  -e snapshot_save=$snapshot_save \
  -e record_save=$record_save \
  $server_name

echo 'all finish'