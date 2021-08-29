#!/usr/bin/env bash

while :; do
  if [ ! -z $sleepTime ]; then
    break
  fi
  read -p "please enter sleepTime(required):" sleepTime
done

while :; do
  if [ ! -z $command ]; then
    break
  fi
  read -p "please enter command(required):" command
done

echo 'sleepTime:'$sleepTime
echo 'command:'$command
echo 'input any key go on, or control+c over'
read

echo 'create volume'
docker volume create local

echo 'stop container'
docker stop script

echo 'remove container'
docker rm script

echo 'remove image'
docker rmi script

echo 'docker build'
docker build -t script .

echo 'docker run'
docker run -d \
  --restart=always \
  --name script \
  -v log:/log \
  -e sleepTime=$sleepTime \
  -e command=$command \
  -v local:local \
  script

echo 'all finish'
