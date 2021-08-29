#!/usr/bin/env bash

#export sleepTime=1
#export command='echo print'

echo 'sleepTime:'$sleepTime'(s)'
if [ -z $sleepTime ]; then
  echo 'sleepTime is none'
  exit 1
fi

echo 'command:'$command
if [ -z "$command" ]; then
  echo 'command is none'
  exit 1
fi

while :; do
  echo $(date "+%Y-%m-%d %H:%M:%S")' exec command start:'$command
  $command
  echo $(date "+%Y-%m-%d %H:%M:%S")' exec command done:'$command

  echo $(date "+%Y-%m-%d %H:%M:%S")' exec sleep start:'$sleepTime'(s)'
  sleep $sleepTime
  echo $(date "+%Y-%m-%d %H:%M:%S")' exec sleep done:'$sleepTime'(s)'
done
