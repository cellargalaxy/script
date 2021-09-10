#!/usr/bin/env bash

logFilePath="log.log"

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo "$text"
  if [ "$(date "+%H%M")" == "0000" ]; then
    echo "" >$logFilePath
  fi
  echo "$text" >>$logFilePath
}

log "execTime: $execTime"
if [ -z "$execTime" ]; then
  log "execTime is none"
  exit 1
fi

log "host: $host"
if [ -z "$host" ]; then
  log "host is none"
  exit 1
fi

log "port: $port"
if [ -z "$port" ]; then
  log "port is none"
  exit 1
fi

log "user: $user"
if [ -z "$user" ]; then
  log "user is none"
  exit 1
fi

log "password: $password"
if [ -z "$password" ]; then
  log "password is none"
  exit 1
fi

log "dbName: $dbName"
if [ -z "$dbName" ]; then
  log "dbName is none"
  exit 1
fi

while :; do
  time=$(date "+%H%M")
  log "now time is: $time"
  runNow=$(<runNow.txt)
  log "runNow is: $runNow"
  if [ "$time" != "$execTime" ] && [ "$runNow" != "true" ]; then
    log "$time not is exec time $execTime or not run now: $runNow"
    sleep 30
    continue
  fi
  log "exec command start"
  bash ./mysqldump.sh "$host" "$port" "$user" "$password" "$dbName" "./local"
  log "exec command done"
  sleep 3600
done