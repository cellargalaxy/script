#!/usr/bin/env bash

#export execHour=17

logFilename="log.log"

echo "">$logFilename

log() {
  time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "$time $*"
  hour=$(date "+%H")
  if [ "$hour" -ne "06" ]; then
    echo "">$logFilename
  fi
  echo "$time $*">>$logFilename
}

log "execHour: $execHour"
if [ -z "$execHour" ]; then
  log "execHour is none"
  exit 1
fi

log "remoteip: $remoteip"
if [ -z "$remoteip" ]; then
  log "remoteip is none"
  exit 1
fi

log "remoteport: $remoteport"
if [ -z "$remoteport" ]; then
  log "remoteport is none"
  exit 1
fi

log "remoteuser: $remoteuser"
if [ -z "$remoteuser" ]; then
  log "remoteuser is none"
  exit 1
fi

log "remotepwd: ***"
if [ -z "$remotepwd" ]; then
  log "remotepwd is none"
  exit 1
fi

log "remotedir: $remotedir"
if [ -z "$remotedir" ]; then
  log "remotedir is none"
  exit 1
fi

log "timeout: $timeout"
if [ -z "$timeout" ]; then
  log "timeout is none"
  exit 1
fi

while :; do
  hour=$(date "+%H")
  log "now hour is: $hour"
  if [ "$hour" -ne "$execHour" ]; then
    log "$hour not is exec hour $execHour"
    sleep 300
    continue
  fi
  log "exec command start"
  ./rsync_mirror.sh "$remoteip" "$remoteport" "$remoteuser" "$remotepwd" "$remotedir" "./local" "$timeout"
  log "exec command done"
  sleep 3600
done
