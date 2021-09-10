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

if [ "$#" != 6 ]; then
  log "error param count: $#, Usage: mysqldump.sh host port user password dbName localPath\n"
  exit 1
fi

host="$1"
port="$2"
user="$3"
password="$4"
dbName="$5"
localPath="$6"

log "host: ""$host"
log "port: ""$port"
log "user: ""$user"
log "password: ***"
log "dbName: ""$dbName"
log "localPath: ""$localPath"

sqlFilePath="$localPath""/""$dbName"".sql"
sqlBackFilePath="$localPath""/""$dbName"".sql.back"

log "sqlFilePath: ""$sqlFilePath"
log "sqlBackFilePath: ""$sqlBackFilePath"

if [ -f "$sqlFilePath" ];then
  rm -rf "$sqlBackFilePath"
  mv "$sqlFilePath" "$sqlBackFilePath"
fi

mysqldump --lock-all-tables --compress --host "$host" --port "$port" -u"$user" -p"$password" "$dbName" > "$sqlFilePath"
result="$?"

if [ "$result" != 0 ];then
  log "mysqldump fail: ""$result"
  rm -rf "$sqlFilePath"
fi