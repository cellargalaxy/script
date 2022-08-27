#!/bin/sh

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo "$text"
}

log "pwd: $(pwd)"

folder="backup"

if [ ! -f "$folder/backup.txt" ]; then
  log "illegal space"
  exit 0
fi

if [ -z "$host" ]; then
  host="$1"
fi
if [ -z "$host" ]; then
  log "must host"
  exit 0
fi

if [ -z "$port" ]; then
  port="$2"
fi
if [ -z "$port" ]; then
  log "must port"
  exit 0
fi

if [ -z "$user" ]; then
  user="$3"
fi
if [ -z "$user" ]; then
  log "must user"
  exit 0
fi

if [ -z "$password" ]; then
  password="$4"
fi
if [ -z "$password" ]; then
  log "must password"
  exit 0
fi

mkdir -p "$folder/$host-$port"
cd "$folder/$host-$port"

database_names=$(mysql --host "$host" --port "$port" -u"$user" -p"$password" -e "show databases;" | grep -vE "Database|information_schema|mysql|performance_schema|sys" | xargs)
log "backup: $host:$port:$database_names"

new_count=0
for database_name in $database_names; do
  filename="$(date "+%Y%m%d_%H%M%S")_$database_name.sql.gz"
  log "backup: $database_name -> $filename"
  mysqldump --lock-all-tables --compress --host "$host" --port "$port" -u"$user" -p"$password" "$database_name" | gzip >"$filename"
  #  touch $filename
  new_count=$(expr $new_count + 1)
done

if [ -z "$save_batch" ]; then
  save_batch=2
fi
save_count=$(expr $new_count \* $save_batch)
log "new_count*save_batch=save_count: $new_count * $save_batch = $save_count"

file_count=$(ls -l | grep ".gz" | wc -l)
log "file_count: $file_count"

while [ 0 -lt $save_count ] && [ $save_count -lt $file_count ]; do
  filename=$(ls -rt | head -n1)
  log "remove: $filename"
  rm -f "$filename"
  file_count=$(expr $file_count - 1)
done
