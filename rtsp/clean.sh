#!/usr/bin/env sh

log_info() {
  echo "INFO $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

if [ -z $snapshot_dir ]; then
  snapshot_dir="output/snapshot"
fi
if [ -z $record_dir ]; then
  record_dir="output/record"
fi
if [ -z $snapshot_save ]; then
  snapshot_save=500
fi
if [ -z $record_save ]; then
  record_save=300
fi

mkdir -p $snapshot_dir
mkdir -p $record_dir

while true; do
    log_info 'clean start, snapshot_save:'$snapshot_save',record_save:'$record_save
    ls -t $snapshot_dir/*.webp | tail -n +$(($snapshot_save+1)) | xargs rm -f
    ls -t $record_dir/*.mp4 | tail -n +$(($record_save+1)) | xargs rm -f
    log_info 'clean done'
    sleep 60
done
