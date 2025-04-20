#!/usr/bin/env bash

if [ -z $snapshot_save ]; then
  snapshot_save=500
fi
if [ -z $record_save ]; then
  record_save=300
fi

snapshot_dir="output/snapshot"
record_dir="output/record"
mkdir -p $snapshot_dir
mkdir -p $record_dir

while true; do
    echo 'clean start,snapshot_save:'$snapshot_save',record_save:'$record_save
    ls -t output/snapshot/*.webp | tail -n +$(($snapshot_save+1)) | xargs rm -f
    ls -t output/record/*.mp4 | tail -n +$(($record_save+1)) | xargs rm -f
    echo 'clean done'
    sleep 60
done


#snapshot_dir="output/snapshot"
#record_dir="output/record"
#
#mkdir -p $snapshot_dir
#mkdir -p $record_dir
#
#while true; do
#  sleep 60
#  echo 'clean start'
#  cd "$snapshot_dir" || exit
#  ls -t *.webp | sed -e '1,500d' | xargs rm -f
#  cd ../..
#  cd "$record_dir" || exit
#  ls -t *.mp4 | sed -e '1,5d' | xargs rm -f
#  cd ../..
#  echo 'clean done'
#done

#cd "$snapshot_dir" || exit
#for webp_file in *.webp; do
#    tar -caf "${webp_file}.tar.xz" "$webp_file"
#    rm -rf $webp_file
#done
#ls -t *.tar.xz | sed -e '1,500d' | xargs rm -f
#cd ../..
#
#cd "$record_dir" || exit
#for mp4_file in *.mp4; do
#    tar -caf "${mp4_file}.tar.xz" "$mp4_file"
#    rm -rf $mp4_file
#done
#ls -t *.tar.xz | sed -e '1,5d' | xargs rm -f
#cd ../..

