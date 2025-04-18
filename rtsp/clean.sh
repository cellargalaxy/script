#!/usr/bin/env bash

mkdir -p snapshots
mkdir -p recordings

while true; do
    sleep 60
    echo 'clean start'
    ls -t snapshots/*.jpg | tail -n +$((5+1)) | xargs rm -f
    ls -t recordings/*.mp4 | tail -n +$((5+1)) | xargs rm -f
    echo 'clean done'
done