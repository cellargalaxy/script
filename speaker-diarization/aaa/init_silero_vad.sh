#!/bin/sh

mkdir -p model
cd model

if [ -e "silero-vad" ]; then
  echo "silero-vad exists, exiting script."
  exit 0
fi

git clone --depth 1 https://github.com/snakers4/silero-vad