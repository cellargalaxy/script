#!/bin/sh

mkdir -p bin
cd bin

if [ -e "ffmpeg" ]; then
  echo "ffmpeg exists, exiting script."
  exit 0
fi

url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz"
filename="ffmpeg.tar.xz"
tempfile="${filename}.part"

# 如果正式文件已经存在，跳过下载
if [ -f "$filename" ]; then
  echo "File $filename already exists. Skipping download."
else
  echo "Downloading $filename..."
  wget -c "$url" -O "$tempfile"
  # 如果 wget 成功退出（返回码为 0），说明下载完成
  if [ $? -eq 0 ]; then
    mv "$tempfile" "$filename"
    echo "Download complete. Renamed to $filename"
  else
    echo "Download incomplete or failed. Keeping partial file: $tempfile"
  fi
fi

tar -xvf $filename
mv ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/* .