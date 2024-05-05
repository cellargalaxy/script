#!/bin/sh

if [ ! -f "jpegoptim-1.5.5-x64-linux.zip" ];then
  wget https://github.com/tjko/jpegoptim/releases/download/v1.5.5/jpegoptim-1.5.5-x64-linux.zip
fi

if [ ! -f "jpegoptim" ];then
  unzip jpegoptim-1.5.5-x64-linux.zip
fi

rm COPYRIGHT LICENSE README
chmod u+x jpegoptim

while :; do
  if [ ! -z "$image_path" ]; then
    break
  fi
  read -p "please enter image_path(required):" image_path
done

echo "deal: $image_path"

echo "input any key go on, or control+c over"
read

find $image_path -name "*.jpg" | xargs mogrify -resize 1024x1024
find $image_path -name "*.jpg" | xargs ./jpegoptim --strip-all -p
