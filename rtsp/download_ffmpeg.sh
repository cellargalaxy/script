#!/usr/bin/env bash

set -e

if [[ -x "./ffmpeg" ]]; then
  echo "✅ 已存在 ./ffmpeg 文件，跳过下载。"
  exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"
echo "检测系统: OS=$OS, ARCH=$ARCH"

case "$OS-$ARCH" in
  Linux-x86_64)
    URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ;;
  Linux-aarch64 | Linux-arm64)
    URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
    ;;
  *)
    echo "❌ 不支持的系统架构: $OS-$ARCH"
    exit 1
    ;;
esac

if [[ -f ffmpeg.tar.xz ]]; then
  echo "📦 已下载，跳过下载。"
else
  echo "🔽 正在下载 FFmpeg for $URL"
  curl -L "$URL" -o ffmpeg.tar.xz
fi

rm -rf ./ffmpeg-master-latest-linux64-gpl
tar -xf ffmpeg.tar.xz
mv ./ffmpeg-master-latest-linux64-gpl/bin/ffmpeg .
rm -rf ffmpeg-master-latest-linux64-gpl
echo "✅ 下载完成！"