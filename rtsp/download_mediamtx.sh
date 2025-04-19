#!/usr/bin/env bash

set -e

if [[ -x "./mediamtx" ]]; then
  echo "✅ 已存在 ./mediamtx 文件，跳过下载。"
  exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"
echo "检测系统: OS=$OS, ARCH=$ARCH"

case "$OS-$ARCH" in
  Linux-x86_64)
    URL="https://github.com/bluenviron/mediamtx/releases/download/v1.12.0/mediamtx_v1.12.0_linux_amd64.tar.gz"
    ;;
  Linux-aarch64 | Linux-arm64)
    URL="https://github.com/bluenviron/mediamtx/releases/download/v1.12.0/mediamtx_v1.12.0_linux_arm64v8.tar.gz"
    ;;
  *)
    echo "❌ 不支持的系统架构: $OS-$ARCH"
    exit 1
    ;;
esac

if [[ -f mediamtx.tar.gz ]]; then
  echo "📦 已下载，跳过下载。"
else
  echo "🔽 正在下载 mediamtx for $URL"
  curl -L "$URL" -o mediamtx.tar.gz
fi

tar -zxvf mediamtx.tar.gz
rm -rf LICENSE
rm -rf mediamtx.yml
echo "✅ 下载完成！"