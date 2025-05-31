#!/bin/bash

# 检查输入文件是否存在
if [ -z "$1" ]; then
    echo "用法: $0 输入视频文件.mkv"
    exit 1
fi

INPUT_VIDEO="$1"
BASENAME=$(basename "$INPUT_VIDEO" .mkv)
AUDIO_FILE="${BASENAME}_audio.wav"

# 1. 提取音频
echo "🎧 提取音频中..."
ffmpeg -y -i "$INPUT_VIDEO" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$AUDIO_FILE"

# 2. 使用 demucs 进行分离
echo "🎛️ 使用 demucs 分离音频..."
python3 -m demucs -d cpu "$AUDIO_FILE"

# 3. 获取分离后人声路径
VOCALS_FILE="separated/htdemucs/${BASENAME}_audio/vocals.wav"

if [ ! -f "$VOCALS_FILE" ]; then
    echo "❌ 未找到人声文件: $VOCALS_FILE"
    exit 2
fi

# 4. 合成人声 + 视频画面
OUTPUT_VIDEO="${BASENAME}_vocals_only.mkv"
echo "🎬 合成新视频（画面 + 分离后人声）..."
ffmpeg -y -i "$INPUT_VIDEO" -i "$VOCALS_FILE" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "$OUTPUT_VIDEO"

echo "✅ 完成！输出文件: $OUTPUT_VIDEO"
