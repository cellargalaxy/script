#!/bin/sh

# 检查是否提供了输入文件
if [ -z "$1" ]; then
  echo "用法: $0 <输入文件路径> <输出文件路径>"
  echo "例如: $0 my_audio.m4a my_audio.wav"
  exit 1
fi

# 检查是否提供了输出文件
if [ -z "$2" ]; then
  echo "用法: $0 <输入文件路径> <输出文件路径>"
  echo "例如: $0 my_audio.m4a my_audio.wav"
  exit 1
fi

INPUT_FILE="$1"
# 基于输入文件名生成输出文件名 (例如 a.mp3 -> a.wav)
OUTPUT_FILE="$2"

# --- 第 1 步: 使用 ffprobe 获取源文件的采样格式 ---
# -v error:         只显示错误信息
# -select_streams a:0: 选择第一个音频流
# -show_entries stream=sample_fmt: 显示采样格式
# -of default=...:  以精简的格式输出，不带键名
SAMPLE_FMT=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_fmt -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")

# 检查是否成功获取到采样格式
if [ -z "$SAMPLE_FMT" ]; then
  echo "错误: 无法从文件 '$INPUT_FILE' 中获取音频采样格式。"
  exit 1
fi

echo "检测到源文件采样格式: $SAMPLE_FMT"

# --- 第 2 步: 根据采样格式选择对应的 WAV PCM 编解码器 ---
# WAV 格式使用 PCM 编码。我们需要根据源文件的位深度（由 sample_fmt 体现）
# 选择正确的 PCM 格式，以避免数据截断或不必要的转换。
# s16 -> signed 16-bit，le -> little-endian (WAV 标准)
# flt -> float 32-bit
# s32 -> signed 32-bit
case "$SAMPLE_FMT" in
  u8|u8p)      CODEC="pcm_u8" ;;
  s16|s16p)    CODEC="pcm_s16le" ;;
  s24|s24p)    CODEC="pcm_s24le" ;;
  s32|s32p)    CODEC="pcm_s32le" ;;
  s64|s64p)    CODEC="pcm_s64le" ;;
  flt|fltp)    CODEC="pcm_f32le" ;;
  dbl|dblp)    CODEC="pcm_f64le" ;;
  *)
    echo "警告: 未知的采样格式 '$SAMPLE_FMT'。将默认使用 pcm_s16le (16-bit)进行转换。"
    CODEC="pcm_s16le"
    ;;
esac

echo "为保证质量，已选择 WAV 编解码器: $CODEC"

# --- 第 3 步: 使用 ffmpeg 和选定的编解码器进行转换 ---
# -i "$INPUT_FILE":      指定输入文件
# -acodec $CODEC:        使用我们上面选择的 PCM 编解码器
# "$OUTPUT_FILE":        指定输出文件
echo "正在转换 '$INPUT_FILE' -> '$OUTPUT_FILE'..."
ffmpeg -i "$INPUT_FILE" -acodec $CODEC "$OUTPUT_FILE"

echo "转换完成！"