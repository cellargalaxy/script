#!/bin/bash

: <<'SCRIPT_INFO'
输入一个任意类型的语音文件的路径，使用ffprobe与ffmpeg，尽可能保留原数据质量，转为wav格式

写一个shell脚本，将方法一封装为一个函数，创建另外一个函数
入参是input文件夹路径与output文件夹路径，遍历input文件夹下面的文件，输出到output文件夹下，如果output文件夹不存在则自动创建
如果output文件夹参数没有输入，则output文件夹「wav」命名
在终端要求输入input文件夹路径与output文件夹路径，input文件夹路径必填，不需要在调用脚本时就填写入参
允许手动选择采样率，可选范围是16000Hz//44100Hz/48000Hz，如果不选则默认原文件的采样率
SCRIPT_INFO

# -----------------------------------------------------------------------------
# 功能: 检查必要的依赖命令 (ffmpeg, ffprobe) 是否存在
# -----------------------------------------------------------------------------
check_dependencies() {
    if ! command -v ffmpeg &> /dev/null || ! command -v ffprobe &> /dev/null; then
        echo "错误: 脚本需要 'ffmpeg' 和 'ffprobe'。"
        echo "请先安装 FFmpeg 套件后再运行此脚本。"
        exit 1
    fi
}

# 全局变量: 用户选择的采样率 (为空则保持原采样率)
SELECTED_SR=""

# -----------------------------------------------------------------------------
# 功能: 将单个音频文件以最高质量转换为 WAV 格式
# 入参 1: 输入文件路径
# 入参 2: 输出文件路径
# -----------------------------------------------------------------------------
convert_to_wav() {
    local input_file="$1"
    local output_file="$2"

    # 1. 使用 ffprobe 获取源文件的采样格式
    local sample_fmt
    sample_fmt=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_fmt -of default=noprint_wrappers=1:nokey=1 "$input_file")

    if [ -z "$sample_fmt" ]; then
        echo "警告: 无法获取 '$input_file' 的采样格式，将跳过此文件。"
        return
    fi

    # 2. 根据采样格式选择最匹配的 WAV PCM 编解码器
    local codec
    case "$sample_fmt" in
      u8|u8p)      codec="pcm_u8" ;;
      s16|s16p)    codec="pcm_s16le" ;;
      s24|s24p)    codec="pcm_s24le" ;;
      s32|s32p)    codec="pcm_s32le" ;;
      s64|s64p)    codec="pcm_s64le" ;;
      flt|fltp)    codec="pcm_f32le" ;;
      dbl|dblp)    codec="pcm_f64le" ;;
      *)
        echo "警告: '$input_file' 的采样格式 '$sample_fmt' 不常见。默认使用 pcm_s16le。"
        codec="pcm_s16le"
        ;;
    esac

    # 3. 构造 ffmpeg 命令参数
    if [ -n "$SELECTED_SR" ]; then
        echo "  -> 正在转换: $(basename "$input_file") (格式: $sample_fmt, 编码器: $codec, 采样率: $SELECTED_SR)"
        ffmpeg -i "$input_file" -acodec "$codec" -ar "$SELECTED_SR" "$output_file" -hide_banner -loglevel error
    else
        echo "  -> 正在转换: $(basename "$input_file") (格式: $sample_fmt, 编码器: $codec, 采样率: 保持原始)"
        ffmpeg -i "$input_file" -acodec "$codec" "$output_file" -hide_banner -loglevel error
    fi
}

# -----------------------------------------------------------------------------
# 功能: 遍历文件夹，批量转换所有文件
# 入参 1: 输入文件夹路径
# 入参 2: (可选) 输出文件夹路径
# -----------------------------------------------------------------------------
batch_convert_folder() {
    local input_dir="$1"
    local output_dir_arg="$2"
    local output_dir

    # 1. 确定输出文件夹路径
    if [ -z "$output_dir_arg" ]; then
        # 获取输入目录的上级目录
        local parent_dir
        parent_dir=$(dirname "$input_dir")
        # 输出目录固定为上级目录下的 wav
        output_dir="${parent_dir}/wav"
        echo "未指定输出文件夹，将使用默认路径: $output_dir"
    else
        output_dir="$output_dir_arg"
    fi

    # 2. 如果输出文件夹不存在，则创建它
    if [ ! -d "$output_dir" ]; then
        echo "输出文件夹不存在，正在创建: $output_dir"
        mkdir -p "$output_dir"
    fi

    # 3. 遍历输入文件夹中的所有文件
    local file_count=0
    for file in "$input_dir"/*; do
        # 检查是否为文件 (而不是目录)
        if [ -f "$file" ]; then
            # 构建输出文件名
            local base_name
            base_name=$(basename -- "$file")
            local output_file="${output_dir}/${base_name%.*}.wav"

            # 调用单个文件转换函数
            convert_to_wav "$file" "$output_file"
            ((file_count++))
        fi
    done

    if [ "$file_count" -eq 0 ]; then
        echo "在 '$input_dir' 中没有找到任何文件可供转换。"
    else
        echo "--------------------------------------------------"
        echo "批量转换完成！共处理了 $file_count 个文件。"
        echo "所有 WAV 文件已保存至: $output_dir"
        echo "--------------------------------------------------"
    fi
}

# -----------------------------------------------------------------------------
# 主函数 - 脚本执行入口
# -----------------------------------------------------------------------------
main() {
    # 首先，检查依赖项
    check_dependencies

    # 提示用户输入路径
    local input_dir
    local output_dir

    echo "--- 音频批量转换为 WAV 工具 ---"

    # 获取输入文件夹路径 (必填)
    while true; do
        read -p "请输入源音频文件夹的完整路径: " input_dir
        if [ -z "$input_dir" ]; then
            echo "错误: 输入文件夹路径不能为空，请重新输入。"
        elif [ ! -d "$input_dir" ]; then
            echo "错误: 文件夹 '$input_dir' 不存在，请检查路径是否正确。"
        else
            break
        fi
    done

    # 获取输出文件夹路径 (选填)
    read -p "请输入输出文件夹的路径 (直接回车则自动创建): " output_dir

    # 采样率选择
    echo ""
    echo "请选择采样率 (回车保持原始采样率):"
    echo "1) 16000 Hz"
    echo "2) 44100 Hz"
    echo "3) 48000 Hz"
    read -p "请输入选项编号 (1/2/3): " sr_choice
    case "$sr_choice" in
        1) SELECTED_SR=16000 ;;
        2) SELECTED_SR=44100 ;;
        3) SELECTED_SR=48000 ;;
        *) SELECTED_SR="" ;;
    esac

    echo "" # 换行

    # 调用批量处理函数
    batch_convert_folder "$input_dir" "$output_dir"
}

# --- 执行主函数 ---
main