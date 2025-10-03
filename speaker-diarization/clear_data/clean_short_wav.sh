#!/bin/bash

# 输入两个参数：文件夹路径 和 最小语音长度（秒）
read -p "请输入文件夹路径: " folder
read -p "请输入最小语音长度(秒): " min_len

# 检查文件夹是否存在
if [[ ! -d "$folder" ]]; then
    echo "❌ 文件夹不存在: $folder"
    exit 1
fi

# 初始化总时长
total_duration=0

# 遍历文件夹下的所有文件
for file in "$folder"/*; do
    # 确保是文件
    [[ -f "$file" ]] || continue

    # 用 ffprobe 获取时长（秒）
    duration=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)

    # 如果时长获取失败，跳过
    if [[ -z "$duration" ]]; then
        echo "⚠️ 无法获取时长: $file"
        continue
    fi

    # 四舍五入取整
    duration_int=$(printf "%.0f" "$duration")

    if (( duration_int < min_len )); then
        echo "🗑️ 删除: $file (时长 ${duration_int}s < ${min_len}s)"
        rm -f "$file"
    else
        echo "✅ 保留: $file (时长 ${duration_int}s)"
        total_duration=$((total_duration + duration_int))
    fi
done

echo "-----------------------------------"
echo "保留下来的文件总时长: ${total_duration} 秒"
echo "脚本将在 60 秒后退出..."
sleep 60
