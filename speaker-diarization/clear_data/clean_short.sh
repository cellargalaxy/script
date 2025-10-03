#!/bin/bash

# 输入两个参数：文件夹路径 和 最小语音长度（秒）
read -p "请输入文件夹路径: " folder
read -p "请输入最小语音长度(秒): " min_len

# 检查文件夹是否存在
if [[ ! -d "$folder" ]]; then
    echo "[ERR ] 文件夹不存在: $folder"
    exit 1
fi

# 初始化总时长
total_duration=0

echo "开始扫描文件夹: $folder"
echo "最小语音长度: ${min_len} 秒"
echo "-----------------------------------"

# 遍历文件夹下的所有文件
for file in "$folder"/*; do
    # 确保是文件
    [[ -f "$file" ]] || continue

    # 用 ffprobe 获取时长（秒）
    duration=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)

    # 如果时长获取失败，跳过
    if [[ -z "$duration" ]]; then
        printf "[SKIP] 无法获取时长: %-40s\n" "$(basename "$file")"
        continue
    fi

    # 四舍五入取整
    duration_int=$(printf "%.0f" "$duration")

    if (( duration_int < min_len )); then
        printf "[DEL ] 删除: %-40s (时长 %4ds < %4ds)\n" "$(basename "$file")" "$duration_int" "$min_len"
        rm -f "$file"
    else
        printf "[KEEP] 保留: %-40s (时长 %4ds)\n" "$(basename "$file")" "$duration_int"
        total_duration=$((total_duration + duration_int))
    fi
done

echo "-----------------------------------"
printf "保留下来的文件总时长: %d 秒\n" "$total_duration"
echo "脚本将在 60 秒后退出..."
sleep 60
