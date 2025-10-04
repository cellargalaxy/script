#!/bin/bash

: <<'SCRIPT_INFO'
写一个shell脚本，read一个文件夹路径，给里面的文件按数字升序重命名
SCRIPT_INFO

# 检查是否有输入参数，如果没有，则提示用户输入
if [ -z "$1" ]; then
    read -p "请输入要重命名文件的文件夹路径: " FOLDER_PATH
else
    FOLDER_PATH="$1"
fi

# 检查路径是否为空
if [ -z "$FOLDER_PATH" ]; then
    echo "错误：未输入文件夹路径。"
    exit 1
fi

# 检查文件夹是否存在
if [ ! -d "$FOLDER_PATH" ]; then
    echo "错误：文件夹 '$FOLDER_PATH' 不存在或不是一个有效的目录。"
    exit 1
fi

# 进入目标文件夹
cd "$FOLDER_PATH" || { echo "错误：无法进入目录 '$FOLDER_PATH'。"; exit 1; }

echo "正在处理文件夹: $(pwd)"

# 计数器，从 1 开始
i=1

# 查找所有文件（-type f），并按默认顺序（通常是字母/数字顺序）循环
# 注意：这个脚本是按 'ls' 的默认顺序来重命名，即如果原文件名是 'a1.txt', 'a10.txt', 'a2.txt'，
# 它会按这个顺序重命名。如果需要更严格的自然数排序，需要更复杂的逻辑。
for file in *; do
    # 检查是否是文件，排除目录
    if [ -f "$file" ]; then
        # 获取文件的扩展名
        extension="${file##*.}"

        # 构造新的文件名。这里使用 %04d 确保数字至少有4位，比如 0001, 0002...
        # 这样有助于保持重命名后的文件在文件管理器中的排序整齐。
        # 如果不需要0填充，可以直接使用 "$i"
        new_name=$(printf "%04d" "$i")

        # 加上扩展名（如果原始文件有扩展名且不等于原文件名本身）
        if [ "$extension" != "$file" ]; then
             new_name="${new_name}.${extension}"
        fi

        # 检查新名称是否与旧名称不同，避免不必要的重命名操作
        if [ "$file" != "$new_name" ]; then
            echo "重命名：'$file' -> '$new_name'"
            mv "$file" "$new_name"
        else
            echo "跳过：'$file' 名称已匹配 '$new_name'"
        fi

        # 计数器递增
        i=$((i + 1))
    fi
done

echo "---"
echo "重命名完成！共处理了 $((i - 1)) 个文件。"