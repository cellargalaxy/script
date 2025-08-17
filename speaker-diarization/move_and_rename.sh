#!/bin/bash
# 遍历所有子目录下的 wav 文件
for f in */*.wav; do
    # 获取目录名，例如 01/01.wav -> 01
    dir=$(dirname "$f")
    # 获取文件名（不带路径）
    base=$(basename "$f")
    # 拼接新名字：目录名_文件名
    new="${dir}_${base}"
    # 移动并重命名到当前目录
    mv "$f" "$new"
done

