"""
写一个python脚本，只用标准库，无外部依赖
1. input输入一个文件夹路径，例如d:/demo
2. 在文件夹所在目录下，可能有一个csv文件：d:/demo.csv，如果该csv文件不存在，则创建
3. csv文件第一列存储的是demo文件夹下面的文件名称
4. 遍历demo下面的文件，如果文件名称在csv文件里则无须在第一列重复添加，否则添加到csv的第一列中
5. 第二列存储的是对文件的操作，有三种类型的数据
5.1 如果是空则不操作
5.2 如果是「del」字符串则删除文件
5.3 其余情况都是一个文件夹路径，则将该文件移动到该文件夹下，如果文件夹不存在则创建
6. 遍历csv文件第一列，如果文件名称在demo文件夹下存在，则执行第五步中的操作，否则跳过，遍历过程中打印执行情况
7. 最后点击任意键退出脚本
"""

import os
import csv
import shutil
import sys


def main():
    folder = input("请输入文件夹路径（例如 d:/demo）: ").strip()
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"文件夹不存在: {folder}")
        sys.exit(1)

    # csv文件路径
    parent_dir = os.path.dirname(folder)
    folder_name = os.path.basename(folder)
    csv_path = os.path.join(parent_dir, f"{folder_name}.csv")

    # 读取或创建csv
    csv_dict = {}  # filename -> operation
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    csv_dict[row[0]] = row[1] if len(row) > 1 else ""
    else:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            pass  # 创建空csv

    # 获取文件夹下的所有文件（不包括子文件夹）
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # 补全csv文件（仅添加新文件名，操作列为空）
    changed = False
    for filename in files:
        if filename not in csv_dict:
            csv_dict[filename] = ""
            changed = True

    # 保存补全后的csv
    if changed:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for k, v in csv_dict.items():
                writer.writerow([k, v])

    # 遍历csv执行操作
    print("执行文件操作：")
    for filename, operation in csv_dict.items():
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            print(f"[跳过] 文件不存在: {filename}")
            continue
        if not operation.strip():
            print(f"[无操作] {filename}")
        elif operation.strip().lower() == "del":
            try:
                os.remove(file_path)
                print(f"[删除] {filename}")
            except Exception as e:
                print(f"[删除失败] {filename}: {e}")
        else:
            target_folder = operation.strip()
            target_folder = os.path.abspath(target_folder)
            if not os.path.exists(target_folder):
                try:
                    os.makedirs(target_folder)
                    print(f"[创建文件夹] {target_folder}")
                except Exception as e:
                    print(f"[创建文件夹失败] {target_folder}: {e}")
                    continue
            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"[移动] {filename} -> {target_folder}")
            except Exception as e:
                print(f"[移动失败] {filename}: {e}")

    input("操作完成，按任意键退出...")


if __name__ == "__main__":
    main()
