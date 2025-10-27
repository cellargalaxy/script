"""
写一个python脚本，只用标准库，无外部依赖
1. 脚本启动后，手动在终端输入多个参数
1.1 填写输入文件夹路径，必填，例如d:/mp3
1.1 填写输出文件夹路径，必填，例如f:/
2. 将输入文件夹及其下面的文件拷贝到输出文件夹下，例如f:/mp3
3. 将输出文件夹下的文件，使用五位数字编号进行重命名
4. 最后点击任意键退出脚本
"""

import os
import shutil
import sys


def get_user_paths():
    """
    提示用户输入源文件夹和输出文件夹路径。
    循环提示，直到输入有效的文件夹路径。
    """
    # 1.1 获取输入文件夹路径
    while True:
        src_dir = input("请输入【输入】文件夹路径 (例如 d:/mp3): ").strip()
        if not src_dir:
            print("错误：路径不能为空。")
        elif not os.path.isdir(src_dir):
            print(f"错误：路径 '{src_dir}' 不是一个有效的文件夹。请重试。")
        else:
            break

    # 1.2 获取输出文件夹路径
    while True:
        dst_dir = input("请输入【输出】文件夹路径 (例如 f:/): ").strip()
        if not dst_dir:
            print("错误：路径不能为空。")
        elif not os.path.isdir(dst_dir):
            print(f"错误：路径 '{dst_dir}' 不是一个有效的文件夹。请重试。")
        else:
            break

    return src_dir, dst_dir


def copy_directory(src_dir, dst_root):
    """
    将源文件夹拷贝到目标文件夹下。
    例如：copy_directory("d:/mp3", "f:/") 将创建 "f:/mp3"
    """

    # 规范化路径并获取源文件夹的名称
    src_dir_norm = os.path.normpath(src_dir)
    src_folder_name = os.path.basename(src_dir_norm)

    # 2. 计算最终的目标路径
    full_dst_path = os.path.join(dst_root, src_folder_name)

    print(f"\n--- 步骤 2: 拷贝文件夹 ---")
    print(f"源: {src_dir}")
    print(f"目标: {full_dst_path}")

    # 检查目标是否已存在，防止 copytree 失败
    if os.path.exists(full_dst_path):
        print(f"\n错误：目标路径 '{full_dst_path}' 已经存在。")
        print("为防止数据丢失，请先手动删除该文件夹，或选择一个不同的输出文件夹。")
        return None

    try:
        shutil.copytree(src_dir, full_dst_path)
        print("拷贝完成。")
        return full_dst_path
    except Exception as e:
        print(f"\n拷贝过程中发生错误: {e}")
        return None


def rename_files_in_directory(target_dir):
    """
    3. 递归重命名目标文件夹下的所有文件。
    使用两阶段重命名，以防止文件名冲突（例如 a.txt -> 00001.txt,
    而 00001.txt 原本就存在）。
    """
    print(f"\n--- 步骤 3: 重命名文件 ---")
    print(f"正在扫描: {target_dir}")

    # 阶段 1: 收集所有文件并重命名为唯一的临时名称

    files_to_rename = []
    # 递归遍历所有子文件夹
    for dirpath, _, filenames in os.walk(target_dir):
        for filename in filenames:
            # 确保我们不会意外地重命名我们刚创建的临时文件
            if not filename.startswith("__temp_rename_"):
                files_to_rename.append(os.path.join(dirpath, filename))

    if not files_to_rename:
        print("未在目标文件夹中找到需要重命名的文件。")
        return

    print(f"阶段 1: 找到 {len(files_to_rename)} 个文件，正在重命名为临时文件...")

    temp_files_map = {}  # 存储 {临时路径: 原始扩展名}

    for i, old_path in enumerate(files_to_rename):
        try:
            directory = os.path.dirname(old_path)
            # 分离文件名和扩展名
            _, extension = os.path.splitext(old_path)

            # 临时名称 (保证唯一性)
            temp_name = f"__temp_rename_{i:08d}{extension}"
            temp_path = os.path.join(directory, temp_name)

            os.rename(old_path, temp_path)
            temp_files_map[temp_path] = extension
        except Exception as e:
            print(f"  - 警告 (阶段 1): 无法重命名 '{old_path}': {e}")

    # 阶段 2: 将临时文件重命名为最终的数字编号
    print(f"\n阶段 2: 正在将 {len(temp_files_map)} 个临时文件重命名为最终编号...")

    counter = 1
    # 按临时路径排序，以确保重命名顺序是可预测的
    sorted_temp_paths = sorted(temp_files_map.keys())

    for temp_path in sorted_temp_paths:
        if not os.path.exists(temp_path):
            continue  # 可能在阶段1失败了

        try:
            directory = os.path.dirname(temp_path)
            extension = temp_files_map[temp_path]  # 获取原始扩展名

            # 格式化为五位数字
            new_name = f"{counter:05d}{extension}"
            new_path = os.path.join(directory, new_name)

            # 安全检查：如果目标数字已存在（理论上不应该，但以防万一）
            if os.path.exists(new_path):
                print(f"  - 警告 (阶段 2): 目标路径 '{new_path}' 意外存在。跳过 '{temp_path}'。")
                continue

            os.rename(temp_path, new_path)
            # 获取相对路径以便清晰显示
            relative_temp = os.path.relpath(temp_path, target_dir)
            relative_new = os.path.relpath(new_path, target_dir)
            print(f"  - {relative_temp} -> {relative_new}")

            counter += 1
        except Exception as e:
            print(f"  - 警告 (阶段 2): 无法重命名 '{temp_path}': {e}")

    print(f"\n重命名完成。总共处理 {counter - 1} 个文件。")


def wait_for_exit():
    """
    4. 等待用户按键退出
    """
    print("\n------------------------------")
    print("所有操作已完成。")
    input("按 Enter 键退出脚本...")


def main():
    """
    主执行函数
    """
    print("--- 文件夹拷贝与五位数字重命名脚本 ---")
    print("注意：本脚本仅使用Python标准库。\n")

    try:
        # 1. 获取路径
        src_dir, dst_root = get_user_paths()

        # 2. 拷贝
        new_target_dir = copy_directory(src_dir, dst_root)

        # 3. 重命名 (如果拷贝成功)
        if new_target_dir:
            rename_files_in_directory(new_target_dir)

    except KeyboardInterrupt:
        print("\n\n操作被用户中断。正在退出...")
    except Exception as e:
        print(f"\n发生未预料的严重错误: {e}")
    finally:
        # 4. 退出
        wait_for_exit()


if __name__ == "__main__":
    main()
