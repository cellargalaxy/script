"""
写一个python脚本，使用纯英文编写，只用标准库，无外部依赖
1. 脚本启动后，手动在终端输入两个参数
1.1 填写输入文件夹路径，必填，例如d:/mp3
1.1 填写输出文件夹路径，必填，例如f:/
2. 在输出文件夹下创建一个文件夹，文件夹名称格式是「原文件夹名称_YYYYMMDD」，例如f:/mp3_20251212
2. 递归遍历输入文件夹的全部文件
3. 将输入文件夹的文件复制到原文件夹名称_YYYYMMDD
4. 输出文件夹下的文件名名称，使用输入文件夹的路径，使用「-」拼接而来。如果原文件名称都是ascii，则保留原文件名称，都在使用四位数字编号进行重命名
4.1 例如输入文件是「d:/mp3/aaa/bbb/ccc.mp3」，输出文件是「f:/mp3_20251212/aaa_bbb_ccc.mp3」
4.1 例如输入文件是「d:/mp3/aaa/bbb/文件名称.mp3」，输出文件是「f:/mp3_20251212/aaa_bbb_0001.mp3」
5. 最后点击任意键退出脚本
"""

import os
import shutil
import datetime


def is_ascii(text: str) -> bool:
    """Check whether a string contains ASCII characters only"""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def main():
    # Read user input
    src_dir = input("Enter source folder path: ").strip()
    dst_root = input("Enter output folder path: ").strip()

    if not os.path.isdir(src_dir):
        print("Source folder does not exist.")
        input("Press any key to exit...")
        return

    if not os.path.isdir(dst_root):
        print("Output folder does not exist.")
        input("Press any key to exit...")
        return

    # Prepare destination folder name
    src_dir = os.path.abspath(src_dir)
    dst_root = os.path.abspath(dst_root)

    src_name = os.path.basename(src_dir.rstrip("\\/"))
    today = datetime.datetime.now().strftime("%Y%m%d")
    dst_dir = os.path.join(dst_root, f"{src_name}_{today}")

    os.makedirs(dst_dir, exist_ok=True)

    counter = 1

    # Walk through source directory
    for root, _, files in os.walk(src_dir):
        for file_name in files:
            src_file = os.path.join(root, file_name)

            # Build relative path
            rel_path = os.path.relpath(src_file, src_dir)
            rel_no_ext, ext = os.path.splitext(rel_path)

            # Replace path separators with underscore
            safe_name = rel_no_ext.replace("\\", "_").replace("/", "_")

            if is_ascii(file_name):
                new_name = safe_name + ext
            else:
                new_name = f"{safe_name.rsplit('_', 1)[0]}_{counter:04d}{ext}"
                counter += 1

            dst_file = os.path.join(dst_dir, new_name)

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            shutil.copy2(src_file, dst_file)

    print("Copy completed successfully.")
    input("Press any key to exit...")


if __name__ == "__main__":
    main()
