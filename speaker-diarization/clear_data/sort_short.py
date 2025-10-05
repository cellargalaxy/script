"""
写一个python脚本，只用标准库，无外部依赖
环境变量中有ffmpeg与ffprobe
1. 脚本启动后，手动在终端输入三个参数
1.1 输入文件夹路径，必填，例如d:/demo
1.2 输出文件夹路径，选填，不填默认与输入文件夹路径同级，名称为short，例如d:/short
1.3 最小长度（秒）
2. 输入文件夹下有多个音频/视频文件
3. 遍历输入文件夹下全部文件，将长度小于最小长度的文件，移动到输出文件夹下，在终端输出操作日志
3. 统计并打印保留下来的文件的总长度
4. 最后点击任意键退出脚本
"""

import os
import shutil
import subprocess


def get_media_duration(file_path):
    # Use ffprobe to get the duration of media file
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"无法获取文件长度: {file_path}, 错误: {e}")
        return None


def main():
    print("请输入输入文件夹路径（必填，例如 d:/demo）：")
    input_dir = input().strip()
    if not input_dir or not os.path.isdir(input_dir):
        print("输入路径无效或不是真实文件夹。")
        return

    print("请输入输出文件夹路径（选填，默认与输入文件夹同级，名称为data_short）：")
    output_dir = input().strip()
    if not output_dir:
        parent = os.path.dirname(os.path.abspath(input_dir))
        output_dir = os.path.join(parent, "data_short")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("请输入最小长度（秒）：")
    try:
        min_length = float(input().strip())
        if min_length < 0:
            raise ValueError
    except ValueError:
        print("最小长度必须为正数。")
        return

    keep_total_duration = 0.0
    files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    moved_count = 0
    for f in files:
        abs_path = os.path.join(input_dir, f)
        duration = get_media_duration(abs_path)
        if duration is None:
            print(f"跳过无法识别的文件: {f}")
            continue
        if duration < min_length:
            # Move to output_dir
            shutil.move(abs_path, os.path.join(output_dir, f))
            print(f"移动文件到 {output_dir}: {f} (时长: {duration:.2f}秒)")
            moved_count += 1
        else:
            keep_total_duration += duration
            print(f"保留文件: {f} (时长: {duration:.2f}秒)")

    print("\n操作完成！")
    print(f"共移动 {moved_count} 个文件到 {output_dir}")
    print(f"保留下来的文件总时长: {keep_total_duration:.2f} 秒\n")
    input("按任意键退出...")


if __name__ == "__main__":
    main()
