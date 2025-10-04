$pythonScript = @"
"""
写一个python脚本，只用标准库，无外部依赖
环境变量中有ffmpeg与ffprobe
1. 脚本启动后，手动在终端输入两个参数，文件夹路径与最小长度（秒）
2. 文件夹路径下有多个音频/视频文件
3. 遍历文件夹下全部文件，删除长度小于最小长度的文件，在终端输出操作日志
3. 统计并打印保留下来的文件的总长度
4. 最后点击任意键退出脚本
"""

import os
import sys
import subprocess


def get_duration(file_path):
    """Return duration in seconds using ffprobe (must be in PATH)."""
    try:
        # ffprobe command to get duration
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else 0.0
    except Exception as e:
        print(f"  [ERROR] Getting duration for {file_path}: {e}")
        return 0.0


def main():
    # 1. 用户输入
    folder = input("请输入文件夹路径: ").strip()
    min_length_str = input("请输入最小长度(秒): ").strip()
    try:
        min_length = float(min_length_str)
    except ValueError:
        print("最小长度格式错误，需为数字。")
        sys.exit(1)
    if not os.path.isdir(folder):
        print("文件夹路径不存在。")
        sys.exit(1)

    print(f"\n遍历文件夹: {folder}")
    print(f"最小长度: {min_length} 秒\n")

    total_duration = 0.0
    kept_files = []

    # 2. 遍历文件夹
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            duration = get_duration(fpath)
            if duration < min_length:
                try:
                    os.remove(fpath)
                    print(f"  删除: {fname} ({duration:.2f}s < {min_length}s)")
                except Exception as e:
                    print(f"  [ERROR] 删除失败: {fname} - {e}")
            else:
                print(f"  保留: {fname} ({duration:.2f}s)")
                kept_files.append(fname)
                total_duration += duration

    print("\n操作完成。")
    print(f"保留下来的文件数量: {len(kept_files)}")
    print(f"保留文件的总长度: {total_duration:.2f} 秒")
    print(f"保留文件的总长度: {total_duration / 60.0:.2f} 分")

    input("\n按任意键退出...")


if __name__ == "__main__":
    main()

"@
$scriptPath = "$env:TEMP\temp_script.py"
Set-Content -Path $scriptPath -Value $pythonScript
python $scriptPath
Remove-Item $scriptPath  # 清理临时文件