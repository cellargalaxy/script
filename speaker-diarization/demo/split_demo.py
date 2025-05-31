import os
import subprocess
import math

# 输入文件路径
input_file = '../long.mkv'
output_dir = 'split_demo'
os.makedirs(output_dir, exist_ok=True)

# 获取视频总时长（单位：秒）
def get_duration(filename):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
         'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

# 切割视频
def split_video(input_file, output_dir, window=200, overlap=20):
    duration = get_duration(input_file)
    step = window - overlap
    segments = []

    start = 0
    index = 1
    while start + window <= duration:
        output_path = os.path.join(output_dir, f'part_{index:03d}.mkv')
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(start), '-t', str(window),
            '-i', input_file, '-c', 'copy', output_path
        ])
        segments.append(output_path)
        start += step
        index += 1

    # 如果还有剩余时间且不足一个完整窗口，合并到上一个片段
    if start < duration:
        last_start = start - step
        last_duration = duration - last_start
        output_path = os.path.join(output_dir, f'part_{index-1:03d}_extended.mkv')
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(last_start), '-t', str(last_duration),
            '-i', input_file, '-c', 'copy', output_path
        ])
        os.remove(segments[-1])  # 删除旧的片段
        segments[-1] = output_path  # 替换为合并的新片段

    print(f"✅ 完成，共生成 {len(segments)} 个片段。")

# 执行分割
split_video(input_file, output_dir)
