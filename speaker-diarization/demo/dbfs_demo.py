import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

# ----------------- 配置 -----------------
AUDIO_FILE ='/workspace/script/speaker-diarization/material/test.wav'  # 替换成你的音频文件路径
CHUNK_MS = 20  # 0.02 秒 = 20 毫秒

# ----------------- 步骤 1: 加载音频文件 -----------------
try:
    print(f"正在加载音频文件: {AUDIO_FILE}...")
    # pydub 可以自动识别文件格式
    audio = AudioSegment.from_file(AUDIO_FILE)
    print("加载成功。")
except Exception as e:
    print(f"加载音频文件时出错: {e}")
    print("请确保你已安装 FFmpeg 并且文件路径正确。")
    exit()

# ----------------- 步骤 2: 分割并计算每个块的 dBFS -----------------
# pydub 测量的是 dBFS (decibels relative to full scale)，即相对于数字系统最大音量的分贝值。
# 0 dBFS 是最大值，所有的值都是负数。
print(f"正在以 {CHUNK_MS} 毫秒的块大小计算声音大小...")
audio_duration_ms = len(audio)
dbfs_values = []
time_points_ms = []

# 遍历音频块
for i in range(0, audio_duration_ms, CHUNK_MS):
    # 截取音频片段
    chunk = audio[i:i + CHUNK_MS]

    # 检查片段是否为空（音频末尾可能不足一个完整块）
    if len(chunk) < 1:
        continue

    # 计算当前块的音量 (dBFS)
    # .dBFS 属性返回 AudioSegment 的平均响度
    loudness_dbfs = chunk.dBFS

    # 记录时间和音量
    dbfs_values.append(loudness_dbfs)
    time_points_ms.append(i)

# 将时间点从毫秒转换为秒
time_points_s = [t / 1000.0 for t in time_points_ms]

print(f"共计算了 {len(dbfs_values)} 个时间点上的声音大小。")

# ----------------- 步骤 3: 绘图 -----------------
print("正在使用 matplotlib 绘图...")
plt.figure(figsize=(15, 6))  # 设置图表大小

# 绘制数据
plt.plot(time_points_s, dbfs_values, color='skyblue')

# 添加标题和标签
plt.title(f'Audio Loudness (dBFS) per {CHUNK_MS}ms Chunk')
plt.xlabel('Time (seconds)')
plt.ylabel('Loudness (dBFS)')

# 添加参考线
plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='0 dBFS (Max Loudness)')
# 通常-60 dBFS被认为是无声或极静的参考
plt.axhline(y=-60, color='gray', linestyle=':', linewidth=0.5, label='Typical Noise Floor')

# 设置Y轴刻度，使图表更易读
plt.yticks(range(-100, 1, 10))
plt.grid(True, linestyle='--')
plt.legend()

# 显示图表
plt.show()
print("绘图完成。")