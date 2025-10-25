from pydub import AudioSegment
import util
import tool_ten_vad
import math
import tool_subt
import tool_loudness

logger = util.get_logger()


def diffusion(tags, tag):
    for i, _ in enumerate(tags):
        if i == 0:
            continue
        if tags[i - 1] == tag and tags[i] == 0:
            tags[i] = tag

    for i in range(len(tags) - 2, -1, -1):
        if tags[i] == 0 and tags[i + 1] == tag:
            tags[i] = tag

    return tags


def part_detect(audio_path,
                frame_rate=50,
                speech_threshold=0.8,
                silence_threshold=0.2,
                volume_threshold=-80,
                ):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    confidences = tool_ten_vad.vad_confidence(audio, frame_rate)
    if len(confidences) == 0:
        logger.error(f"人声置信度为空")
        raise ValueError(f"人声置信度为空")

    tags = []
    for i, confidence in enumerate(confidences):
        if confidence <= silence_threshold:
            tags.append(-1)
        elif speech_threshold <= confidence:
            tags.append(1)
        else:
            tags.append(0)

    volumes = tool_loudness.get_loudness(audio, frame_rate)
    if len(tags) != len(volumes):
        logger.error(f"人声置信度与响度长度不一致, tags: {len(tags)}, volumes: {len(volumes)}")
        raise ValueError(f"人声置信度与响度长度不一致, tags: {len(tags)}, volumes: {len(volumes)}")
    for i, tag in enumerate(tags):
        if tags[i] == 0 and volumes[i] <= volume_threshold:
            tags[i] = -1
    bbb(audio_path, tags)

    tags = diffusion(tags, 1)
    tags = diffusion(tags, -1)

    segments = []
    for i, tag in enumerate(tags):
        if tag == 1:
            vad_type = 'speech'
        elif tag == -1:
            vad_type = 'silence'
        else:
            logger.error(f"非法人声标签: {tag}")
            raise ValueError(f"非法人声标签: {tag}")
        if i > 0 and tags[i - 1] != tags[i]:
            pre_end = segments[-1]['end']
            segments.append({'start': pre_end, 'end': 0, 'vad_type': vad_type})
        if len(segments) == 0:
            segments.append({'start': 0, 'end': 0, 'vad_type': vad_type})
        end = math.floor((i + 1) * (1000.0 / frame_rate))
        segments[-1]['end'] = end

    if last_end < segments[-1]['end']:
        segments[-1]['end'] = last_end
    if segments[-1]['end'] < last_end:
        pre_end = segments[-1]['end']
        segments.append({'start': pre_end, 'end': last_end, 'vad_type': 'silence'})

    segments = tool_subt.init_segments(segments)
    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.unit_segments(segments, 'vad_type')
    tool_subt.check_coherent_segments(segments)

    return segments


import matplotlib.pyplot as plt
import numpy as np


def plot(tags, sy, ey):
    # 1. 模拟您的置信度数据
    # 这是一个示例数组，您可以替换为您的实际数据
    confidence_data = tags
    time_per_segment = 0.02  # 每个数据点代表的秒数

    # 2. 计算时间轴（X轴）
    # 时间点的数量等于数据点的数量
    num_segments = len(confidence_data)

    # 使用 numpy 生成时间点数组，从 0 开始，间隔 0.02 秒
    # 数组长度与 confidence_data 相同
    time_points = np.arange(0, num_segments * time_per_segment, time_per_segment)

    # 3. 创建图形
    plt.figure(figsize=(12, 5))

    # 4. 使用阶梯图（Step Plot）进行绘图
    # 'where="post"' 表示数值变化发生在时间间隔的末尾，
    # 这样能清晰地看到一个值持续了 0.02 秒。
    plt.step(time_points, confidence_data,
             where='post',
             label='Confidence Level',
             color='C0',
             linewidth=2)

    # 5. 添加辅助线和标签
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # 绘制 y=0 的辅助线

    # 优化 Y 轴刻度，只显示 -1, 0, 1
    plt.yticks([-1, 0, 1], ['-1 (低)', '0 (中)', '1 (高)'])
    plt.ylim(sy, ey)  # 稍微扩展 y 轴范围以获得更好的视觉效果
    plt.axhline(0.8, color='green', linestyle='-', linewidth=1, alpha=0.7, label='0.8')
    plt.axhline(0.5, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='0.5')
    plt.axhline(0.2, color='red', linestyle='-', linewidth=1, alpha=0.7, label='0.2')

    plt.title('置信度随时间的变化 (每段 0.02s)', fontsize=15)
    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('置信度', fontsize=12)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()  # 自动调整布局，避免标签重叠

    # 6. 显示图形
    plt.show()


def bbb(AUDIO_FILE, tags):
    # ----------------- 配置 -----------------
    CHUNK_MS = 20  # 0.02 秒 = 20 毫秒

    # ----------------- 步骤 1 & 2: 加载并处理音频数据 (与上一个脚本相同) -----------------
    try:
        print(f"正在加载音频文件: {AUDIO_FILE}...")
        audio = AudioSegment.from_file(AUDIO_FILE)
    except Exception as e:
        print(f"加载音频文件时出错: {e}")
        print("请确保你已安装 FFmpeg 并且文件路径正确。")
        exit()

    audio_duration_ms = len(audio)
    dbfs_values = []
    time_points_ms = []

    for i in range(0, audio_duration_ms, CHUNK_MS):
        chunk = audio[i:i + CHUNK_MS]
        if len(chunk) < 1:
            continue

        loudness_dbfs = chunk.dBFS
        dbfs_values.append(loudness_dbfs)
        time_points_ms.append(i)

    time_points_s = [t / 1000.0 for t in time_points_ms]
    num_chunks = len(dbfs_values)
    print(f"音频数据处理完成，共 {num_chunks} 个块。")

    # ----------------- 步骤 3: 模拟或加载你的 tags 数组 -----------------
    # ⚠️ 注意: 你的 tags 数组长度必须与 dbfs_values 数组长度相同！
    # 因为它们都代表每 0.02 秒的数据。

    # 以下是使用 numpy 模拟一个 tags 数组的示例，你应该用你自己的真实 tags 替换它
    np.random.seed(42)
    # 假设 tags 数组的长度与 dbfs_values 相同
    # tags = np.random.choice([-1, 0, 1], size=num_chunks, p=[0.1, 0.6, 0.3])
    print(f"Tags 数组加载/生成完成，长度为 {len(tags)}")

    if len(tags) != num_chunks:
        print("错误: Tags 数组长度与音频数据长度不匹配，无法进行对比绘图。")
        exit()

    # ----------------- 步骤 4: 使用双 Y 轴绘图 -----------------
    print("正在使用 matplotlib 双 Y 轴绘图...")
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # --- 绘制 左 Y 轴 (声音大小 - dBFS) ---
    color_dbfs = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Loudness (dBFS)', color=color_dbfs)
    # 绘制 dBFS 数据
    line1 = ax1.plot(time_points_s, dbfs_values, color=color_dbfs, label='Loudness (dBFS)')
    ax1.tick_params(axis='y', labelcolor=color_dbfs)
    ax1.grid(True, linestyle='--', axis='y')  # 仅在左侧Y轴上显示网格
    ax1.axhline(y=-20, color='b', linestyle=':', linewidth=1)  # 增加一个dBFS参考线

    # --- 绘制 右 Y 轴 (置信度 - Tags) ---
    # 实例化一个新的 Axes 对象，它与 ax1 共享 X 轴
    ax2 = ax1.twinx()
    color_tags = 'tab:red'
    ax2.set_ylabel('Confidence Tags', color=color_tags)
    # 绘制 tags 数据，使用 'o' 标记表示离散点
    line2 = ax2.plot(time_points_s, tags, color=color_tags, linestyle='', marker='.', markersize=4,
                     label='Confidence Tags')
    ax2.tick_params(axis='y', labelcolor=color_tags)

    # 设置右 Y 轴的刻度只显示 -1, 0, 1
    ax2.set_yticks([-1, 0, 1])

    # 限制右 Y 轴的范围，使离散点更清晰
    ax2.set_ylim(-1.5, 1.5)

    # --- 整合图表设置 ---
    plt.title(f'Audio Loudness vs. Confidence Tags (Chunk Size: {CHUNK_MS}ms)')

    # 合并图例，因为有两个 Axes 对象
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 自动调整布局以防止标签重叠
    fig.tight_layout()

    plt.show()
    print("双 Y 轴对比绘图完成。")


# part_detect('/workspace/script/speaker-diarization/material/test.wav')
