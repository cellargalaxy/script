from pyannote.audio import Pipeline
import numpy as np
import matplotlib.pyplot as plt

# 设置音频路径和 Hugging Face Token
AUDIO_FILE = "../gen_sub/output/demo/noise_reduction/htdemucs/wav/vocals.wav"  # 替换为你的音频路径
HUGGINGFACE_TOKEN = ""  # 替换为你的 Hugging Face token

# 加载 Voice Activity Detection pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=HUGGINGFACE_TOKEN
)

# 使用 pipeline 的 get_speech_probabilities 方法获取置信度序列
vad_scores = pipeline.get_speech_probabilities(AUDIO_FILE)

# 提取时间戳和对应置信度
timestamps = np.array([t for t, _ in vad_scores])
scores = np.array([s for _, s in vad_scores])

# 静音检测参数
threshold = 0.5
min_silence_duration = 0.3  # 单位：秒

# 获取语音掩码
speech_mask = scores > threshold

# 找到变化点
change_points = np.where(np.diff(speech_mask.astype(int)) != 0)[0] + 1
segments = []

# 考虑开头结尾
if speech_mask[0] == 0:
    change_points = np.insert(change_points, 0, 0)
if speech_mask[-1] == 0:
    change_points = np.append(change_points, len(speech_mask))

# 提取静音段
for i in range(0, len(change_points), 2):
    start = timestamps[change_points[i]]
    end = timestamps[change_points[i + 1]]
    if end - start >= min_silence_duration:
        segments.append((start, end))

# 输出静音段
print("Detected silent segments:")
for s, e in segments:
    print(f"Start: {s:.2f}s, End: {e:.2f}s, Duration: {e - s:.2f}s")

# 绘图
plt.figure(figsize=(14, 5))
plt.plot(timestamps, scores, label="Speech Probability", color="blue")
plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
for s, e in segments:
    plt.axvspan(s, e, color='red', alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Speech Probability")
plt.title("Voice Activity Detection with Silent Segments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
