import os
import numpy as np
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline
from pyannote.core.utils.generators import pairwise

# ==== 配置 ====
HUGGINGFACE_TOKEN = ""  # 替换为你的 Hugging Face token
AUDIO_FILE = "../short.wav"          # 替换为你的音频文件路径
THRESHOLD = 0.5                        # 语音判断的置信度阈值
TIME_RESOLUTION = 0.01                # 10ms 精度

# ==== 加载模型 ====
print("Loading model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=HUGGINGFACE_TOKEN
)

# ==== 分析音频 ====
print(f"Processing audio: {AUDIO_FILE}")
output = pipeline(AUDIO_FILE)
scores = output["scores"].resize(TIME_RESOLUTION)

# ==== 提取时间轴与概率 ====
times = [t for t, _ in scores]
probs = scores.data[:, 0]
speech_activity = probs > THRESHOLD

# ==== 可视化 ====
plt.figure(figsize=(12, 4))
plt.plot(times, probs, label="Speech probability", color="blue")
plt.fill_between(times, 0, 1, where=speech_activity, color="orange", alpha=0.3, label="Speech detected")
plt.axhline(y=THRESHOLD, color="red", linestyle="--", label=f"Threshold = {THRESHOLD}")
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.title("Voice Activity Detection (10ms resolution)")
plt.legend()
plt.tight_layout()
plt.savefig("vad_output.png")
plt.show()

# ==== 输出语音段落 ====
print("\nDetected speech regions:")
regions = scores.support(threshold=THRESHOLD)
for segment in regions:
    region_scores = scores.crop(segment)
    avg_score = np.mean(region_scores.data)
    print(f" - Speech from {segment.start:.2f}s to {segment.end:.2f}s (confidence={avg_score:.2f})")
