from ten_vad import TenVad
from pydub import AudioSegment
import numpy as np


def pydub2TenVad(audio):
    raw_data = audio.raw_data
    simple_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("pydub转TenVad，非法sample_width: {}".format(sample_width))
    audio_np = np.frombuffer(raw_data, dtype=dtype)
    if channels > 1:
        audio_np = audio_np.reshape((-1, channels))
    return simple_rate, audio_np


data = AudioSegment.from_wav('/workspace/script/speaker-diarization/gen_subt_v7/output/mao/extract_simple/wav.wav')
frame_rate = 50
simple_rate, data = pydub2TenVad(data)
frame_simple = int(simple_rate / frame_rate)
frame_cnt = data.shape[0] // frame_simple
ten_vad_instance = TenVad(frame_simple)
vad_confidence = []
for i in range(frame_cnt):
    audio_data = data[i * frame_simple: (i + 1) * frame_simple]
    probability, _ = ten_vad_instance.process(audio_data)
    vad_confidence.append(probability)
print('len(vad_confidence)', len(vad_confidence))

from scipy.signal import medfilt


def median_filter_smooth(confidence_scores, kernel_size):
    """
    使用中值滤波对置信度分数进行平滑。

    :param confidence_scores: VAD 置信度数组 (numpy array)
    :param kernel_size: 滤波核大小（必须是奇数）
    :return: 平滑后的置信度数组
    """
    # 中值滤波要求 kernel_size 必须是奇数
    smoothed_scores = medfilt(confidence_scores, kernel_size=kernel_size)

    return smoothed_scores

kernel_size = 11
vad_confidence = median_filter_smooth(vad_confidence, kernel_size)

from hmmlearn import hmm

conf = np.array(vad_confidence).reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=0)
model.fit(conf)

states = model.predict(conf)

# 让平均置信度高的那类标为“有人声”
if np.mean(conf[states == 0]) > np.mean(conf[states == 1]):
    states = 1 - states

labels = states


def smooth_labels(labels, window=5):
    smoothed = np.convolve(labels, np.ones(window) / window, mode='same')
    return (smoothed > 0.5).astype(int)


labels = smooth_labels(labels, window=10)  # 约200ms平滑
print('len(labels)', len(labels))

import matplotlib.pyplot as plt
import numpy as np

# 示例数据（请替换为你自己的）

# 时间轴（单位：秒）
t = np.arange(len(vad_confidence)) * 0.02

plt.figure(figsize=(12, 5))

# 1️⃣ 置信度曲线
plt.plot(t, vad_confidence, label='VAD Confidence', color='blue', linewidth=1.5)

# 2️⃣ 聚类标签，画成阶梯图以表示状态
# plt.step(t, labels, where='post', label='Detected Speech (1=Speech)', color='red', linewidth=1.5, alpha=0.7)

# 3️⃣ 可选：在背景中填充语音区域
# plt.fill_between(t, 0, 1, where=labels > 0, color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform())

plt.xlabel("Time (s)")
plt.ylabel("Confidence / Label")
plt.title("VAD Confidence vs Speech/Non-speech Labels")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
