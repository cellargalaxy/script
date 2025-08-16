import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from faster_whisper import WhisperModel
from pydub import AudioSegment
import numpy as np

# 读音频
audio = AudioSegment.from_wav('../gen_subt_v6/output/00525_speech.wav')

# 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

# 创建模型
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# 用数组作为输入
segments, info = model.transcribe(samples)

# 打印结果
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
