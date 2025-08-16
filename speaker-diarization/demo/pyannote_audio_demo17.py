import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from pydub import AudioSegment
import numpy as np
import torch
from pyannote.audio import Pipeline

# 加载音频
audio = AudioSegment.from_wav("../gen_subt_v6/output/long/segment_split/split/00525_speech.wav")

# 转成 numpy.int16 -> float32
samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

print('audio.channels', audio.channels)

# 如果是立体声，转单声道（取平均）
if audio.channels > 1:
    samples = samples.reshape((-1, audio.channels))
    samples = samples.mean(axis=1)

# 归一化到 -1~1
samples /= np.iinfo(audio.array_type).max

# 转成 torch.Tensor，形状 (channels, time)
waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # 单声道

# 加载 pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# 调用 pipeline
diarization = pipeline({"waveform": waveform, "sample_rate": audio.frame_rate})

# 输出结果
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s -> {turn.end:.1f}s: {speaker}")
