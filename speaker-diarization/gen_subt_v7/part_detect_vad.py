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


data = AudioSegment.from_wav('/workspace/script/speaker-diarization/gen_subt_v7/output/mao/separate_stem/noreverb.wav')
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

import numpy as np
from hmmlearn import hmm

conf = np.array(vad_confidence).reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=0)
model.fit(conf)

states = model.predict(conf)

# 让平均置信度高的那类标为“有人声”
if np.mean(conf[states == 0]) > np.mean(conf[states == 1]):
    states = 1 - states

labels = states
