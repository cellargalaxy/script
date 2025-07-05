from ten_vad import TenVad
import numpy as np
from pydub import AudioSegment

audio = AudioSegment.from_wav('../demo_eng_single.wav')

raw_data = audio.raw_data
rate = audio.frame_rate
channels = audio.channels
sample_width = audio.sample_width

if sample_width == 1:
    dtype = np.uint8  # 8位WAV一般是无符号
elif sample_width == 2:
    dtype = np.int16
elif sample_width == 4:
    dtype = np.int32
else:
    raise ValueError("Unsupported sample width: {}".format(sample_width))

audio_np = np.frombuffer(raw_data, dtype=dtype)
if channels > 1:
    audio_np = audio_np.reshape((-1, channels))

hop_size = int(rate / 10)
threshold = 0.5
ten_vad_instance = TenVad(hop_size, threshold)  # Create a TenVad instance
num_frames = audio_np.shape[0] // hop_size  # 帧数
for i in range(num_frames):
    audio_data = audio_np[i * hop_size: (i + 1) * hop_size]
    out_probability, out_flag = ten_vad_instance.process(audio_data)
    print("[%d] %0.6f, %d" % (i, out_probability, out_flag))
