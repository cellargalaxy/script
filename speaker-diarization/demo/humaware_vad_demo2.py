import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn'

import torch
from pydub import AudioSegment
import numpy as np
import os
import util
from torch.hub import load_state_dict_from_url
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename="HumAwareVAD.jit")
vad_model = torch.jit.load(model_path)
vad_model.eval()

# ===== 1. 使用 pydub 读取音频 =====
audio = AudioSegment.from_file(
    "/workspace/script/speaker-diarization/gen_subt_v8/output/BOW_AND_ARROW/extract_stem/noreverb.wav"
)

# ===== 2. 转为单声道 =====
audio = audio.set_channels(1)

# ===== 3. 转为 16kHz =====
audio = audio.set_frame_rate(16000)
sr = 16000

# ===== 4. 转为 torch waveform =====
samples = np.array(audio.get_array_of_samples())

# pydub 默认是 int16，需要归一化到 [-1, 1]
waveform = torch.from_numpy(samples).float() / 32768.0

# shape: (1, num_samples)
waveform = waveform.unsqueeze(0)

device = util.get_device_type()
waveform = waveform.to(device)

# ===== 后续代码保持不变 =====
frame_size = 512
hop_size = 512

outputs = []

for i in range(0, waveform.shape[1] - frame_size + 1, hop_size):
    frame = waveform[:, i:i + frame_size]
    out = vad_model(frame, sr)
    outputs.append(out)

vad_output = torch.cat(outputs, dim=0)
vad_scores = vad_output.squeeze(1).detach().cpu().tolist()

print(vad_scores)
