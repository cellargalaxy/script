import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from pydub import AudioSegment
import tempfile
import os
from audiosr import build_model, super_resolution

audio = AudioSegment.from_file("/workspace/script/speaker-diarization/material/006.wav")  # 或已在内存中的 AudioSegment

# 导出到临时文件
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
    audio.export(tmp_path, format="wav")

# 构建模型并运行
model = build_model(model_name="basic", device="cuda")  # 或 "cpu"
waveform = super_resolution(model, tmp_path, seed=42, guidance_scale=3.5, ddim_steps=50)

# 按仓库里做法保存
import numpy as np, soundfile as sf
out_wav = (waveform[0] * 32767).astype(np.int16).T
sf.write("out_from_pydub.wav", out_wav, samplerate=48000)

# 清理临时文件
os.remove(tmp_path)