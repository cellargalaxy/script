import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn'

import torch
import torchaudio

vad_model = torch.jit.load("HumAwareVAD.jit")
vad_model.eval()
device = next(vad_model.parameters()).device

waveform, sr = torchaudio.load(
    "/workspace/script/speaker-diarization/gen_subt_v8/output/BOW_AND_ARROW/extract_stem/noreverb.wav")
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
waveform = waveform.float()
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sr = 16000
waveform = waveform.to(device)

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
