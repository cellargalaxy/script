import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import torch
import nemo.collections.asr as nemo_asr

vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
    model_name="nvidia/frame_vad_multilingual_marblenet_v2.0")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad_model = vad_model.to(device)
vad_model.eval()

import librosa

input_signal = librosa.load("../demo_jpn_single.wav", sr=16000, mono=True)[0]
input_signal = torch.tensor(input_signal).unsqueeze(0).float()
input_signal_length = torch.tensor([input_signal.shape[1]]).long()

# Perform inference
with torch.no_grad():
    torch_outputs = vad_model(
        input_signal=input_signal.to(device),
        input_signal_length=input_signal_length.to(device),
    ).cpu()

import torch.nn.functional as F

logits = torch_outputs[0]

probs_all = F.softmax(logits, dim=-1)

# 提取每一帧为“语音”的概率（即索引1）
speech_probs = probs_all[:, 1].cpu().numpy()

print("前10帧为语音的概率：", speech_probs[:10])
