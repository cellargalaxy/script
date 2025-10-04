import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 模型和特征提取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
model.eval()

# 2. 读取 wav 文件
def load_wav(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze().numpy()

# 3. 音频路径
wav_dir = "../material/shigeju/wav"
wav_paths = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]

# 4. 逐个计算 embedding
embeddings_list = []
for path in wav_paths:
    audio = load_wav(path)
    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=16000)
    # 迁移到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    # 归一化并搬回 CPU
    emb = F.normalize(emb, dim=-1).cpu()
    embeddings_list.append(emb)

# 5. 合并成 N x D 矩阵
embeddings = torch.vstack(embeddings_list)

# 6. 计算相似度矩阵
cosine_sim = torch.mm(embeddings, embeddings.T)
print("相似度矩阵：")
print(cosine_sim)
