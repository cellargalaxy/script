import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch.nn.functional as F

# 1. 模型和特征提取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")

# 2. 读取 wav 文件
def load_wav(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:  # 需要重采样到 16k
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

# 假设你有多个 wav 文件
wav_dir = "/home/dt/data/image_vol/code/script/speaker-diarization/material/shigeju/wav"
wav_paths = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]

# 3. 提取特征并计算 embedding
audio_arrays = [load_wav(p) for p in wav_paths]
inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt", sampling_rate=16000)
with torch.no_grad():
    embeddings = model(**inputs).embeddings
embeddings = F.normalize(embeddings, dim=-1).cpu()

# 4. 计算相似度矩阵
cosine_sim = torch.mm(embeddings, embeddings.T)  # N x N 矩阵
print("相似度矩阵：")
print(cosine_sim)

# 5. 例子：判断前两个文件是不是同一个人
similarity = cosine_sim[0, 1].item()
threshold = 0.86  # 阈值可调
if similarity < threshold:
    print(f"{wav_paths[0]} 和 {wav_paths[1]} 不是同一个人 (sim={similarity:.2f})")
else:
    print(f"{wav_paths[0]} 和 {wav_paths[1]} 是同一个人 (sim={similarity:.2f})")
