import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. 模型初始化
# -----------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
model.eval()

# -----------------------------
# 2. 读取音频
# -----------------------------
def load_wav(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze().numpy()

# -----------------------------
# 3. 计算 embedding（逐个文件）
# -----------------------------
def get_embedding(path):
    audio = load_wav(path)
    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    emb = F.normalize(emb, dim=-1).cpu().numpy().squeeze()
    return emb

# -----------------------------
# 4. 聚类
# -----------------------------
def speaker_clustering(wav_paths, threshold=0.5):
    embedding_list = [get_embedding(p) for p in wav_paths]
    embeddings = np.stack(embedding_list)

    clustering = AgglomerativeClustering().instantiate({
        "method": "average",
        "min_cluster_size": 0,
        "threshold": threshold
    })

    # 最大聚类数设置为音频数量的一半（至少 3 个簇）
    max_clusters = max(math.ceil(len(wav_paths) / 2), 3)
    print("max_clusters",max_clusters)
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=max_clusters)

    cluster_map = {}
    for path, cluster in zip(wav_paths, clusters):
        cluster_map.setdefault(cluster, []).append(os.path.basename(path))

    # 转成列表形式
    groups = list(cluster_map.values())
    return groups

# -----------------------------
# 5. 测试
# -----------------------------
wav_dir = "../material/shigeju/wav"
wav_paths = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]

groups = speaker_clustering(wav_paths, threshold=0.05)
print("聚类结果：")
for i, g in enumerate(groups):
    print(f"Cluster {i+1},{len(g)}: {g}")
