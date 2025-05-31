import os
import torchaudio
import torch
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

# 1. 加载模型
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp/spkrec"
)

# 2. 读取音频并提取嵌入
def get_embedding(file_path):
    signal, sr = torchaudio.load(file_path)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resample(signal)
    with torch.no_grad():
        emb = model.encode_batch(signal).squeeze().cpu().numpy()
    emb = emb.flatten()
    return emb

# 3. 批量处理所有 wav 文件
folder = "whisper_timestamped_spilt"
files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]

embeddings = np.array([get_embedding(f) for f in files])
print(embeddings.shape)

# 4. 层次聚类
distance_matrix = squareform(pdist(embeddings, metric="cosine"))
cluster = AgglomerativeClustering(
    affinity='precomputed', linkage='average', distance_threshold=0.4, n_clusters=None
)
labels = cluster.fit_predict(distance_matrix)

# 5. 输出结果
clusters = defaultdict(list)
for f, label in zip(files, labels):
    clusters[label].append(os.path.basename(f))

for label, group in clusters.items():
    print(f"Cluster {label}: {group}")
