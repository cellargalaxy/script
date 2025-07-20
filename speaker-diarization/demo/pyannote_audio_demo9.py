from pyannote.audio import Inference
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设你已经准备好 10 个音频路径
audio_files = [
    "../gen_subt_v4/output/long_jpn/segment_split/00001_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00002_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00003_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00004_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00005_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00006_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00007_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00008_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00009_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00010_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00011_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00013_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00014_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00015_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00016_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00017_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00018_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00019_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00020_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00021_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00023_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00024_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00025_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00026_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00028_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00029_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00030_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00031_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00032_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00033_speech.wav",
    "../gen_subt_v4/output/long_jpn/segment_split/00034_speech.wav",
]

# 1. 加载 pyannote 的 embedding 提取器（默认是 speaker embedding）
inference = Inference("pyannote/embedding", window="whole")

# 2. 提取每个音频的 embedding
embeddings = []
for path in audio_files:
    emb = inference(path)
    embeddings.append(emb)

# 转为 numpy 数组
X = np.vstack(embeddings)

# 3. 降维（选一种）
# 方法 A: 使用 PCA
# X_reduced = PCA(n_components=2).fit_transform(X)

# 方法 B: 使用 t-SNE
X_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)

# 4. 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=range(len(audio_files)), cmap='tab10', s=100)
for i, name in enumerate(audio_files):
    plt.text(X_reduced[i, 0], X_reduced[i, 1], f"{i + 1}", fontsize=12)
plt.title("Speaker Embedding Distribution (t-SNE)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.colorbar()
plt.show()
