import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


# 1. 加载模型
model =  EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 2. 读取并提取嵌入
def extract_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    embedding = model.encode_batch(signal).squeeze().detach().cpu().numpy()
    return embedding

# 3. 遍历所有音频
audio_dir = '../aaa/output/long/segment_split'
wav_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('speech.wav')]
embeddings = [extract_embedding(f) for f in wav_files]

# 4. 聚类
X = np.vstack(embeddings)
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(X)  # 可调 threshold
labels = clustering.labels_

# 5. 输出每个文件对应的聚类标签
for file, label in zip(wav_files, labels):
    print(f"{file} -> Cluster {label}")
