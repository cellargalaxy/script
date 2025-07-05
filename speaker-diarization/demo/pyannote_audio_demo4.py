import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import torch
import numpy as np
import os
import util
from pyannote.audio import Model

model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.environ.get('auth_token', ''))

from scipy.spatial.distance import cdist
from pyannote.audio import Inference

inference = Inference(model, window="whole")
inference.to(torch.device("cuda"))


def cdist_distance(a, b):
    embedding1 = inference(a)
    embedding2 = inference(b)
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
    return distance


results = []
confidences = []
audio_dir = '../aaa/output/long/segment_split'
for file in util.listdir(audio_dir):
    if not file.endswith('speech.wav') or file.endswith('00068_speech.wav'):
        continue
    audio_path = os.path.join(audio_dir, file)
    score = cdist_distance('../aaa/output/long/segment_split/00068_speech.wav', audio_path)
    print(audio_path, 1 - score)
    confidences.append(1 - score)
    results.append({"path": audio_path, "score": 1 - score})

# print()
# arr_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
# top_n = max(1, int(len(arr_sorted) * 0.5))
# top_5_percent = arr_sorted[:top_n]
# for item in top_5_percent:
#     print(item)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

confidences = np.array(confidences)

# KMeans聚类（2类）
confidences_reshape = confidences.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(confidences_reshape)
labels = kmeans.labels_

# 将标签和原数据一起排序，便于可视化
sort_idx = np.argsort(confidences)
confidences_sorted = confidences[sort_idx]
labels_sorted = labels[sort_idx]

plt.figure(figsize=(8, 4))
for label in np.unique(labels_sorted):
    idx = labels_sorted == label
    plt.scatter(np.where(idx)[0], confidences_sorted[idx], label=f'Class {label}', s=100)
plt.plot(range(len(confidences_sorted)), confidences_sorted, color='gray', alpha=0.3, linestyle='--')
plt.xlabel('Index (sorted)')
plt.ylabel('Confidence')
plt.title('pyannote_audio_demo4.py')
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()

means = [confidences[labels == i].mean() for i in range(2)]
trust_label = np.argmax(means)
print(f"可信类别标签是：{trust_label}")
