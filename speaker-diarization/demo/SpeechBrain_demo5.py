import os
from speechbrain.inference.speaker import SpeakerRecognition
import util
import json

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                               run_opts={"device": "cuda"})

confidences = []
audio_dir = '../aaa/output/long/segment_split'
files = util.listdir(audio_dir)
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        audio_path_i = os.path.join(audio_dir, files[i])
        audio_path_j = os.path.join(audio_dir, files[j])
        score, prediction = verification.verify_files(audio_path_i, audio_path_j)
        confidences.append(score.item())

util.save_file(json.dumps(confidences), 'SpeechBrain_demo5.json')

import matplotlib.pyplot as plt

plt.hist(confidences, bins=100, range=(0, 1), edgecolor='black')
plt.xlabel('Value range')
plt.ylabel('Count')
plt.title('SpeechBrain_demo5')
plt.show()

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
plt.title('SpeechBrain_demo5.py')
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()

means = [confidences[labels == i].mean() for i in range(2)]
trust_label = np.argmax(means)
print(f"可信类别标签是：{trust_label}")
