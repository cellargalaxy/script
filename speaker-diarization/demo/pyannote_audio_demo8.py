import json
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


if not util.path_exist('pyannote_audio_demo8.json'):
    confidences = []
    audio_dir = '../aaa/output/long/segment_split'
    files = util.listdir(audio_dir)
    files = [s for s in files if "speech" in s]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            audio_path_i = os.path.join(audio_dir, files[i])
            audio_path_j = os.path.join(audio_dir, files[j])
            score = cdist_distance(audio_path_i, audio_path_j)
            confidences.append({"path_i": audio_path_i, "path_j": audio_path_j, "confidence": 1 - score})
    util.save_file(json.dumps(confidences), 'pyannote_audio_demo8.json')
else:
    content = util.read_file('pyannote_audio_demo8.json', '[]')
    confidences = json.loads(content)

def get_top_confidence(percent):
    min_confidence = 1
    max_confidence = 0
    for item in confidences:
        if item['confidence'] < min_confidence:
            min_confidence = item['confidence']
        if item['confidence'] > max_confidence:
            max_confidence = item['confidence']
    confidence = min_confidence + (max_confidence - min_confidence) * percent
    return confidence

def get_top_percent(percent):
    confidence = get_top_confidence(percent)
    results = []
    for item in confidences:
        if confidence <= item['confidence']:
            results.append(item)
    return results


# 并查集实现
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def get_uf_cnt(confidences):
    uf = UnionFind()
    for item in confidences:
        uf.union(item['path_i'], item['path_j'])
    groups = {}
    for item in confidences:
        for path in [item['path_i'], item['path_j']]:
            root = uf.find(path)
            groups.setdefault(root, set()).add(path)
    result = [list(paths) for paths in groups.values()]
    return result


max_top_percent = 0
max_uf_cnt = 0
for x in np.arange(1, 0, -0.01):
    percent = round(x, 2)
    top_percent = get_top_percent(percent)
    result = get_uf_cnt(top_percent)
    print("top_percent", percent, get_top_confidence(percent),len(result))
    if max_uf_cnt <= len(result):
        max_uf_cnt = len(result)
        max_top_percent = percent
    else:
        break

print()
top_percent = get_top_percent(max_top_percent)
result = get_uf_cnt(top_percent)
print("top_percent", max_top_percent, get_top_confidence(max_top_percent),len(result))
util.save_file(json.dumps(result), 'pyannote_audio_demo8_result.json')
