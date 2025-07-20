from sklearn.cluster import AgglomerativeClustering
import torch
import numpy as np
import os
import util
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.audio import Inference

inference = None
embedding_map = {}


def get_inference(auth_token):
    global inference
    if inference:
        return inference
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token)
    inference = Inference(model, window="whole")
    inference.to(torch.device(util.get_device_type()))
    return inference


def get_embedding(path, auth_token):
    global embedding_map
    embedding = embedding_map.get(path, None)
    if embedding is not None:
        return embedding
    inference = get_inference(auth_token)
    embedding = inference(path)
    embedding = np.array(embedding).reshape(1, -1)
    embedding_map[path] = embedding
    return embedding


def confidence_detect(path_i, path_j, auth_token):
    embedding_i = get_embedding(path_i, auth_token)
    embedding_j = get_embedding(path_j, auth_token)
    distance = cdist(embedding_i, embedding_j, metric="cosine")[0, 0]
    confidence = 1 - distance
    return confidence


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

embedding_list = []
for i in range(len(audio_files)):
    embedding = get_embedding(audio_files[i], '')
    embedding = np.squeeze(embedding)  # 确保是 (D,)
    embedding_list.append(embedding)

from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

# 1. 假设你已有 pyannote 生成的 embedding 向量： (N, D)
embeddings = np.array(embedding_list)

# 2. 推荐使用 cosine 距离前先 L2 标准化
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 3. 用 UMAP 预降维（提高 HDBSCAN 的稳定性）
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
embedding_umap = reducer.fit_transform(embeddings)

# 4. HDBSCAN 聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=3,  # 可调，控制最小类规模
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)
cluster_labels = clusterer.fit_predict(embedding_umap)

# 5. 可获取每个点的聚类置信度
probabilities = clusterer.probabilities_

# cluster_labels == -1 的为未归类噪声
for label, prob in zip(cluster_labels, probabilities):
    print(f"label: {label}, confidence: {prob:.2f}")
