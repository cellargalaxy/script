from pyannote.audio.pipelines.clustering import AgglomerativeClustering
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


audio_dir = '../gen_subt_v4/output/long_jpn/segment_split'
audio_names = util.listdir(audio_dir)
audio_names = sorted([s for s in audio_names if "speech.wav" in s])
print(audio_names)

for i, audio_name in enumerate(audio_names):
    if i + 10 > len(audio_names):
        break
    if audio_names[i] != '00062_speech.wav':
        continue
    group_names = audio_names[i:i + 10]
    group_paths = [os.path.join(audio_dir, name) for name in group_names]

    embedding_list = []
    for j, group_path in enumerate(group_paths):
        embedding = get_embedding(group_path, '')
        embedding = np.squeeze(embedding)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    clustering = AgglomerativeClustering().instantiate(
        {
            "method": "average",
            "min_cluster_size": 0,
            "threshold": 0.5,
        }
    )
    clusters = clustering.cluster(
        embeddings=embeddings, min_clusters=1, max_clusters=len(embedding_list)
    )
    print(clusters)
    for filename, label in zip(group_names, clusters):
        short_name = filename.split('/')[-1]
        print(f"File: {short_name}, Label: {label}")
