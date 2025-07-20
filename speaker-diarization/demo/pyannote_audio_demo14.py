from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import torch
import numpy as np
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
]

embedding_list = []
for i in range(len(audio_files)):
    embedding = get_embedding(audio_files[i], '')
    embedding = np.squeeze(embedding)  # 确保是 (D,)
    embedding_list.append(embedding)

embeddings = np.array(embedding_list)

clustering = AgglomerativeClustering().instantiate(
    {
        "method": "average",
        "min_cluster_size": 0,
        "threshold": 0.5,
    }
)

# request 2 clusters
clusters = clustering.cluster(
    embeddings=embeddings, min_clusters=2, max_clusters=len(audio_files)
)
print(clusters)
for filename, label in zip(audio_files, clusters):
    short_name = filename.split('/')[-1]
    print(f"File: {short_name}, Label: {label}")
