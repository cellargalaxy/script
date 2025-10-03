from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import torch
import numpy as np
import util
from pyannote.audio import Model
from pyannote.audio import Inference
import math

logger = util.get_logger()

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


def exec_gc():
    global inference
    inference = None
    global embedding_map
    embedding_map = None
    util.exec_gc()


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


def speaker_detect(speaks, auth_token):
    embedding_list = []
    for i, speak in enumerate(speaks):
        embedding = get_embedding(speak['wav_path'], auth_token)
        embedding = np.squeeze(embedding)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    # threshold小，聚类数量多，反之亦然
    clustering = AgglomerativeClustering().instantiate({"method": "average", "min_cluster_size": 0, "threshold": 0.5})
    max_clusters = math.ceil(len(embedding_list) / 2.0)
    max_clusters = max(max_clusters, 3)
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=max_clusters)

    cluster_map = {}
    for speak, cluster in zip(speaks, clusters):
        group = cluster_map.get(cluster, [])
        group.append(speak['file_name'])
        cluster_map[cluster] = group
    groups = []
    for cluster in cluster_map:
        groups.append(cluster_map[cluster])
    return groups
