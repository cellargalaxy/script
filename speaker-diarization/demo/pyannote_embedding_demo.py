import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

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


def speaker_detect(audio_dir, auth_token):
    files = util.listdir(audio_dir)
    embedding_list = []
    for i, file in enumerate(files):
        wav_path = os.path.join(audio_dir, file)
        embedding = get_embedding(wav_path, auth_token)
        embedding = np.squeeze(embedding)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    clustering = AgglomerativeClustering().instantiate({"method": "average", "min_cluster_size": 0, "threshold": 0.5})
    max_clusters = math.ceil(len(embedding_list) / 2.0)
    max_clusters = max(max_clusters, 3)
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=max_clusters)

    cluster_map = {}
    for file, cluster in zip(files, clusters):
        wav_path = os.path.join(audio_dir, file)
        group = cluster_map.get(cluster, [])
        group.append(wav_path)
        cluster_map[cluster] = group
    groups = []
    for cluster in cluster_map:
        groups.append(cluster_map[cluster])
    groups = [sorted(inner) for inner in groups]
    groups = sorted(groups, key=lambda x: len(x), reverse=True)
    return groups


groups = speaker_detect('/workspace/script/speaker-diarization/gen_subt_v7/output/mkv/split_audio/segment_divide_path',
                        '')
for i, group in enumerate(groups):
    for j, wav_path in enumerate(group):
        output_path = os.path.join('output', 'pyannote_embedding', str(i), util.get_file_basename(wav_path))
        util.copy_file(wav_path, output_path)
