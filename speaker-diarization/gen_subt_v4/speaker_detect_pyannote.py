import json
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import torch
import numpy as np
import util
from pyannote.audio import Model
from pyannote.audio import Inference

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


def detect_segments(segments, auth_token):
    audio_names = []
    audio_paths = []
    for i, segment in enumerate(segments):
        audio_names.append(segment['file_name'])
        audio_paths.append(segment['wav_path'])

    embedding_list = []
    for i, audio_path in enumerate(audio_paths):
        embedding = get_embedding(audio_path, auth_token)
        embedding = np.squeeze(embedding)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    clustering = AgglomerativeClustering().instantiate({"method": "average", "min_cluster_size": 0, "threshold": 0.5})
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=len(embedding_list))

    cluster_map = {}
    for file_name, cluster in zip(audio_names, clusters):
        group = cluster_map.get(cluster, [])
        group.append(file_name)
        cluster_map[cluster] = group
    groups = []
    for cluster in cluster_map:
        groups.append(cluster_map[cluster])
    return groups


def speaker_detect(segment_split_path, auth_token, min_speech_duration_ms=1000):
    content = util.read_file(segment_split_path)
    segments = json.loads(content)

    speech_segments = []
    for i, segment in enumerate(segments):
        if segment['vad_type'] != 'speech':
            continue
        if segment['end'] - segment['start'] < min_speech_duration_ms:
            continue
        speech_segments.append(segment)

    groups = []
    speech_segments_len = len(speech_segments)
    step = 5
    window_size = 10
    i = 0
    while i < speech_segments_len:
        current_end = i + window_size
        if speech_segments_len - i < 2 * window_size and speech_segments_len - i <= window_size + step:
            result = detect_segments(speech_segments[i:speech_segments_len], auth_token)
            groups.extend(result)
            break
        else:
            result = detect_segments(speech_segments[i:current_end], auth_token)
            groups.extend(result)
        i += step
    return groups
