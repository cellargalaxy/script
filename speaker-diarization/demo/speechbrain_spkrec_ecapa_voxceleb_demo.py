import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import os
from sklearn.cluster import AgglomerativeClustering
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def extract_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    embedding = model.encode_batch(signal).squeeze().detach().cpu().numpy()
    return embedding


from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import numpy as np
import util
import math

logger = util.get_logger()

inference = None
embedding_map = {}


def speaker_detect(audio_dir, auth_token):
    files = util.listdir(audio_dir)
    embedding_list = []
    for i, file in enumerate(files):
        wav_path = os.path.join(audio_dir, file)
        embedding = extract_embedding(wav_path)
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
        output_path = os.path.join('output', 'speechbrain_spkrec_ecapa_voxceleb_demo', str(i), util.get_file_basename(wav_path))
        util.copy_file(wav_path, output_path)
