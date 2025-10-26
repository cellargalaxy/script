import util
import os
import speaker_detect_speechbrain
from pydub import AudioSegment
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import math
import numpy as np
import tool_subt

logger = util.get_logger()


def speaker_detect(audio_path, segment_divide_path, output_dir):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    srt_path = os.path.join(output_dir, 'speaker_detect.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(segment_divide_path)
    embedding_list = []
    for i, segment in enumerate(segments):
        cut = audio[segment['start']:segment['end']]
        embedding = speaker_detect_speechbrain.extract_embedding(cut)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    clustering = AgglomerativeClustering().instantiate({"method": "average", "min_cluster_size": 0, "threshold": 0.5})
    max_clusters = math.ceil(len(embedding_list) / 2.0)
    max_clusters = max(max_clusters, 4)
    max_clusters = min(max_clusters, 64)
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=max_clusters)

    if len(segments) != len(clusters):
        logger.error(f"句子与说话人长度不一致, segments: {len(segments)}, clusters: {len(clusters)}")
        raise ValueError(f"句子与说话人长度不一致, segments: {len(segments)}, clusters: {len(clusters)}")

    for i, segment in enumerate(segments):
        segments[i]['speaker'] = clusters[i]

    util.save_as_json(segments, json_path)
    tool_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def exec(manager):
    logger.info("speaker_detect,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    segment_divide_path = manager.get('segment_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), 'speaker_detect')
    speaker_detect(audio_path, segment_divide_path, output_dir)
    logger.info("speaker_detect,leave: %s", util.json_dumps(manager))
    speaker_detect_speechbrain.exec_gc()
