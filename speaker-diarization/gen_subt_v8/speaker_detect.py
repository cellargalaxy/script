import util
import os
import speaker_detect_pyannote_wespeaker
from pydub import AudioSegment
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
import math
import numpy as np
import tool_subt
from collections import Counter

logger = util.get_logger()


def rank_arr(arr):
    """
    将输入整数数组中的元素按其出现频率进行降序排名，
    频率最高的数字分配为 0，次高为 1，以此类推。
    Args:
        arr: 待处理的整数列表。
    Returns:
        一个新的列表，其中原始值已被替换为它们的频率排名。
    """
    if not arr:
        return []

    # 1. 统计频率
    # Counter 得到一个字典 {数字: 出现次数}
    counts = Counter(arr)

    # 2. 确定排名
    # 将 items 转换为 (出现次数, 数字) 元组列表
    # 排序规则：
    # - 首先按出现次数 (x[1]) 降序排列 (使用 -x[1])
    # - 其次按数字本身 (x[0]) 升序排列 (保证频率相同时的稳定性)
    # sorted_items 示例: [(-3, 10), (-2, 5), (-1, 20)]
    sorted_items = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    # 3. 构建映射表 (旧值 -> 新排名)
    # 排名从 0 开始
    # rank_map 示例: {10: 0, 5: 1, 20: 2}
    rank_map = {}
    for rank, (value, count) in enumerate(sorted_items):
        rank_map[value] = rank

    # 4. 遍历原始数组，分配新值
    result = [rank_map[value] for value in arr]

    return result


def speaker_detect(audio_path, segment_divide_path, output_dir, min_duration=2000):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    srt_path = os.path.join(output_dir, 'speaker_detect.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(segment_divide_path)
    for i, segment in enumerate(segments):
        if segments[i]['duration'] < min_duration:
            segments[i]['speaker'] = 'other'
            continue

    embedding_list = []
    for i, segment in enumerate(segments):
        if segments[i].get('speaker', None) == 'other':
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        embedding = speaker_detect_pyannote_wespeaker.extract_embedding(cut)
        embedding_list.append(embedding)
    embeddings = np.array(embedding_list)

    # threshold越小簇越多
    clustering = AgglomerativeClustering().instantiate({"method": "average", "min_cluster_size": 0, "threshold": 0.5})
    max_clusters = math.ceil(len(embedding_list) / 2.0)
    max_clusters = max(max_clusters, 4)
    max_clusters = min(max_clusters, 32)
    clusters = clustering.cluster(embeddings=embeddings, min_clusters=1, max_clusters=max_clusters)
    speakers = []
    for i, cluster in enumerate(clusters):
        speakers.append(int(cluster))

    if len(embedding_list) != len(speakers):
        logger.error(f"句子与说话人长度不一致, embedding_list: {len(embedding_list)}, speakers: {len(speakers)}")
        raise ValueError(f"句子与说话人长度不一致, embedding_list: {len(embedding_list)}, speakers: {len(speakers)}")

    speakers = rank_arr(speakers)
    speakers_iterator = iter(speakers)
    for i, segment in enumerate(segments):
        if segments[i].get('speaker', None) == 'other':
            continue
        speakers = next(speakers_iterator)
        segments[i]['speaker'] = f'speaker{speakers:02d}'

    util.save_as_json(segments, json_path)
    tool_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def exec(manager):
    logger.info("speaker_detect,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    segment_divide_path = manager.get('segment_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), 'speaker_detect')
    json_path = speaker_detect(audio_path, segment_divide_path, output_dir)
    manager['speaker_detect_path'] = json_path
    logger.info("speaker_detect,leave: %s", util.json_dumps(manager))
    speaker_detect_pyannote_wespeaker.exec_gc()
