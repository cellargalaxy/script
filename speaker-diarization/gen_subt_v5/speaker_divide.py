import numpy as np
import os
import util
import json
from collections import defaultdict
import util_subt

logger = util.get_logger()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x != self.parent.setdefault(x, x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def merge_groups(groups):
    uf = UnionFind()

    # 连接在同一个组中的元素
    for group in groups:
        for i in range(1, len(group)):
            uf.union(group[i], group[0])

    # 聚合所有属于同一个父节点的元素
    merged = defaultdict(list)
    for item in {x for group in groups for x in group}:
        root = uf.find(item)
        merged[root].append(item)

    result = [sorted(group) for group in merged.values()]
    result.sort(key=lambda x: len(x), reverse=True)
    return result


def speaker_divide(segment_split_path, speaker_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'speaker_divide.json')
    srt_path = os.path.join(output_dir, 'speaker_divide.srt')
    if util.path_exist(json_path):
        return json_path

    content = util.read_file(speaker_detect_path)
    groups = json.loads(content)
    groups = merge_groups(groups)
    cluster_map = {}
    for i, group in enumerate(groups):
        for j, file_name in enumerate(group):
            cluster_map[file_name] = f"speaker_{i:02d}"

    content = util.read_file(segment_split_path)
    segments = json.loads(content)
    for i, segment in enumerate(segments):
        speaker = cluster_map.get(segments[i]['file_name'], 'unknown')
        segments[i]['speaker'] = speaker

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silence=True)
    return json_path


def exec(manager):
    logger.info("speaker_divide,enter: %s", json.dumps(manager))
    segment_split_path = manager.get('segment_split_path')
    speaker_detect_path = manager.get('speaker_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_divide")
    speaker_divide_path = speaker_divide(segment_split_path, speaker_detect_path, output_dir)
    manager['speaker_divide_path'] = speaker_divide_path
    logger.info("speaker_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
