import os
import util
import json
from collections import defaultdict

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


def speaker_divide(speak_path, speaker_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'speaker_divide.json')
    if util.path_exist(json_path):
        return json_path

    groups = util.read_file_to_obj(speaker_detect_path)
    groups = merge_groups(groups)
    cluster_map = {}
    for i, group in enumerate(groups):
        for j, file_name in enumerate(group):
            speaker = f"speaker{i:03d}"
            if len(group) <= 1:
                speaker = 'unknown'
            cluster_map[file_name] = speaker

    speaks = util.read_file_to_obj(speak_path)
    for i, speak in enumerate(speaks):
        speaker = cluster_map[speaks[i]['file_name']]
        speaks[i]['speaker'] = speaker

    segment_map = {}
    for i, speak in enumerate(speaks):
        segments = segment_map.get(speak['speaker'], [])
        segments.extend(speak['segments'])
        segment_map[speak['speaker']] = segments

    speaks = []
    for speaker, segments in segment_map.items():
        segments = sorted(segments, key=lambda x: x['file_name'])
        speak = {
            'file_name': speaker,
            'segments': segments,
        }
        speaks.append(speak)
    speaks = sorted(speaks, key=lambda x: x['file_name'])

    util.save_as_json(speaks, json_path)
    return json_path
