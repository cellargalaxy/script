import speaker_divide_pyannote
import numpy as np
import os
import util
import json
from pydub import AudioSegment

logger = util.get_logger()


def get_threshold(confidences, percent):
    min_confidence = 1
    max_confidence = 0
    for item in confidences:
        if item['confidence'] < min_confidence:
            min_confidence = item['confidence']
        if max_confidence < item['confidence']:
            max_confidence = item['confidence']
    threshold = min_confidence + (max_confidence - min_confidence) * percent
    return threshold


def get_top_confidences(confidences, percent):
    threshold = get_threshold(confidences, percent)
    top_confidences = []
    for item in confidences:
        if threshold <= item['confidence']:
            top_confidences.append(item)
    return top_confidences


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def union_find(confidences):
    uf = UnionFind()
    for item in confidences:
        uf.union(item['path_i'], item['path_j'])
    groups = {}
    for item in confidences:
        for path in [item['path_i'], item['path_j']]:
            root = uf.find(path)
            groups.setdefault(root, set()).add(path)
    results = [list(paths) for paths in groups.values()]
    results = [sorted(sub) for sub in results]  # 第二维字符串升序排序
    results.sort(key=len, reverse=True)  # 第一维按照子数组数量降序排序
    return results


def best_union_find(confidences):
    best_results = []
    max_group_cnt = 0
    for percent in np.arange(1, 0, -0.01):
        percent = round(percent, 2)
        top_confidences = get_top_confidences(confidences, percent)
        results = union_find(top_confidences)
        logger.info("并查集求优, percent: %s, group_cnt: %s", percent, len(results))
        if max_group_cnt <= len(results):
            max_group_cnt = len(results)
            best_results = results
        else:
            break
    return best_results


def speaker_divide(speaker_detect_path, output_dir):
    speaker_divide_path = os.path.join(output_dir, 'speaker_divide.json')
    if util.path_exist(speaker_divide_path):
        return

    content = util.read_file(speaker_detect_path)
    confidences = json.loads(content)

    results = best_union_find(confidences)
    util.save_as_json(results, speaker_divide_path)

    split_dir = os.path.join(output_dir, 'split')
    util.delete_path(split_dir)
    for i, result in enumerate(results):
        for j, file_path in enumerate(result):
            cp_path = os.path.join(split_dir, f"speaker_{i}", util.get_file_basename(file_path))
            util.mkdir(cp_path)
            util.copy_file(file_path, cp_path)


def exec(manager):
    logger.info("speaker_divide,enter: %s", json.dumps(manager))
    speaker_detect_path = manager.get('speaker_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_divide")
    speaker_divide(speaker_detect_path, output_dir)
    manager['speaker_divide_path'] = output_dir
    logger.info("speaker_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
