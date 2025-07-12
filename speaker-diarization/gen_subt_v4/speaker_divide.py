import numpy as np
import os
import util
import json

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
    best_percent = 0
    best_confidences = []
    best_results = []
    max_group_cnt = 0
    for percent in np.arange(1, 0, -0.01):
        percent = round(percent, 2)
        top_confidences = get_top_confidences(confidences, percent)
        results = union_find(top_confidences)
        if max_group_cnt <= len(results):
            best_percent = percent
            max_group_cnt = len(results)
            best_results = results
            best_confidences = top_confidences
        else:
            break
    logger.info("并查集求优, percent: %s, group_cnt: %s", best_percent, len(best_results))
    return best_confidences, best_results


def speaker_divide(speaker_detect_dir, output_dir):
    speaker_divide_path = os.path.join(output_dir, 'speaker_divide.json')
    if util.path_exist(speaker_divide_path):
        return

    union_confidences = []
    files = util.listdir(speaker_detect_dir)
    for file in files:
        if not file.endswith('.json'):
            continue
        logger.info("并查集求优: %s", file)
        speaker_detect_path = os.path.join(speaker_detect_dir, file)
        content = util.read_file(speaker_detect_path)
        confidences = json.loads(content)
        confidences, _ = best_union_find(confidences)
        confidences = sorted(confidences, key=lambda x: x['confidence'], reverse=True)
        util.save_as_json(confidences, os.path.join(output_dir, file))
        union_confidences.extend(confidences)

    results = union_find(union_confidences)
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
    speaker_detect_dir = manager.get('speaker_detect_dir')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_divide")
    speaker_divide(speaker_detect_dir, output_dir)
    manager['speaker_divide_dir'] = output_dir
    logger.info("speaker_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
