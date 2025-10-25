import math

from requests import delete

import util
import os
import tool_subt

logger = util.get_logger()


def calculate_overlap(seg1: dict, seg2: dict) -> int:
    """
    计算两个时间段的重叠长度。
    Args:
        seg1: 格式为 {"start": int, "end": int} 的字典。
        seg2: 格式为 {"start": int, "end": int} 的字典。
    Returns:
        重叠部分的长度，如果没有重叠则返回 0。
    """
    overlap_start = max(seg1["start"], seg2["start"])
    overlap_end = min(seg1["end"], seg2["end"])
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    else:
        return 0


def find_overlap_segment(segment: dict, segments: list[dict]) -> dict | None:
    """
    在 segments 列表中寻找与 segment 重叠最大的时间段。
    只有当重叠长度超过 segment 自身长度的一半时，才认为有效。
    Args:
        segment: 参考时间段，格式为 {"start": int, "end": int, "duration": int}。
        segments: 待搜索的时间段列表。
    Returns:
        重叠最大的时间段字典，如果没有有效重叠的则返回 None。
    """
    overlap_threshold = segment["duration"] / 2
    max_overlap = 0
    max_overlap_segment = None
    for current_segment in segments:
        overlap = calculate_overlap(segment, current_segment)
        if max_overlap < overlap and overlap_threshold < overlap:
            max_overlap = overlap
            max_overlap_segment = current_segment
    return max_overlap_segment


def segment_divide(part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)

    speechs = []
    for i, part in enumerate(parts):
        if parts[i]['vad_type'] != 'speech':
            continue
        segment = find_overlap_segment(parts[i], segments)
        if segment:
            parts[i]['segment_index'] = segment['index']
            parts[i]['text'] = segment['text']
            speechs.append(parts[i])
    speechs = tool_subt.unit_segments(speechs, 'segment_index')
    for i, speech in enumerate(speechs):
        speechs[i].pop('segment_index')

    speechs = tool_subt.init_segments(speechs)
    speechs = tool_subt.fix_overlap_segments(speechs)
    tool_subt.check_discrete_segments(speechs)

    util.save_as_json(speechs, json_path)
    tool_subt.save_segments_as_srt(speechs, srt_path)
    return json_path


def exec(manager):
    logger.info("segment_divide,enter: %s", util.json_dumps(manager))
    part_detect_path = manager.get('part_detect_path')
    segment_detect_path = manager.get('segment_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    json_path = segment_divide(part_detect_path, segment_detect_path, output_dir)
    manager['segment_divide_path'] = json_path
    logger.info("segment_divide,leave: %s", util.json_dumps(manager))
    util.exec_gc()
