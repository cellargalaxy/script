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


def find_max_overlap_segment(segment: dict, segments: list[dict]) -> dict | None:
    """
    在 segments 列表中寻找与 segment 重叠最大的时间段。
    Args:
        segment: 参考时间段，格式为 {"start": int, "end": int}。
        segments: 待搜索的时间段列表。
    Returns:
        重叠最大的时间段字典，如果没有重叠的则返回 None。
    """
    max_overlap = 0
    max_overlap_segment = None
    for current_segment in segments:
        overlap = calculate_overlap(segment, current_segment)
        if max_overlap < overlap:
            max_overlap = overlap
            max_overlap_segment = current_segment
    if 0 < max_overlap:
        return max_overlap_segment
    else:
        return None


def segment_divide(part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)

    for i, part in enumerate(parts):
        if parts[i]['vad_type'] != 'speech':
            continue
        segment = find_max_overlap_segment(parts[i], segments)
        if segment:
            parts[i]['segment_index'] = segment['index']
            parts[i]['text'] = segment['text']
        else:
            parts[i]['vad_type'] = 'silence'
            parts[i]['text'] = ''
    parts = tool_subt.unit_segments(parts, 'vad_type')
    for i, part in enumerate(parts):
        parts[i]['segment_index'] = -(i + 1)
        if i == 0 or i == len(parts) - 1:
            continue
        if parts[i]['vad_type'] != 'silence':
            continue
        if parts[i - 1]['segment_index'] == parts[i + 1]['segment_index']:
            parts[i]['segment_index'] = parts[i - 1]['segment_index']
    parts = tool_subt.unit_segments(parts, 'segment_index')

    for i, part in enumerate(parts):
        parts[i].pop("vad_type")
        parts[i].pop("segment_index")

    parts = tool_subt.init_segments(parts)
    parts = tool_subt.fix_overlap_segments(parts)
    tool_subt.check_discrete_segments(parts)

    util.save_as_json(parts, json_path)
    tool_subt.save_segments_as_srt(parts, srt_path)
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
