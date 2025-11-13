import util
import os
import tool_subt
from pydub import AudioSegment
import part_detect_vad
import math

logger = util.get_logger()


def find_first_start(segments, start, end):
    for i, segment in enumerate(segments):
        start_overlap = max(start, segment['start'])
        end_overlap = min(end, segment['end'])
        if start_overlap <= end_overlap:
            return start_overlap
    return None


def find_first_end(segments, start, end):
    for i, segment in enumerate(segments):
        start_overlap = max(start, segment['start'])
        end_overlap = min(end, segment['end'])
        if start_overlap <= end_overlap:
            return end_overlap
    return None


def box_segments(segments, start, end):
    segments = util.deepcopy_obj(segments)
    result = []
    for i, segment in enumerate(segments):
        if start <= segment['end'] and segment['start'] <= end:
            result.append(segment)
    if len(result) > 0 and result[0]['start'] < start:
        result[0]['start'] = start
    if len(result) > 0 and end < result[-1]['end']:
        result[-1]['end'] = end
    tool_subt.check_discrete_segments(result)
    return result


def scrap_segments(segments, time):
    left = None
    middle = None
    right = None
    for i, segment in enumerate(segments):
        if segment['end'] < time:
            left = segment
        if segment['start'] <= time and time <= segment['end']:
            middle = segment
        if not right and time < segment['start']:
            right = segment
    return left, middle, right


def segment_divide(audio_path, part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)
    tool_subt.check_discrete_segments(segments)

    silences = []
    for i, part in enumerate(parts):
        if part['vad_type'] != 'silence':
            continue
        silences.append(part)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        left = (segments[i - 1]['end'] + segments[i - 1]['start']) / 2.0
        left = math.ceil(left)
        left = max(segments[i - 1]['end'] - 1000, left)
        right = (segments[i]['end'] + segments[i]['start']) / 2.0
        right = math.floor(right)
        right = min(segments[i]['start'] + 1000, right)
        gaps = box_segments(silences, left, right)
        left_gap, middle_gap, right_gap = scrap_segments(gaps, segments[i]['start'])
        middle = segments[i]['start']
        if middle_gap:
            middle = (middle_gap['end'] + middle_gap['start']) / 2.0
            middle = math.floor(middle)
        elif right_gap:
            middle = (right_gap['end'] + right_gap['start']) / 2.0
            middle = math.floor(middle)
        elif left_gap:
            middle = (left_gap['end'] + left_gap['start']) / 2.0
            middle = math.floor(middle)
        segments[i - 1]['end'] = middle
        segments[i]['start'] = middle

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.init_segments(segments)
    tool_subt.check_discrete_segments(segments)

    util.save_as_json(segments, json_path)
    tool_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def exec(manager):
    logger.info("segment_divide,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    part_detect_path = manager.get('part_detect_path')
    segment_detect_path = manager.get('segment_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    json_path = segment_divide(audio_path, part_detect_path, segment_detect_path, output_dir)
    manager['segment_divide_path'] = json_path
    logger.info("segment_divide,leave: %s", util.json_dumps(manager))
    util.exec_gc()
