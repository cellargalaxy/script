import os
import tool_subt
import math
import util
from pydub import AudioSegment

logger = util.get_logger()


def box_segments(segments, start, end):
    segments = util.deepcopy_obj(segments)
    result = []
    for i, segment in enumerate(segments):
        if start <= segment['end'] and segment['start'] <= end:
            result.append(segment)
    tool_subt.check_discrete_segments(result)
    if len(result) > 0 and result[0]['start'] < start:
        result[0]['start'] = start
    if len(result) > 0 and end < result[-1]['end']:
        result[-1]['end'] = end + 1
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

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)
    segments = tool_subt.fill_segments(segments, last_end=last_end, vad_type='silene')
    tool_subt.check_discrete_segments(segments)

    silences = []
    for i, part in enumerate(parts):
        if part['vad_type'] != 'silence':
            continue
        silences.append(part)

    for i, segment in enumerate(segments):
        segments[i]['segment_divide_type'] = 'default'
        if i == 0:
            continue
        left_side = (segments[i - 1]['end'] + segments[i - 1]['start']) / 2.0
        left_side = math.ceil(left_side)
        left_side = max(segments[i - 1]['end'] - 1000, left_side)
        right_side = (segments[i]['end'] + segments[i]['start']) / 2.0
        right_side = math.floor(right_side)
        right_side = min(segments[i]['start'] + 1000, right_side)
        gaps = box_segments(silences, left_side, right_side)
        left_gap, middle_gap, right_gap = scrap_segments(gaps, segments[i]['start'])
        left_cut = -1
        if left_gap:
            left_cut = (left_gap['end'] + left_gap['start']) / 2.0
            left_cut = math.floor(left_cut)
        middle_cut = -1
        if middle_gap:
            middle_cut = (middle_gap['end'] + middle_gap['start']) / 2.0
            middle_cut = math.floor(middle_cut)
        right_cut = -1
        if right_gap:
            right_cut = (right_gap['end'] + right_gap['start']) / 2.0
            right_cut = math.floor(right_cut)
        if middle_cut >= 0:
            segments[i]['segment_divide_type'] = 'middle'
            segments[i - 1]['end'] = middle_cut
            segments[i]['start'] = middle_cut
        elif left_cut >= 0 and right_cut < 0:
            segments[i]['segment_divide_type'] = 'left_only'
            segments[i - 1]['end'] = left_cut
            segments[i]['start'] = left_cut
        elif right_cut >= 0 and left_cut < 0:
            segments[i]['segment_divide_type'] = 'right_only'
            segments[i - 1]['end'] = right_cut
            segments[i]['start'] = right_cut
        elif abs(left_cut - segments[i]['start']) < abs(right_cut - segments[i]['start']):
            segments[i]['segment_divide_type'] = 'left_near'
            segments[i - 1]['end'] = left_cut
            segments[i]['start'] = left_cut
        elif abs(right_cut - segments[i]['start']) < abs(left_cut - segments[i]['start']):
            segments[i]['segment_divide_type'] = 'right_near'
            segments[i - 1]['end'] = right_cut
            segments[i]['start'] = right_cut

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
