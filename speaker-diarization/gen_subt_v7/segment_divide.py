import math
import util
import os
import tool_subt

logger = util.get_logger()


def match_silence(start, end, silences, max_ms=1000):
    timestamp = math.floor((start + end) / 2.0)
    silence = {'timestamp': -max_ms * 10}
    for i, segment in enumerate(silences):
        distance_i = abs(timestamp - segment['timestamp'])
        distance_j = abs(timestamp - silence['timestamp'])
        if distance_i <= distance_j:
            silence = segment
    if abs(timestamp - silence['timestamp']) <= max_ms:
        return silence
    return None


def segment_divide(part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)

    silences = []
    for i, segment in enumerate(parts):
        if segment['vad_type'] != 'silence':
            continue
        segment['timestamp'] = math.floor((segment['start'] + segment['end']) / 2.0)
        silences.append(segment)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        pre_end = segments[i - 1]['end']
        start = segments[i]['start']
        silence = match_silence(pre_end, start, silences)
        if silence:
            segments[i - 1]['end'] = silence['start']
            segments[i]['start'] = silence['end']

    segments = tool_subt.fix_overlap_segments(segments)
    tool_subt.check_discrete_segments(segments)

    for i, segment in enumerate(segments):
        segments[i]['index'] = i
        segments[i]['duration'] = segments[i]['end'] - segments[i]['start']

    util.save_as_json(segments, json_path)
    tool_subt.save_segments_as_srt(segments, srt_path)
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
