import util
import tool_subt
import os
import sys

logger = util.get_logger()


def split_segments(segments, index):
    left = []
    middle = []
    right = []
    for i, segment in enumerate(segments):
        if i < index:
            left.append(segment)
        elif i == index:
            middle.append(segment)
        else:
            right.append(segment)
    return left, middle, right


def sum_segments_duration(segments):
    duration = 0
    for i, segment in enumerate(segments):
        duration += segments[i]['duration']
    return duration


def group_segments(segments, min_speech_ms=1000 * 20):
    segments = util.deepcopy_obj(segments)
    segments = tool_subt.init_segments(segments)

    silences = []
    for i, segment in enumerate(segments):
        if i == 0 or i == len(segments) - 1:
            continue
        if segments[i]['vad_type'] != 'silence':
            continue
        silences.append(segments[i])
    silences = sorted(silences, key=lambda x: x['duration'], reverse=True)

    for i, silence in enumerate(silences):
        index = silence['index']
        left, middle, right = split_segments(segments, index)
        left_duration = sum_segments_duration(left)
        right_duration = sum_segments_duration(right)
        if min_speech_ms <= left_duration and min_speech_ms <= right_duration:
            left_groups = group_segments(left, min_speech_ms)
            right_groups = group_segments(right, min_speech_ms)
            groups = []
            groups.extend(left_groups)
            groups.append(middle)
            groups.extend(right_groups)
            return groups

    groups = []
    groups.append(segments)
    return groups


def part_divide(part_detect_path, output_dir, min_duration=200):
    json_path = os.path.join(output_dir, 'part_divide.json')
    srt_path = os.path.join(output_dir, 'part_divide.srt')
    if util.path_exist(json_path):
        return json_path

    segments = util.read_file_to_obj(part_detect_path)

    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] == 'silence' and segments[i]['duration'] < min_duration:
            segments[i]['vad_type'] = 'speech'
    segments = tool_subt.unit_segments(segments, 'vad_type')
    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] == 'speech' and segments[i]['duration'] < min_duration:
            segments[i]['vad_type'] = 'silence'
    segments = tool_subt.unit_segments(segments, 'vad_type')
    util.save_as_json(segments, os.path.join(output_dir, 'unit_mini.json'))
    tool_subt.save_segments_as_srt(segments, os.path.join(output_dir, 'unit_mini.srt'), skip_silence=True)

    groups = group_segments(segments)
    results = []
    for i, group in enumerate(groups):
        result = {"start": sys.maxsize, "end": 0, "vad_type": "silence"}
        for j, segment in enumerate(group):
            result['start'] = min(result['start'], segment['start'])
            result['end'] = max(result['end'], segment['end'])
            if segment['vad_type'] == 'speech':
                result['vad_type'] = 'speech'
        results.append(result)

    results = tool_subt.init_segments(results)
    results = tool_subt.fix_overlap_segments(results)
    results = tool_subt.unit_segments(results, 'vad_type')
    tool_subt.check_coherent_segments(results)

    util.save_as_json(results, json_path)
    tool_subt.save_segments_as_srt(results, srt_path)
    return json_path


def exec(manager):
    logger.info("part_divide,enter: %s", util.json_dumps(manager))
    part_detect_path = manager.get('part_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "part_divide")
    part_divide_path = part_divide(part_detect_path, output_dir)
    manager['part_divide_path'] = part_divide_path
    logger.info("part_divide,leave: %s", util.json_dumps(manager))
    util.exec_gc()
