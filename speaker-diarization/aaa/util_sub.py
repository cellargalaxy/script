import json
from whisperx.utils import get_writer
import util
import os
import pysubs2
import math
import copy

logger = util.get_logger()


def fix_overlap_segments(segments):
    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i - 1]['end'] <= segments[i]['start']:
            continue
        mean = math.floor((segments[i - 1]['end'] + segments[i]['start']) / 2.0)
        segments[i]['start'] = mean
        segments[i - 1]['end'] = mean
    return segments


def unit_segments(segments):
    list = []
    for i, segment in enumerate(segments):
        vad_type = ''
        if len(list) > 0:
            vad_type = list[len(list) - 1]['vad_type']
        if segment['vad_type'] == vad_type:
            list[len(list) - 1]['end'] = segment['end']
        else:
            list.append(segment)
    return list


def check_segments(segments):
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int):
            logger.error("检查segments，start类型非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int):
            logger.error("检查segments，end类型非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i != 0:
            pre_end = segments[i - 1]['end']
            if pre_end != start:
                logger.error("检查segments，pre_end与start非法: %s, %s", i, json.dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


def gradual_segments(segments, gradual_duration_ms=500):
    if len(segments) <= 1:
        return segments
    segments = copy.deepcopy(segments)
    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] != 'silene':
            continue
        duration = segments[i]['end'] - segments[i]['start']
        if i == 0:
            if duration < gradual_duration_ms * 2:
                segments[i]['end'] = 0
                segments[i + 1]['start'] = segments[i]['start']
            else:
                segments[i]['end'] = segments[i]['end'] - gradual_duration_ms
                segments[i + 1]['start'] = segments[i + 1]['start'] - gradual_duration_ms
            continue
        if i == len(segments) - 1:
            if duration < gradual_duration_ms * 2:
                segments[i - 1]['end'] = segments[i]['end']
                segments[i]['end'] = 0
            else:
                segments[i - 1]['end'] = segments[i - 1]['end'] + gradual_duration_ms
                segments[i]['start'] = segments[i]['start'] + gradual_duration_ms
            continue
        if duration < gradual_duration_ms * 4:
            mean = math.floor((segments[i]['end'] + segments[i]['start']) / 2.0)
            segments[i - 1]['end'] = mean
            segments[i + 1]['start'] = mean
            segments[i]['end'] = 0
        else:
            segments[i - 1]['end'] = segments[i - 1]['end'] + gradual_duration_ms
            segments[i]['start'] = segments[i]['start'] + gradual_duration_ms
            segments[i]['end'] = segments[i]['end'] - gradual_duration_ms
            segments[i + 1]['start'] = segments[i + 1]['start'] - gradual_duration_ms
    gradual = []
    for i, segment in enumerate(segments):
        if segment['end'] == 0:
            continue
        gradual.append(segment)
    check_segments(gradual)
    return gradual


def save_sub_as_vtt(audio_path, sub, save_dir=''):
    if not save_dir:
        save_dir = util.get_ancestor_dir(audio_path)
    util.mkdir(save_dir)
    vtt_writer = get_writer("vtt", save_dir)
    vtt_writer(
        sub,
        audio_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )


def save_sub_as_json(audio_path, sub, save_dir=''):
    if not save_dir:
        save_dir = util.get_ancestor_dir(audio_path)
    json_path = os.path.join(save_dir, util.get_file_name(audio_path) + '.json')
    util.save_file(json.dumps(sub), json_path)


def save_segments_as_srt(segments, file_path):
    results = []
    for i, segment in enumerate(segments):
        start = segment['start'] / 1000.0
        end = segment['end'] / 1000.0
        text = segment.get('text', '')
        if not text:
            text = f"[{segment.get('vad_type', '')}|{segment.get('speaker', '')}] {segment['start']}->{segment['end']}"
        obj = {'start': start, 'end': end, 'text': text}
        results.append(obj)
    util.mkdir(file_path)
    subs = pysubs2.load_from_whisper(results)
    subs.save(file_path)
