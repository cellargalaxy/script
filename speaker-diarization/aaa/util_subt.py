import json
from whisperx.utils import get_writer
import util
import pysubs2
import math
import copy
import util_vad

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


def unit_segments(segments, type_key):
    results = []
    for i, segment in enumerate(segments):
        key = ''
        if len(results) > 0:
            key = results[len(results) - 1][type_key]
        if segment[type_key] == key:
            results[len(results) - 1]['end'] = segment['end']
        else:
            results.append(segment)
    return results


def check_coherent_segments(segments):  # 连贯与离散  Coherent  discrete
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int):
            logger.error("检查segments，start类型非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int):
            logger.error("检查segments，end类型非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, segment:%s, %s", i, json.dumps(segment),
                         json.dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i > 0:
            pre_end = segments[i - 1]['end']
            if pre_end != start:
                logger.error("检查segments，pre_end与start非法: %s, segment:%s, %s", i, json.dumps(segment),
                             json.dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


def check_discrete_segments(segments):
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int) and not isinstance(start, float):
            logger.error("检查segments，start类型非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int) and not isinstance(end, float):
            logger.error("检查segments，end类型非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, segment:%s, %s", i, json.dumps(segment), json.dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, segment:%s, %s", i, json.dumps(segment),
                         json.dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i > 0:
            pre_end = segments[i - 1]['end']
            if start < pre_end:
                logger.error("检查segments，pre_end与start非法: %s, segment:%s, %s", i, json.dumps(segment),
                             json.dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


def gradual_segments(segments, gradual_duration_ms=500, audio_data=None):
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
            if audio_data:
                has_silene, min_probability, probability_ms = util_vad.has_silene_by_data(
                    audio_data[segments[i]['start']:segments[i]['end']])
                mean = min_probability
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
        if segment['start'] == segment['end']:
            continue
        gradual.append(segment)
    check_coherent_segments(gradual)
    return gradual


def shift_subt_time(subt, duration_ms):
    duration = duration_ms / 1000.0

    segments = subt.get('segments', [])
    for i, segment in enumerate(segments):
        segments[i]['start'] = segments[i]['start'] + duration
        segments[i]['end'] = segments[i]['end'] + duration
        words = segments[i].get('words', [])
        for j, word in enumerate(words):
            words[j]['start'] = words[j]['start'] + duration
            words[j]['end'] = words[j]['end'] + duration
        if len(words) > 0:
            segments[i]['words'] = words
    subt['segments'] = segments

    return subt


def shift_segments_time(segments, duration_ms):
    for i, segment in enumerate(segments):
        segments[i]['start'] = segments[i]['start'] + duration_ms
        segments[i]['end'] = segments[i]['end'] + duration_ms
    return segments


def subt2segments(subt):
    segments = subt.get('segments', [])
    for i, segment in enumerate(segments):
        segments[i].pop('words', None)
        segments[i]['start'] = round(segments[i]['start'] * 1000)
        if segments[i]['start'] < 0:
            segments[i]['start'] = 0
        segments[i]['end'] = round(segments[i]['end'] * 1000)
    segments = fix_overlap_segments(segments)
    return segments


def save_segments_as_srt(segments, save_path, skip_silene=False):
    results = []
    for i, segment in enumerate(segments):
        start = segment['start'] / 1000.0
        end = segment['end'] / 1000.0
        text = segment.get('text', '')
        if not text:
            text = f"[{segment.get('vad_type', '')}|{segment.get('speaker', '')}] {segment['start']}->{segment['end']}"
        if skip_silene and segment.get('vad_type', '') == 'silene':
            continue
        obj = {'start': start, 'end': end, 'text': text}
        results.append(obj)
    util.mkdir(save_path)
    subs = pysubs2.load_from_whisper(results)
    subs.save(save_path)


def save_subt_as_srt(subt, save_path):
    highlight_words = False
    segments = subt['segments']
    for i, segment in enumerate(segments):
        if segment.get('words', []):
            highlight_words = True
    save_dir = util.get_ancestor_dir(save_path)
    util.mkdir(save_dir)
    writer = get_writer("srt", save_dir)
    writer(
        subt,
        util.get_file_basename(save_path),
        {"max_line_width": None, "max_line_count": None, "highlight_words": highlight_words},
    )
