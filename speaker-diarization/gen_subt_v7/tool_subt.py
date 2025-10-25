import util
import tool_whisperx
import pysubs2
import math

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


def unit_segments(segments, type_key, type_value=None):
    results = []
    for i, segment in enumerate(segments):
        if type_value and segment[type_key] != type_value:
            results.append(segment)
            continue
        pre_type = ''
        if len(results) > 0:
            pre_type = results[-1].get(type_key, '')
        if segment[type_key] == pre_type:
            results[-1]['end'] = segment['end']
        else:
            results.append(segment)
    return results


def fill_segments(segments, last_end=None, vad_type=None):
    full = []
    for i, segment in enumerate(segments):
        pre_end = 0
        if i > 0:
            pre_end = segments[i - 1]['end']
        if pre_end < segments[i]['start']:
            obj = {"start": pre_end, "end": segments[i]['start']}
            if vad_type:
                obj['vad_type'] = vad_type
            full.append(obj)
        full.append(segments[i])
    if last_end and len(full) > 0 and full[-1]['end'] < last_end:
        obj = {"start": full[-1]['end'], "end": last_end}
        if vad_type:
            obj['vad_type'] = vad_type
        full.append(obj)
    return full


def clipp_segments(segments, last_end):
    clipp = []
    for i, segment in enumerate(segments):
        if last_end < segment['start']:
            continue
        if last_end < segment['end']:
            segment['end'] = last_end
        clipp.append(segment)
    return clipp


def check_coherent_segments(segments):
    """连贯"""
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int):
            logger.error("检查segments，start类型非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int):
            logger.error("检查segments，end类型非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i > 0:
            pre_end = segments[i - 1]['end']
            if pre_end != start:
                logger.error("检查segments，pre_end与start非法: %s, segment:%s, %s", i,
                             util.json_dumps(segment),
                             util.json_dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


def check_discrete_segments(segments):
    """离散"""
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int) and not isinstance(start, float):
            logger.error("检查segments，start类型非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int) and not isinstance(end, float):
            logger.error("检查segments，end类型非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, segment:%s, %s", i,
                         util.json_dumps(segment),
                         util.json_dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i > 0:
            pre_end = segments[i - 1]['end']
            if start < pre_end:
                logger.error("检查segments，pre_end与start非法: %s, segment:%s, %s", i,
                             util.json_dumps(segment),
                             util.json_dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


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


def save_segments_as_srt(segments, save_path, skip_silence=False):
    results = []
    for i, segment in enumerate(segments):
        index = i
        if segment.get('index', 0):
            index = segment.get('index', 0)
        start = segment['start'] / 1000.0
        end = segment['end'] / 1000.0
        text = f"{index:05d} {segment['start']}>{segment['end']}"
        if segment.get('vad_type', ''):
            text = f"{index:05d} {segment['start']}>{segment['end']} {segment.get('vad_type', '')}"
        if segment.get('speaker', ''):
            text = f"{index:05d} {segment['start']}>{segment['end']} {segment.get('speaker', '')}"
        if segment.get('text', ''):
            text = f"{index:05d} {segment['start']}>{segment['end']} {segment.get('text', '')}"
        if skip_silence and segment.get('vad_type', '') == 'silence':
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
    writer = tool_whisperx.get_writer("srt", save_dir)
    writer(
        subt,
        util.get_file_basename(save_path),
        {"max_line_width": None, "max_line_count": None, "highlight_words": highlight_words},
    )
