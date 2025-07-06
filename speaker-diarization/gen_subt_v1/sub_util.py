import json
from whisperx.utils import get_writer
import util
import os
import pysubs2

logger = util.get_logger()


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


def save_sub_as_vtt(audio_path, sub, save_dir=''):
    if not save_dir:
        save_dir = util.get_file_dir(audio_path)
    util.mkdir(save_dir)
    vtt_writer = get_writer("vtt", save_dir)
    vtt_writer(
        sub,
        audio_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )


def save_sub_as_json(audio_path, sub, save_dir=''):
    if not save_dir:
        save_dir = util.get_file_dir(audio_path)
    json_path = os.path.join(save_dir, util.get_file_name(audio_path) + '.json')
    util.save_file(json.dumps(sub), json_path)


def save_segments_as_srt(segments, file_path):
    results = []
    for i, segment in enumerate(segments):
        start = segment['start'] / 1000.0
        end = segment['end'] / 1000.0
        text = segment.get('text', '')
        if not text:
            text = f"[{segment.get('type', '')}|{segment.get('speaker', '')}] {segment['start']}->{segment['end']}"
        obj = {'start': start, 'end': end, 'text': text}
        results.append(obj)
    util.mkdir(file_path)
    subs = pysubs2.load_from_whisper(results)
    subs.save(file_path)
