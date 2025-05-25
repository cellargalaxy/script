import json
import whisper_timestamped as whisper
from whisperx.utils import get_writer
import util
import os
import gc
import torch
import ffprobe_util
from collections import Counter


def join_sub_json(input_dir):
    subtitle = {
        "segments": [],
        "word_segments": [],
        "language": '',
        "languages": [],
    }
    start = 0
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if not util.path_isfile(file_path):
            continue
        if '.json' in file_path or '.vtt' in file_path:
            continue
        video_duration = ffprobe_util.get_video_duration(file_path)
        json_path = os.path.join(input_dir, util.get_file_name(file_path) + '.json')
        if not util.path_exist(json_path):
            start = start + video_duration
            continue
        content = util.read_file(json_path)
        sub = json.loads(content)

        offset = 0.998  # 由于视频剪辑，字幕合并后会有偏移，进行校正
        segments = sub.get('segments', [])
        for i, segment in enumerate(segments):
            segments[i]['start'] = segments[i]['start'] + (start * offset)
            segments[i]['end'] = segments[i]['end'] + (start * offset)
            words = segments[i].get('words', [])
            for j, word in enumerate(words):
                words[j]['start'] = words[j]['start'] + (start * offset)
                words[j]['end'] = words[j]['end'] + (start * offset)
            segments[i]['words'] = words
        sub['segments'] = segments

        word_segments = sub.get('word_segments', [])
        for i, word_segment in enumerate(word_segments):
            word_segments[i]['start'] = word_segments[i]['start'] + (start * offset)
            word_segments[i]['end'] = word_segments[i]['end'] + (start * offset)
        sub['word_segments'] = word_segments

        subtitle['segments'].extend(sub['segments'])
        subtitle['word_segments'].extend(sub['word_segments'])
        language = sub.get('language', '')
        if language:
            subtitle['languages'].append(language)
        start = start + video_duration
    if len(subtitle['languages']) > 0:
        counter = Counter(subtitle['languages'])
        most_common = counter.most_common(1)
        subtitle['language'] = most_common[0][0] if most_common else ''
    return subtitle


def join_sub_and_save(video_path, input_dir, save_dir):
    subtitle = join_sub_json(input_dir)

    json_path = os.path.join(save_dir, util.get_file_name(video_path) + '.json')
    util.save_file(json.dumps(subtitle), json_path)

    vtt_writer = get_writer("vtt", save_dir)
    vtt_writer(
        subtitle,
        video_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )


def join_sub_and_save_by_manager(manager):
    video_path = manager.get('video_path')
    input_dir = manager.get('split_video_dir')
    save_dir = os.path.join(manager.get('output_dir'), "subtitle")
    join_sub_and_save(video_path, input_dir, save_dir)
    manager['subtitle_save_dir'] = save_dir
