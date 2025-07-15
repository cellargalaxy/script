import util
import util_subt
import os
import json
from collections import Counter
import segment_detect_faster_whisper
import segment_detect_whisper_timestamped

logger = util.get_logger()


def gen_and_save(audio_path, output_dir, auth_token):
    subt = segment_detect_whisper_timestamped.segment_detect(audio_path, auth_token)
    filename = util.get_file_name(audio_path)
    json_path = os.path.join(output_dir, f"{filename}.json")
    util.save_as_json(subt, json_path)
    srt_path = os.path.join(output_dir, f"{filename}.srt")
    util_subt.save_subt_as_srt(subt, srt_path)
    return json_path


def gen_and_join(part_divide_path, part_split_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'segment_detect.json')
    srt_path = os.path.join(output_dir, 'segment_detect.srt')
    if util.path_exist(json_path):
        return json_path

    split_dir = os.path.join(output_dir, 'split')
    for file in util.listdir(part_split_dir):
        file_path = os.path.join(part_split_dir, file)
        if not util.path_isfile(file_path):
            continue
        if not file_path.endswith('speech.wav'):
            continue
        gen_and_save(file_path, split_dir, auth_token)

    subtitle = {
        "segments": [],
        "language": '',
        "languages": [],
    }

    content = util.read_file(part_divide_path)
    segments = json.loads(content)
    for i, segment in enumerate(segments):
        split_path = os.path.join(split_dir, f'{i:05d}_{segment["vad_type"]}.json')
        if not util.path_exist(split_path):
            continue
        content = util.read_file(split_path)
        subt = json.loads(content)
        subt = util_subt.shift_subt_time(subt, segment['start'])
        subtitle['segments'].extend(subt['segments'])
        if subt['language']:
            subtitle['languages'].append(subt['language'])
    if len(subtitle['languages']) > 0:
        counter = Counter(subtitle['languages'])
        most_common = counter.most_common(1)
        subtitle['language'] = most_common[0][0] if most_common else ''
    del subtitle['languages']

    util.save_as_json(subtitle, json_path)
    util_subt.save_subt_as_srt(subtitle, srt_path)
    return json_path


def exec(manager):
    logger.info("segment_detect,enter: %s", json.dumps(manager))
    part_divide_path = manager.get('part_divide_path')
    part_split_dir = manager.get('part_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "segment_detect")
    segment_detect_path = gen_and_join(part_divide_path, part_split_dir, output_dir, auth_token)
    manager['segment_detect_path'] = segment_detect_path
    logger.info("segment_detect,leave: %s", json.dumps(manager))
    segment_detect_whisper_timestamped.exec_gc()
