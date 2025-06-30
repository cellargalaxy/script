import util
import util_subt
import os
import json
from collections import Counter
import subt_gen_whisperx
import subt_gen_whisper_timestamped
import subt_gen_faster_whisper

logger = util.get_logger()


def gen_and_save(audio_path, output_dir, auth_token):
    subt = subt_gen_faster_whisper.subt_gen(audio_path, auth_token)
    filename = util.get_file_name(audio_path)
    json_path = os.path.join(output_dir, f"{filename}.json")
    util.save_as_json(subt, json_path)
    srt_path = os.path.join(output_dir, f"{filename}.srt")
    util_subt.save_subt_as_srt(subt, srt_path)
    return json_path


def gen_and_join(part_divide_path, part_split_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'subt_gen.json')
    srt_path = os.path.join(output_dir, 'subt_gen.srt')
    if util.path_exist(json_path):
        return json_path

    split_dir = os.path.join(output_dir, 'split')
    for file in os.listdir(part_split_dir):
        file_path = os.path.join(part_split_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        gen_and_save(file_path, split_dir, auth_token)

    subtitle = {
        "segments": [],
        "word_segments": [],
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
        subtitle['word_segments'].extend(subt['word_segments'])
        if subt['language']:
            subtitle['languages'].append(subt['language'])
    if len(subtitle['languages']) > 0:
        counter = Counter(subtitle['languages'])
        most_common = counter.most_common(1)
        subtitle['language'] = most_common[0][0] if most_common else ''

    util.save_as_json(subtitle, json_path)
    util_subt.save_subt_as_srt(subtitle, srt_path)
    return json_path


def subt_gen_by_manager(manager):
    logger.info("subt_gen,enter,manager: %s", json.dumps(manager))
    part_divide_path = manager.get('part_divide_path')
    part_split_dir = manager.get('part_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "subt_gen")
    subt_gen_path = gen_and_join(part_divide_path, part_split_dir, output_dir, auth_token)
    manager['subt_gen_path'] = subt_gen_path
    logger.info("subt_gen,leave,manager: %s", json.dumps(manager))
