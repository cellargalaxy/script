import util
import util_subt
import os
import json
from collections import Counter
import speaker_class_pyannote

logger = util.get_logger()


def class_and_save_subt(audio_path, output_dir, auth_token):
    segments = speaker_class_pyannote.speaker_class(audio_path, auth_token)
    filename = util.get_file_name(audio_path)
    json_path = os.path.join(output_dir, f"{filename}.json")
    util_subt.save_as_json(segments, json_path)
    srt_path = os.path.join(output_dir, f"{filename}.srt")
    util_subt.save_segments_as_srt(segments, srt_path)
    return srt_path


def class_and_join_subt(audio_batch_path, audio_split_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'speaker_class.json')
    if util.path_exist(json_path):
        return json_path

    split_dir = os.path.join(output_dir, 'split')
    for file in os.listdir(audio_split_dir):
        file_path = os.path.join(audio_split_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        class_and_save_subt(file_path, split_dir, auth_token)

    classes = []

    content = util.read_file(audio_batch_path)
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

    util_subt.save_as_json(subtitle, json_path)
    srt_path = os.path.join(output_dir, 'gen_subt.srt')
    util_subt.save_subt_as_srt(subtitle, srt_path)
    return json_path


def gen_subt_by_manager(manager):
    logger.info("gen_subt,enter,manager: %s", json.dumps(manager))
    audio_batch_path = manager.get('audio_batch_path')
    audio_split_dir = manager.get('audio_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "gen_subt")
    gen_subt_path = gen_and_join_subt(audio_batch_path, audio_split_dir, output_dir, auth_token)
    manager['gen_subt_path'] = gen_subt_path
    logger.info("gen_subt,leave,manager: %s", json.dumps(manager))
