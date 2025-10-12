import util
import util_subt
import os
from collections import Counter
import segment_detect_faster_whisper

logger = util.get_logger()


def gen_and_save(segment, auth_token):
    wav_path = segment['wav_path']
    json_path = segment['json_path']
    srt_path = segment['srt_path']
    if not wav_path.endswith('speech.wav'):
        return
    if util.path_exist(json_path):
        return json_path
    subt = segment_detect_faster_whisper.segment_detect(wav_path, auth_token)
    util.save_as_json(subt, json_path)
    util_subt.save_subt_as_srt(subt, srt_path)


def gen_and_join(part_split_path, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'segment_detect.json')
    srt_path = os.path.join(output_dir, 'segment_detect.srt')
    if util.path_exist(json_path):
        return json_path

    segments = util.read_file_to_obj(part_split_path)

    split_dir = os.path.join(output_dir, 'split')
    for i, segment in enumerate(segments):
        segments[i]['json_path'] = os.path.join(split_dir, f"{segments[i]['file_name']}.json")
        segments[i]['srt_path'] = os.path.join(split_dir, f"{segments[i]['file_name']}.srt")

    for i, segment in enumerate(segments):
        gen_and_save(segment, auth_token)

    subtitle = {
        "segments": [],
        "language": '',
        "languages": [],
    }

    for i, segment in enumerate(segments):
        split_path = segment['json_path']
        if not util.path_exist(split_path):
            continue
        subt = util.read_file_to_obj(split_path)
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
    logger.info("segment_detect,enter: %s", util.json_dumps(manager))
    part_split_path = manager.get('part_split_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "segment_detect")
    json_path = gen_and_join(part_split_path, output_dir, auth_token)
    manager['segment_detect_path'] = json_path
    logger.info("segment_detect,leave: %s", util.json_dumps(manager))
    segment_detect_faster_whisper.exec_gc()
