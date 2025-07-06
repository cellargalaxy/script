import util
import util_subt
import os
import json
import speaker_detect_pyannote

logger = util.get_logger()


def detect_and_save(audio_path, output_dir, auth_token):
    segments = speaker_detect_pyannote.speaker_detect(audio_path, auth_token)
    filename = util.get_file_name(audio_path)
    json_path = os.path.join(output_dir, f"{filename}.json")
    util.save_as_json(segments, json_path)
    srt_path = os.path.join(output_dir, f"{filename}.srt")
    util_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def detect_and_join(part_divide_path, part_split_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    srt_path = os.path.join(output_dir, 'speaker_detect.srt')
    if util.path_exist(json_path):
        return json_path

    split_dir = os.path.join(output_dir, 'split')
    for file in util.listdir(part_split_dir):
        file_path = os.path.join(part_split_dir, file)
        if not util.path_isfile(file_path):
            continue
        if not file_path.endswith('speech.wav'):
            continue
        detect_and_save(file_path, split_dir, auth_token)

    segments = []

    content = util.read_file(part_divide_path)
    obj = json.loads(content)
    for i, segment in enumerate(obj):
        split_path = os.path.join(split_dir, f'{i:05d}_{segment["vad_type"]}.json')
        if not util.path_exist(split_path):
            continue
        content = util.read_file(split_path)
        segs = json.loads(content)
        segs = util_subt.shift_segments_time(segs, segment['start'])
        segments.extend(segs)

    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def speaker_detect_by_manager(manager):
    logger.info("speaker_detect,enter,manager: %s", json.dumps(manager))
    part_divide_path = manager.get('part_divide_path')
    part_split_dir = manager.get('part_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_detect")
    speaker_detect_path = detect_and_join(part_divide_path, part_split_dir, output_dir, auth_token)
    manager['speaker_detect_path'] = speaker_detect_path
    logger.info("speaker_detect,leave,manager: %s", json.dumps(manager))
