import util
import util_subt
import os
import part_detect_pyannote

logger = util.get_logger()


def part_detect(audio_path, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'part_detect.json')
    srt_path = os.path.join(output_dir, 'part_detect.srt')
    if util.path_exist(json_path):
        return json_path
    segments = part_detect_pyannote.part_detect(audio_path, auth_token=auth_token)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silence=True)
    return json_path


def exec(manager):
    logger.info("part_detect,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "part_detect")
    part_detect_path = part_detect(audio_path, output_dir, auth_token)
    manager['part_detect_path'] = part_detect_path
    logger.info("part_detect,leave: %s", util.json_dumps(manager))
    util.exec_gc()
