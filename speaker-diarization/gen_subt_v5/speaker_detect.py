import util
import os
import json
import speaker_detect_pyannote

logger = util.get_logger()


def speaker_detect(segment_split_path, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    if util.path_exist(json_path):
        return json_path
    groups = speaker_detect_pyannote.speaker_detect(segment_split_path, auth_token)
    util.save_as_json(groups, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_detect,enter: %s", json.dumps(manager))
    segment_split_path = manager.get('segment_split_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_detect")
    speaker_detect_path = speaker_detect(segment_split_path, output_dir, auth_token)
    manager['speaker_detect_path'] = speaker_detect_path
    logger.info("speaker_detect,leave: %s", json.dumps(manager))
    speaker_detect_pyannote.exec_gc()
