import util
import util_subt
import os
import json
from collections import Counter
import speaker_detect_pyannote

logger = util.get_logger()


def speaker_detect(audio_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    if util.path_exist(json_path):
        return json_path
    confidences = speaker_detect_pyannote.speaker_detect(audio_dir, auth_token)
    util.save_as_json(confidences, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_detect,enter: %s", json.dumps(manager))
    audio_dir = manager.get('segment_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_detect")
    speaker_detect_path = speaker_detect(audio_dir,   output_dir, auth_token)
    manager['speaker_detect_path'] = speaker_detect_path
    logger.info("speaker_detect,leave: %s", json.dumps(manager))
    speaker_detect_pyannote.exec_gc()
