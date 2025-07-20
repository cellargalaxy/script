import util
import os
import json
import speaker_detect_pyannote

logger = util.get_logger()


def speaker_detect(audio_dir, output_dir, auth_token):
    if util.path_exist(output_dir):
        return
    confidences = speaker_detect_pyannote.speaker_detect(audio_dir, auth_token)
    util.save_as_json(confidences, os.path.join(output_dir, 'pyannote.json'))
    return


def exec(manager):
    logger.info("speaker_detect,enter: %s", json.dumps(manager))
    audio_dir = manager.get('segment_split_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_detect")
    speaker_detect(audio_dir, output_dir, auth_token)
    manager['speaker_detect_dir'] = output_dir
    logger.info("speaker_detect,leave: %s", json.dumps(manager))
    speaker_detect_pyannote.exec_gc()
