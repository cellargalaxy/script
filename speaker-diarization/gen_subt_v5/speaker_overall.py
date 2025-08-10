import speaker_detect
import speaker_divide
import speaker_join
import util
import json
import os

logger = util.get_logger()


def speaker_overall(speak_path, output_dir, auth_token):
    speaker_detect_path = speaker_detect.speaker_detect(speak_path, output_dir, auth_token, window=100000000)
    speaker_divide_path = speaker_divide.speaker_divide(speak_path, speaker_detect_path, output_dir)
    speaker_join_path = speaker_join.speaker_join(speaker_divide_path, output_dir)
    return speaker_join_path


def exec(manager):
    logger.info("speaker_overall,enter: %s", json.dumps(manager))
    speaker_join_path = manager.get('speaker_join_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_overall")
    speaker_overall_path = speaker_overall(speaker_join_path, output_dir, auth_token)
    manager['speaker_overall_path'] = speaker_overall_path
    logger.info("speaker_overall,leave: %s", json.dumps(manager))
    util.exec_gc()
