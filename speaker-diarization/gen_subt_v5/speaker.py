import speaker_detect
import speaker_divide
import speaker_join0
import util
import json
import os

logger = util.get_logger()


def f1(segment_split_path, output_dir, auth_token):
    speaker_detect_path = speaker_detect.speaker_detect(segment_split_path, output_dir, auth_token)
    speaker_divide_path = speaker_divide.speaker_divide(segment_split_path, speaker_detect_path, output_dir)
    speaker_join_path = speaker_join.speaker_join(speaker_divide_path, output_dir)
    return speaker_join_path


def f2(speaker_join_path, output_dir, auth_token):
    speaker_detect_path = speaker_detect.speaker_detect(speaker_join_path, output_dir, auth_token, window=100000000)
    speaker_divide_path = speaker_divide.speaker_divide(speaker_join_path, speaker_detect_path, output_dir)
    speaker_join_path = speaker_join.speaker_join(speaker_divide_path, output_dir)
    return speaker_join_path


def exec(manager):
    logger.info("speaker_f1,enter: %s", json.dumps(manager))
    segment_split_path = manager.get('segment_split_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_f1")

    speaker_join_path = f1(segment_split_path, output_dir, auth_token)
    output_dir = os.path.join(manager.get('output_dir'), "speaker_f2")
    f2(speaker_join_path, output_dir, auth_token)

    logger.info("speaker_f1,leave: %s", json.dumps(manager))
    util.exec_gc()
