import util
import json
import util_subt
import os
import activity_detect_pyannote

logger = util.get_logger()


def activity_detect(audio_path, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'activity_detect.json')
    srt_path = os.path.join(output_dir, 'activity_detect.srt')
    if util.path_exist(json_path):
        return json_path

    # segments = activity_detect_ten_vad.activity_detect(audio_path)
    # util.save_file(json.dumps(segments), os.path.join(output_dir, 'ten_vad.json'))
    # util_subt.save_segments_as_srt(segments, os.path.join(output_dir, 'ten_vad.srt'))

    segments = activity_detect_pyannote.activity_detect(audio_path, auth_token=auth_token)
    util.save_file(json.dumps(segments), json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silene=True)
    return json_path


def activity_detect_by_manager(manager):
    logger.info("activity_detect,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('merge_audio_channel_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "activity_detect")
    activity_detect_path = activity_detect(audio_path, output_dir, auth_token)
    manager['activity_detect_path'] = activity_detect_path
    logger.info("activity_detect,leave,manager: %s", json.dumps(manager))
