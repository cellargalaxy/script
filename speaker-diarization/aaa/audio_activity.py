import util
import json
import util_sub
import os
import audio_activity_pyannote

logger = util.get_logger()


def audio_activity(audio_path, output_dir, auth_token):
    audio_activity_path = os.path.join(output_dir, 'audio_activity.json')
    if util.path_exist(audio_activity_path):
        return audio_activity_path
    segments = audio_activity_pyannote.audio_activity(audio_path, auth_token=auth_token)
    util.save_file(json.dumps(segments), audio_activity_path)
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'audio_activity.srt'))
    return audio_activity_path


def audio_activity_by_manager(manager):
    logger.info("audio_activity,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('merge_audio_channel_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "audio_activity")
    audio_activity_path = audio_activity(audio_path, output_dir, auth_token)
    manager['audio_activity_path'] = audio_activity_path
    logger.info("audio_activity,leave,manager: %s", json.dumps(manager))
