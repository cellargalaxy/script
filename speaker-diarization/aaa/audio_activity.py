import util
import json
import util_sub
import os
import audio_activity_silero_vad
import audio_activity_ten_vad
import audio_activity_pyannote

logger = util.get_logger()


def audio_activity(audio_path, output_dir, auth_token):
    segments = audio_activity_pyannote.audio_activity(audio_path, auth_token=auth_token)
    util.save_file(json.dumps(segments), os.path.join(output_dir, 'pyannote.json'))
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'pyannote.srt'))

    segments = audio_activity_silero_vad.audio_activity(audio_path)
    util.save_file(json.dumps(segments), os.path.join(output_dir, 'silero_vad.json'))
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'silero_vad.srt'))

    segments = audio_activity_ten_vad.audio_activity(audio_path)
    util.save_file(json.dumps(segments), os.path.join(output_dir, 'ten_vad.json'))
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'ten_vad.srt'))


def audio_activity_by_manager(manager):
    logger.info("split_video,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('noise_reduction_audio_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "audio_activity")
    audio_activity(audio_path, output_dir, auth_token)
    manager['audio_activity_dir'] = output_dir
    logger.info("split_video,leave,manager: %s", json.dumps(manager))
