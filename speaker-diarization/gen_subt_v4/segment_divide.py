import util
import json
import util_subt
import os
from pydub import AudioSegment

logger = util.get_logger()


def segment_divide(audio_path, segment_detect_path, output_dir,
                   min_silene_duration_ms=500,
                   min_speech_duration_ms=1000):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    content = util.read_file(segment_detect_path)
    subt = json.loads(content)
    segments = util_subt.subt2segments(subt)
    for i, segment in enumerate(segments):
        segments[i]['vad_type'] = 'speech'
    segments = util_subt.fill_segments(segments, last_end=last_end, vad_type='silene')
    segments = util_subt.clipp_segments(segments, last_end)
    for i, segment in enumerate(segments):
        if segments[i]['end'] - segments[i]['start'] < min_speech_duration_ms:
            segments[i]['too_mini_speech'] = True
    segments = util_subt.gradual_segments(segments, gradual_duration_ms=min_silene_duration_ms, audio_data=audio)
    for i, segment in enumerate(segments):
        if segments[i].get('too_mini_speech', False):
            segments[i]['vad_type'] = 'silene'
            del segments[i]['too_mini_speech']
    segments = util_subt.unit_segments(segments, 'vad_type', type_value='silene')

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silene=True)

    return json_path


def exec(manager):
    logger.info("segment_divide,enter: %s", json.dumps(manager))
    audio_path = manager.get('merge_channel_path')
    segment_detect_path = manager.get('segment_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    segment_divide_path = segment_divide(audio_path, segment_detect_path, output_dir)
    manager['segment_divide_path'] = segment_divide_path
    logger.info("segment_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
