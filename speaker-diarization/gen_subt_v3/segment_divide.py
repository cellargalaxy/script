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
    if len(segments) > 0 and last_end < segments[-1]['end']:
        segments[-1]['end'] = last_end
    gradual = []
    for i, segment in enumerate(segments):
        if segment['end'] - segment['start'] < min_speech_duration_ms:
            continue
        gradual.append(segment)
    for i, segment in enumerate(gradual):
        gradual[i]['vad_type'] = 'speech'
    segments = []
    for i, segment in enumerate(gradual):
        pre_end = 0
        if i > 0:
            pre_end = gradual[i - 1]['end']
        if pre_end < gradual[i]['start']:
            segments.append({"start": pre_end, "end": gradual[i]['start'], "vad_type": 'silene'})
        segments.append(gradual[i])
    if len(segments) > 0 and segments[-1]['end'] < last_end:
        segments.append({"start": segments[-1]['end'], "end": last_end, "vad_type": 'silene'})
    gradual = segments
    util_subt.check_coherent_segments(gradual)
    gradual = util_subt.gradual_segments(gradual, gradual_duration_ms=min_silene_duration_ms, audio_data=audio)
    util.save_file(json.dumps(gradual), json_path)
    util_subt.save_segments_as_srt(gradual, srt_path, skip_silene=True)

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
