import util
import json
import util_subt
import util_vad
import os
import math
from pydub import AudioSegment

logger = util.get_logger()


def segment_divide(audio_path, segment_detect_path, output_dir, min_silence_duration_ms=500):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    subt = util.read_file_to_obj(segment_detect_path)
    segments = util_subt.subt2segments(subt)
    for i, segment in enumerate(segments):
        segments[i]['vad_type'] = 'speech'
    segments = util_subt.fill_segments(segments, last_end=last_end, vad_type='silence')
    segments = util_subt.clipp_segments(segments, last_end)

    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            continue
        if segments[i]['vad_type'] != 'speech':
            continue
        start = math.ceil((segments[i]['start'] + segments[i]['end']) / 2.0)
        start = max(start, segments[i]['end'] - 0)
        end = math.floor((segments[i + 1]['start'] + segments[i + 1]['end']) / 2.0)
        end = min(end, segments[i + 1]['start'] + 1000)
        if end - start >= 50:
            cut = audio[start:end]
            has_silence, probability, probability_ms = util_vad.find_valley(cut)
            mean = start + probability_ms
            segments[i]['end'] = mean
            segments[i + 1]['start'] = mean

    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] != 'speech':
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        left_ms, right_ms = util_vad.trim_silence(cut)
        segments[i]['end'] = segments[i]['end'] - (segments[i]['end'] - segments[i]['start'] - right_ms)
        segments[i]['start'] = segments[i]['start'] + left_ms
    segments = util_subt.fill_segments(segments, last_end=last_end, vad_type='silence')
    segments = util_subt.unit_segments(segments, 'vad_type', type_value='silence')

    segments = util_subt.gradual_segments(segments, gradual_duration_ms=min_silence_duration_ms, audio_data=audio)
    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] != 'speech':
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        has_speech, max_probability, probability_ms = util_vad.has_speech_by_data(cut)
        if not has_speech:
            segments[i]['vad_type'] = 'silence'
    segments = util_subt.unit_segments(segments, 'vad_type', type_value='silence')

    for i, segment in enumerate(segments):
        segments[i]['file_name'] = f"{i:05d}_{segments[i]['vad_type']}"

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silence=True)
    return json_path


def exec(manager):
    logger.info("segment_divide,enter: %s", json.dumps(manager))
    audio_path = manager.get('audio_path')
    segment_detect_path = manager.get('segment_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    segment_divide_path = segment_divide(audio_path, segment_detect_path, output_dir)
    manager['segment_divide_path'] = segment_divide_path
    logger.info("segment_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
