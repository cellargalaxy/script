import util
import json
import util_subt
import util_vad
import os
import math
from pydub import AudioSegment

logger = util.get_logger()


def segment_divide(audio_path, segment_detect_path, output_dir, min_silene_duration_ms=500):
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
        if i == 0:
            continue
        if segments[i - 1]['vad_type'] != 'speech':
            continue
        if segments[i]['vad_type'] != 'speech':
            continue

        mean_start = math.ceil((segments[i - 1]['start'] + segments[i - 1]['end']) / 2.0)
        mean_start = max(mean_start, segments[i - 1]['end'] - 100)
        mean_end = math.floor((segments[i]['start'] + segments[i]['end']) / 2.0)
        mean_end = min(mean_end, segments[i]['start'] + 150)
        mean_probability = 1
        mean_mean = segments[i]['start']
        if mean_end - mean_start >= 50:
            cut = audio[mean_start:mean_end]
            has_silene, mean_probability, probability_ms = util_vad.has_silene_by_data(cut)
            mean_mean = mean_start + probability_ms

        right_start = segments[i]['start']
        right_end = math.floor((segments[i]['start'] + segments[i]['end']) / 2.0)
        right_end = min(right_end, segments[i]['start'] + 800)
        right_probability = 1
        right_mean = segments[i]['start']
        if right_end - right_start >= 50:
            cut = audio[right_start:right_end]
            has_silene, right_probability, probability_ms = util_vad.has_silene_by_data(cut)
            right_mean = right_start + probability_ms

        mean = segments[i]['start']
        if mean_probability < right_probability:
            mean = mean_mean
        if right_probability < mean_probability:
            mean = right_mean
        segments[i - 1]['end'] = mean
        segments[i]['start'] = mean

    segments = util_subt.gradual_segments(segments, gradual_duration_ms=min_silene_duration_ms, audio_data=audio)
    for i, segment in enumerate(segments):
        if segments[i]['vad_type'] != 'speech':
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        has_speech, max_probability, probability_ms = util_vad.has_speech_by_data(cut)
        if not has_speech:
            segments[i]['vad_type'] = 'silene'
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
