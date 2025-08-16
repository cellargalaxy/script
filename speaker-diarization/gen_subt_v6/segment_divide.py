import util
import json
import util_subt
import util_vad
import os
import math
from pydub import AudioSegment
import segment_divide_faster_whisper

logger = util.get_logger()


def has_speech(audio, segments):
    results = []
    for i, segment in enumerate(segments):
        if segment['end'] - segment['start'] < 50:
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        has_speech, max_probability, probability_ms = util_vad.has_speech_by_data(cut)
        if not has_speech:
            continue
        results.append(segment)
    return results


def find_valley(audio, segments):
    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            continue
        start = math.ceil((segments[i]['start'] + segments[i]['end']) / 2.0)
        start = max(start, segments[i]['end'] - 0)
        end = math.floor((segments[i + 1]['start'] + segments[i + 1]['end']) / 2.0)
        end = min(end, segments[i]['end'] + 1000)
        if end - start >= 50:
            cut = audio[start:end]
            has_silence, probability, probability_ms = util_vad.find_valley(cut)
            mean = start + probability_ms
            segments[i]['end'] = mean
            if segments[i + 1]['start'] < mean:
                segments[i + 1]['start'] = mean
    return segments


def trim_silence(audio, segments):
    for i, segment in enumerate(segments):
        cut = audio[segments[i]['start']:segments[i]['end']]
        left_ms, right_ms = util_vad.trim_silence(cut)
        if left_ms and right_ms:
            segments[i]['end'] = segments[i]['end'] - (segments[i]['end'] - segments[i]['start'] - right_ms)
            segments[i]['start'] = segments[i]['start'] + left_ms
    return segments


def split_segment(audio, segments):
    results = []
    for i, segment in enumerate(segments):
        if segment['end'] - segment['start'] < 3000:
            results.append(segment)
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        segs = segment_divide_faster_whisper.segment_divide(cut)
        if len(segs) <= 1:
            results.append(segment)
            continue
        segs = trim_silence(cut, segs)
        segs = find_valley(cut, segs)
        segs = util_subt.clipp_segments(segs, len(cut))
        segs[0]['start'] = 0
        segs[-1]['end'] = len(cut)
        segs = util_subt.shift_segments_time(segs, segment['start'])
        results.extend(segs)
    return results


def segment_divide(audio_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    subt = util.read_file_to_obj(segment_detect_path)
    segments = util_subt.subt2segments(subt)
    segments = util_subt.clipp_segments(segments, last_end)

    segments = has_speech(audio, segments)
    util.save_as_json(segments, os.path.join(output_dir, 'has_speech.json'))

    segments = find_valley(audio, segments)
    util.save_as_json(segments, os.path.join(output_dir, 'find_valley.json'))

    segments = split_segment(audio, segments)
    util.save_as_json(segments, os.path.join(output_dir, 'split_segment.json'))

    segments = trim_silence(audio, segments)
    util.save_as_json(segments, os.path.join(output_dir, 'trim_silence.json'))

    for i, segment in enumerate(segments):
        segments[i]['vad_type'] = 'speech'
    segments = util_subt.fill_segments(segments, last_end=last_end, vad_type='silence')
    segments = util_subt.clipp_segments(segments, last_end)
    util.save_as_json(segments, os.path.join(output_dir, 'fill_segments.json'))

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
