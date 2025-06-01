import json
import util
import os
import math
import sub_util
from pydub import AudioSegment

logger = util.get_logger()

min_silene_duration_default = 1 * 1000
edge_duration_default = 500


def detect_audio_activity_point(audio_path, json_path):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)
    del audio
    util.exec_gc()

    content = util.read_file(json_path)
    sub = json.loads(content)
    segments = []
    for i, segment in enumerate(sub.get('segments', [])):
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        start = math.floor(segment['start'] * 1000)
        end = math.floor(segment['end'] * 1000)
        if pre_end < start:
            segments.append({"start": pre_end, "end": start, "type": "silene"})
        if start < end:
            segments.append({"start": start, "end": end, "type": "speech"})
    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[len(segments) - 1]['end']
    if pre_end < last_end:
        segments.append({"start": pre_end, "end": last_end, "type": "silene"})
    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i - 1]['end'] <= segments[i]['start']:
            continue
        mean = math.floor((segments[i - 1]['end'] + segments[i]['start']) / 2.0)
        segments[i]['start'] = mean
        segments[i - 1]['end'] = mean

    sub_util.check_segments(segments)
    logger.info("检测语音活动点,segments: %s", json.dumps(segments))
    return segments

def detect_audio_split_point(segments,
                             min_silene_duration=min_silene_duration_default,
                             edge_duration=edge_duration_default):
    for i, segment in enumerate(segments):
        if len(segments) <= 1:
            continue
        if segments[i]['type'] != 'silene':
            continue
        silene_duration = segments[i]['end'] - segments[i]['start']
        if i == 0:
            if min_silene_duration + edge_duration <= silene_duration:
                segments[i]['end'] = segments[i]['end'] - edge_duration
                segments[i + 1]['start'] = segments[i + 1]['start'] - edge_duration
            else:
                segments[i + 1]['start'] = segments[i]['start']
                segments[i]['start'] = -1
                segments[i]['end'] = -1
            continue
        if i == len(segments) - 1:
            if min_silene_duration + edge_duration <= silene_duration:
                segments[i - 1]['end'] = segments[i - 1]['end'] + edge_duration
                segments[i]['start'] = segments[i]['start'] + edge_duration
            else:
                segments[i - 1]['end'] = segments[i]['end']
                segments[i]['start'] = -1
                segments[i]['end'] = -1
            continue
        if min_silene_duration + (edge_duration * 2) <= silene_duration:
            segments[i - 1]['end'] = segments[i - 1]['end'] + edge_duration
            segments[i]['start'] = segments[i]['start'] + edge_duration
            segments[i]['end'] = segments[i]['end'] - edge_duration
            segments[i + 1]['start'] = segments[i + 1]['start'] - edge_duration
        else:
            segments[i - 1]['end'] = segments[i - 1]['end'] + math.floor(silene_duration / 2.0)
            segments[i + 1]['start'] = segments[i - 1]['end']
            segments[i]['start'] = -1
            segments[i]['end'] = -1
    ss = []
    for i, segment in enumerate(segments):
        if segments[i]['start'] < 0 or segments[i]['end'] < 0:
            continue
        ss.append(segments[i])
    segments = ss

    sub_util.check_segments(segments)
    logger.info("检测语音剪切点,segments: %s", json.dumps(segments))
    return segments

def split_audio(audio_path, json_path, output_dir, **kwargs):
    activity_segments = detect_audio_activity_point(audio_path, json_path)
    util.save_file(json.dumps(activity_segments), os.path.join(output_dir, 'meta/activity_segments.json'))
    sub_util.save_segments_as_srt(activity_segments, os.path.join(output_dir, 'meta/activity_segments.srt'))

    split_segments = detect_audio_split_point(activity_segments,
                                              min_silene_duration=kwargs.get('min_silene_duration',
                                                                             min_silene_duration_default),
                                              edge_duration=kwargs.get('edge_duration', edge_duration_default),
                                              )
    util.save_file(json.dumps(split_segments), os.path.join(output_dir, 'meta/split_segments.json'))
    sub_util.save_segments_as_srt(split_segments, os.path.join(output_dir, 'meta/split_segments.srt'))

    audio = AudioSegment.from_wav(audio_path)
    for index, segment in enumerate(split_segments):
        output_path = os.path.join(output_dir, f'{index:05d}_{segment["type"]}.wav')
        cut = audio[segment['start']:segment['end']]
        cut.export(output_path, format="wav")

def split_video_by_manager(manager):
    audio_path = manager.get('noise_reduction_audio_path')
    json_path = os.path.join(manager.get('join_sub_dir'), 'wav.json')
    output_dir = os.path.join(manager.get('output_dir'), "audio_split_segment")
    min_silene_duration = manager.get('min_silene_duration', min_silene_duration_default)
    edge_duration = manager.get('edge_duration', edge_duration_default)
    split_audio(audio_path, json_path, output_dir,
                min_silene_duration=min_silene_duration,
                edge_duration=edge_duration,
                )
    manager['audio_split_segment_dir'] = output_dir
