import json
from pyannote.audio import Pipeline
import gc
import torch
import ffprobe_util
import util
import os
import ffmpeg_util
import copy

logger = util.get_logger()


def detect_audio_activity_point(audio_path, auth_token=''):
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=auth_token)
    result = pipeline(audio_path)
    segments = []
    for segment in result.get_timeline():
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        segments.append({"start": pre_end, "end": segment.start, "type": "silene"})
        segments.append({"start": segment.start, "end": segment.end, "type": "speech"})
    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[len(segments) - 1]['end']
    all_end = ffprobe_util.get_video_duration(audio_path)
    if pre_end < all_end:
        segments.append({"start": pre_end, "end": all_end, "type": "silene"})
    del pipeline
    del result
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("检测语音活动点,segments: %s", json.dumps(segments))
    return segments


def detect_audio_split_point(audio_path, auth_token='', min_silene_duration=2, edge_duration=1,
                             speech_duration=30):  # todo
    segments = detect_audio_activity_point(audio_path, auth_token=auth_token)

    silene_points = []
    for i, segment in enumerate(segments):
        if segment['type'] != 'silene':
            continue
        point = (segment['start'] + segment['end']) / 2
        silene_points.append(point)

    ss = []
    for i, segment in enumerate(segments):
        if len(ss) == 0:
            ss.append(segments[i])
            continue
        if segments[i]['type'] != ss[len(ss) - 1]['type']:
            ss.append(segments[i])
            continue
        ss[len(ss) - 1]['end'] = segments[i]['end']
    segments = ss

    for i, segment in enumerate(segments):
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
            segments[i - 1]['end'] = segments[i - 1]['end'] + (silene_duration / 2)
            segments[i + 1]['start'] = segments[i - 1]['end']
            segments[i]['start'] = -1
            segments[i]['end'] = -1
    ss = []
    for i, segment in enumerate(segments):
        if segments[i]['start'] < 0 or segments[i]['end'] < 0:
            continue
        ss.append(segments[i])
    segments = ss

    ss = []
    for i, segment in enumerate(segments):
        if len(ss) == 0:
            ss.append(segments[i])
            continue
        if segments[i]['type'] != ss[len(ss) - 1]['type']:
            ss.append(segments[i])
            continue
        ss[len(ss) - 1]['end'] = segments[i]['end']
    segments = ss

    ss = []
    for i, segment in enumerate(segments):
        if segment['type'] != 'speech':
            ss.append(segment)
            continue
        if segment['end'] - segment['start'] < speech_duration * 2:
            ss.append(segment)
            continue
        start = segment['start']
        ends = copy.deepcopy(silene_points)
        ends.append(segment['end'])
        for j, end in enumerate(ends):
            if segment['end'] < end:
                break
            if end - start < speech_duration:
                continue
            ss.append({"start": start, "end": end, "type": "speech"})
            start = end
        if start < segment['end'] and len(ss) == 0:
            ss.append({"start": start, "end": segment['end'], "type": "speech"})
        elif start < segment['end']:
            ss[len(ss) - 1]['end'] = segment['end']
    segments = ss

    logger.info("检测语音剪切点,segments: %s", json.dumps(segments))
    return segments


def split_video(video_path, audio_path, output_dir, auth_token='', min_silene_duration=2, edge_duration=1):
    ext = util.get_file_ext(video_path)
    segments = detect_audio_split_point(audio_path, auth_token, min_silene_duration, edge_duration)
    for index, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f'{index:05d}_{segment["type"]}.{ext}')
        ffmpeg_util.cut_video(video_path, segment['start'], segment['end'], output_path)


def split_video_by_manager(manager):
    video_path = manager.get('demucs_video_path')
    audio_path = manager.get('demucs_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "split_video")
    auth_token = manager.get('auth_token', '')
    min_silene_duration = manager.get('min_silene_duration', 2)
    edge_duration = manager.get('edge_duration', 1)
    split_video(video_path, audio_path, output_dir, auth_token, min_silene_duration, edge_duration)
    manager['split_video_dir'] = output_dir
