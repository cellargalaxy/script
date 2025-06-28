import util
import math
import json
import os
import torch
from pyannote.audio import Pipeline

logger = util.get_logger()


def speaker_diarization(audio_path, auth_token=""):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    pipeline.to(torch.device(util.get_device_type()))
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = math.ceil(turn.start * 1000)
        end = math.floor(turn.end * 1000)
        segments.append({"start": start, "end": end, "speaker": speaker})
    logger.info("说话人检测,segments: %s", json.dumps(segments))
    return segments


def match_speaker(segments, start, end):
    speaker_map = {}
    for i, segment in enumerate(segments):
        if end < segment['start']:
            continue
        if segment['end'] < start:
            continue
        if segment['speaker']:
            speaker_map[segment['speaker']] = True
    if len(speaker_map) == 1:
        for key in speaker_map:
            return key
    return ""


def audio_class(audio_path, json_path, output_dir, auth_token=""):
    speaker_segments = speaker_diarization(audio_path, auth_token=auth_token)
    content = util.read_file(json_path)
    audio_segments = json.loads(content)
    for i, segment in enumerate(audio_segments):
        speaker = match_speaker(speaker_segments, segment['start'], segment['end'])
        if not speaker:
            continue
        segment_path = segment['segment_path']
        output_path = os.path.join(output_dir, speaker, util.get_file_basename(segment_path))
        util.copy_file(segment_path, output_path)


def audio_class_by_manager(manager):
    logger.info("audio_class,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('audio_combine_wav_path')
    json_path = manager.get('audio_combine_json_path')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "audio_class")
    util.delete_path(output_dir)
    audio_class(audio_path, json_path, output_dir, auth_token=auth_token)
    manager['audio_class_dir'] = output_dir
    logger.info("audio_class,leave,manager: %s", json.dumps(manager))
