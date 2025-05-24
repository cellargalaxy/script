import json
from pyannote.audio import Pipeline
import gc
import torch
import ffprobe_util
import util
import os
import ffmpeg_util

logger = util.get_logger()


def detect_video_split_point(audio_path, auth_token='', min_speech_duration=60, min_silene_duration=0.5):
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=auth_token)
    result = pipeline(audio_path)
    segments = []
    start = 0
    end = 0
    for segment in result.get_timeline():
        if end <= 0:
            end = segment.end
            continue
        silene_duration = segment.start - end
        if end - start >= min_speech_duration and min_silene_duration <= silene_duration:
            end = end + (silene_duration / 2)
            segments.append({"start": start, "end": end})
            start = end
            continue
        end = segment.end
    end = ffprobe_util.get_video_duration(audio_path)
    if end - start >= min_speech_duration:
        segments.append({"start": start, "end": end})
    elif start < end and len(segments) > 0:
        segments[len(segments) - 1]['end'] = end
    elif start < end:
        segments.append({"start": start, "end": end})
    del pipeline
    del result
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("检测视频剪切点,segments: %s", json.dumps(segments))
    return segments


def split_video(video_path, audio_path, output_dir, auth_token='', min_speech_duration=60, min_silene_duration=0.5):
    ext = util.get_file_ext(video_path)
    segments = detect_video_split_point(audio_path, auth_token, min_speech_duration, min_silene_duration)
    for index, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f'{index:03d}.{ext}')
        ffmpeg_util.cut_video(video_path, segment['start'], segment['end'], output_path)


def split_video_by_manager(manager):
    video_path = manager.get('demucs_video_path')
    audio_path = manager.get('demucs_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "split_video")
    auth_token = manager.get('auth_token', '')
    min_speech_duration = manager.get('min_speech_duration', 60)
    min_silene_duration = manager.get('min_silene_duration', 0.5)
    split_video(video_path, audio_path, output_dir, auth_token, min_speech_duration, min_silene_duration)
    manager['split_video_dir'] = output_dir
