import torch
import util
import math
import json
import os
import sub_util
from pydub import AudioSegment

logger = util.get_logger()


def detect_audio_activity_point(audio_path, sampling_rate=16000):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)
    del audio
    util.exec_gc()

    model, utils = torch.hub.load(repo_or_dir='../model/silero-vad/master', model='silero_vad', source='local')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    wave = read_audio(audio_path, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wave, model,
        return_seconds=True,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=10,
        min_speech_duration_ms=1000,
        threshold=0.6,
    )
    del model
    util.exec_gc()

    segments = []
    for i, segment in enumerate(speech_timestamps):
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        start = math.ceil(segment['start'] * 1000)
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


def split_audio(audio_path, output_dir):
    activity_segments = detect_audio_activity_point(audio_path)
    util.save_file(json.dumps(activity_segments), os.path.join(output_dir, 'activity_segments.json'))
    sub_util.save_segments_as_srt(activity_segments, os.path.join(output_dir, 'activity_segments.srt'))

    segment_dir = os.path.join(output_dir, 'segment')
    util.mkdir(segment_dir)

    pre_end = 0
    segments = []
    silence = AudioSegment.silent(duration=1000)
    combine = AudioSegment.empty()
    combine += silence

    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(activity_segments):
        if segment["type"] != 'speech':
            continue

        cut_path = os.path.join(segment_dir, f'{i:05d}.wav')
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format="wav")

        pre_end += 1000
        segments.append(
            {
                "start": pre_end,
                "end": pre_end + len(cut),
                "type": segment["type"],
                "segment_path": cut_path,
            }
        )
        pre_end += len(cut)
        combine += cut + silence

    combine_wav_path = os.path.join(output_dir, 'combine.wav')
    combine.export(combine_wav_path, format="wav")
    combine_json_path = os.path.join(output_dir, 'combine.json')
    util.save_file(json.dumps(segments), combine_json_path)
    sub_util.save_segments_as_srt(segments, os.path.join(output_dir, 'combine.srt'))

    return combine_wav_path, combine_json_path, segment_dir


def split_video_by_manager(manager):
    audio_path = manager.get('noise_reduction_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "audio_split")
    combine_wav_path, combine_json_path, segment_dir = split_audio(audio_path, output_dir)
    manager['audio_combine_wav_path'] = combine_wav_path
    manager['audio_combine_json_path'] = combine_json_path
    manager['audio_segment_dir'] = segment_dir
