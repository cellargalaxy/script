import torch
import util
import math
import json
import sub_util
from pydub import AudioSegment

logger = util.get_logger()


def audio_activity(audio_path, sample_rate=16000):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)
    del audio
    util.exec_gc()

    model, utils = torch.hub.load(repo_or_dir='model/silero-vad', model='silero_vad', source='local')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    wave = read_audio(audio_path, sampling_rate=sample_rate)
    speech_timestamps = get_speech_timestamps(
        wave, model,
        return_seconds=True,
        sampling_rate=sample_rate,
        min_silence_duration_ms=10,
        min_speech_duration_ms=1000,
        threshold=0.6,
    )

    segments = []
    for i, segment in enumerate(speech_timestamps):
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        start = math.floor(segment['start'] * 1000)
        if start < 0:
            start = 0
        end = math.ceil(segment['end'] * 1000)
        if last_end < end:
            end = last_end
        if pre_end < start:
            segments.append({"start": pre_end, "end": start, "vad_type": 'silene'})
        if start < end:
            segments.append({"start": start, "end": end, "vad_type": 'speech'})

    del model
    util.exec_gc()

    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[len(segments) - 1]['end']
    if pre_end < last_end:
        segments.append({"start": pre_end, "end": last_end, "vad_type": 'silene'})

    segments = sub_util.fix_overlap_segments(segments)
    segments = sub_util.unit_segments(segments)
    sub_util.check_segments(segments)
    logger.info("检测语音活动点,segments: %s", json.dumps(segments))
    return segments
