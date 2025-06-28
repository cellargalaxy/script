import torch
import util
import math
import util_subt
from pydub import AudioSegment

logger = util.get_logger()


def activity_detect(audio_path, sample_rate=16000):
    logger.info("活动检测: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    wave = read_audio(audio_path, sampling_rate=sample_rate)
    speech_timestamps = get_speech_timestamps(
        wave, model,
        return_seconds=True,
        sampling_rate=sample_rate,
        min_silence_duration_ms=10,
        min_speech_duration_ms=1000,
        threshold=0.4,
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

    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[len(segments) - 1]['end']
    if pre_end < last_end:
        segments.append({"start": pre_end, "end": last_end, "vad_type": 'silene'})

    segments = util_subt.fix_overlap_segments(segments)
    segments = util_subt.unit_segments(segments, 'vad_type')
    util_subt.check_segments(segments)

    del audio
    del model
    del wave
    util.exec_gc()

    return segments
