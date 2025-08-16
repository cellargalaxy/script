import util_audio
import util
import torch
from pyannote.audio import Pipeline
import math
import util_subt

logger = util.get_logger()

pipeline = None


def get_pipeline(auth_token):
    global pipeline
    if pipeline:
        return pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    pipeline.to(torch.device(util.get_device_type()))
    return pipeline


def exec_gc():
    global pipeline
    pipeline = None
    util.exec_gc()


def segment_divide(audio, auth_token):
    last_end = len(audio)

    simple_rate, waveform = util_audio.pydub2pyannote(audio)
    pipeline = get_pipeline(auth_token)
    diarization = pipeline({"waveform": waveform, "sample_rate": simple_rate})

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = math.floor(turn.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(turn.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "speaker": speaker})

    segments = util_subt.clipp_segments(segments, last_end)
    segments = util_subt.unit_segments(segments, 'speaker')

    return segments
