import util
import math
import torch
from pyannote.audio import Pipeline
import numpy as np

logger = util.get_logger()

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline:
        return pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token='')
    pipeline.to(torch.device(util.get_device_type()))
    return pipeline


def pydub_pyannote(audio):
    samples = np.array(audio.get_array_of_samples())
    samples_float = samples.astype(np.float32) / audio.max_possible_amplitude
    waveform = torch.tensor(samples_float, dtype=torch.float32)
    waveform = waveform.unsqueeze(0)
    signal = {'waveform': waveform, 'sample_rate': audio.frame_rate}
    return signal


def speaker_detect(audio):
    last_end = len(audio)

    signal = pydub_pyannote(audio)
    pipeline = get_pipeline()
    diarization = pipeline(signal)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = math.floor(turn.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(turn.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "speaker": speaker})

    return segments
