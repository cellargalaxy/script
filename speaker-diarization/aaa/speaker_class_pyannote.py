import util
import math
import json
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

logger = util.get_logger()


def speaker_diarization(audio_path, auth_token):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    pipeline.to(torch.device(util.get_device_type()))
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = math.floor(turn.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(turn.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "speaker": speaker})
    logger.info("说话人检测,segments: %s", json.dumps(segments))
    return segments


