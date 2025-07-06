import util
import math
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import util_subt

logger = util.get_logger()


def speaker_detect(audio_path, auth_token, min_speech_duration_ms=250):
    logger.info("说话人检测: %s", audio_path)

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
        if end - start < min_speech_duration_ms:
            continue
        segments.append({"start": start, "end": end, "speaker": speaker})

    segments = util_subt.fix_overlap_segments(segments)

    del audio
    del pipeline
    del diarization
    util.exec_gc()

    return segments
